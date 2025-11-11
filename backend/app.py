import io
import json
import os
from typing import List, Dict, Any, Optional
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import httpx

# Optional heavy deps imported lazily on startup to keep import cost lower
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn


APP = FastAPI(title="Movie App Recognition API")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
LOGGER = logging.getLogger("recognition")

APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Match notebook training: 112x112 and mean/std 0.5
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
LABELS: Dict[int, Dict[str, Any]] = {}
MOVIE_API_BASE = os.getenv("MOVIE_API_BASE", "http://localhost:8080/api")


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_labels() -> None:
    global LABELS
    # Prefer label_encoder.pkl from training to ensure class order alignment
    enc_paths = [
        os.path.join(_project_root(), "label_encoder.pkl"),
        os.path.join(os.path.dirname(__file__), "label_encoder.pkl"),
    ]
    for p in enc_paths:
        if os.path.exists(p):
            try:
                import pickle

                with open(p, "rb") as f:
                    name_to_idx = pickle.load(f)
                # Build index -> name list in order of class index
                by_idx = {int(v): str(k) for k, v in name_to_idx.items()}
                norm: Dict[int, Dict[str, Any]] = {}
                for idx in sorted(by_idx.keys()):
                    raw_name = by_idx[idx]
                    # Normalize name: replace _ with space, strip trailing underscores
                    pretty = raw_name.replace("_", " ").strip().rstrip("_").strip()
                    # Fix special cases
                    if pretty.lower() == "ai pacino":
                        pretty = "Al Pacino"
                    elif pretty.lower() == "gwyneth paltrow":
                        pretty = "Gwyneth Paltrow"
                    # Capitalize properly (title case)
                    words = pretty.split()
                    if len(words) > 0:
                        # Keep hyphenated names as-is (e.g., "Gordon-Levitt")
                        pretty = " ".join(word.title() if "-" not in word else word for word in words)
                    norm[idx] = {"id": idx, "name": pretty}
                LABELS = norm
                LOGGER.info("Loaded labels from %s (classes=%s)", p, len(LABELS))
                return
            except Exception as exc:
                LOGGER.warning("Failed to load label encoder %s: %s", p, exc)

    labels_path = os.path.join(os.path.dirname(__file__), "labels.json")
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Normalize into {index: {id, name}}
            norm: Dict[int, Dict[str, Any]] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        idx = int(k)
                    except Exception:
                        continue
                    if isinstance(v, dict):
                        actor_id = v.get("id", v.get("name", str(idx)))
                        norm[idx] = {"id": actor_id, "name": v.get("name", str(actor_id))}
                    else:
                        norm[idx] = {"id": v, "name": str(v)}
            elif isinstance(raw, list):
                for idx, name in enumerate(raw):
                    norm[idx] = {"id": name, "name": str(name)}
            LABELS = norm
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Failed to load labels.json: %s, using fallback", exc)
            # Fallback: create basic labels from known actors (matching label_encoder.pkl order)
            LABELS = {
                0: {"id": 0, "name": "Al Pacino"},
                1: {"id": 1, "name": "Adam Driver"},
                2: {"id": 2, "name": "Adrien Brody"},
                3: {"id": 3, "name": "Amy Poehler"},
                4: {"id": 4, "name": "Angelina Jolie"},
                5: {"id": 5, "name": "Anne Hathaway"},
                6: {"id": 6, "name": "Cameron Diaz"},
                7: {"id": 7, "name": "Cate Blanchett"},
                8: {"id": 8, "name": "Channing Tatum"},
                9: {"id": 9, "name": "Charlize Theron"},
                10: {"id": 10, "name": "Colin Farrell"},
                11: {"id": 11, "name": "Courteney Cox"},
                12: {"id": 12, "name": "Daniel Radcliffe"},
                13: {"id": 13, "name": "Drew Barrymore"},
                14: {"id": 14, "name": "Dwayne Johnson"},
                15: {"id": 15, "name": "Gwyneth Paltrow"},
                16: {"id": 16, "name": "Helen Mirren"},
                17: {"id": 17, "name": "Jennifer Lawrence"},
                18: {"id": 18, "name": "Jeremy Renner"},
                19: {"id": 19, "name": "Joseph Gordon-Levitt"},
                20: {"id": 20, "name": "Julia Roberts"},
                21: {"id": 21, "name": "Julianne Moore"},
                22: {"id": 22, "name": "Viola Davis"},
                23: {"id": 23, "name": "Willem"},
                24: {"id": 24, "name": "Zendaya"},
                25: {"id": 25, "name": "Zoe Saldana"},
            }
    else:
        # Fallback: create basic labels from known actors (matching label_encoder.pkl order)
        LABELS = {
            0: {"id": 0, "name": "Al Pacino"},
            1: {"id": 1, "name": "Adam Driver"},
            2: {"id": 2, "name": "Adrien Brody"},
            3: {"id": 3, "name": "Amy Poehler"},
            4: {"id": 4, "name": "Angelina Jolie"},
            5: {"id": 5, "name": "Anne Hathaway"},
            6: {"id": 6, "name": "Cameron Diaz"},
            7: {"id": 7, "name": "Cate Blanchett"},
            8: {"id": 8, "name": "Channing Tatum"},
            9: {"id": 9, "name": "Charlize Theron"},
            10: {"id": 10, "name": "Colin Farrell"},
            11: {"id": 11, "name": "Courteney Cox"},
            12: {"id": 12, "name": "Daniel Radcliffe"},
            13: {"id": 13, "name": "Drew Barrymore"},
            14: {"id": 14, "name": "Dwayne Johnson"},
            15: {"id": 15, "name": "Gwyneth Paltrow"},
            16: {"id": 16, "name": "Helen Mirren"},
            17: {"id": 17, "name": "Jennifer Lawrence"},
            18: {"id": 18, "name": "Jeremy Renner"},
            19: {"id": 19, "name": "Joseph Gordon-Levitt"},
            20: {"id": 20, "name": "Julia Roberts"},
            21: {"id": 21, "name": "Julianne Moore"},
            22: {"id": 22, "name": "Viola Davis"},
            23: {"id": 23, "name": "Willem"},
            24: {"id": 24, "name": "Zendaya"},
            25: {"id": 25, "name": "Zoe Saldana"},
        }
    LOGGER.info("Loaded %s labels", len(LABELS))


def load_model() -> None:
    global MODEL
    if MODEL is not None:
        LOGGER.info("Model already loaded, skipping reload")
        return
    
    # Ensure labels are loaded first
    if not LABELS:
        load_labels()
    
    ckpt_path = os.path.join(_project_root(), "face_classifier_finetuned.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # Prefer TorchScript for portability
    try:
        model = torch.jit.load(ckpt_path, map_location=DEVICE)
        model.eval()
        
        # Test model output shape
        test_input = torch.randn(1, 3, 112, 112).to(DEVICE)
        with torch.no_grad():
            test_output = model(test_input)
            if isinstance(test_output, (list, tuple)):
                test_output = test_output[0]
            num_model_classes = test_output.shape[1]
            num_labels = len(LABELS)
            LOGGER.info("TorchScript model test: output shape=%s, model classes=%s, labels=%s",
                       tuple(test_output.shape), num_model_classes, num_labels)
            if num_model_classes != num_labels:
                LOGGER.error(
                    "CRITICAL: Model has %s classes but labels.json has %s classes! "
                    "Predictions will be incorrect. Please ensure model and labels match.",
                    num_model_classes, num_labels
                )
        
        MODEL = model
        LOGGER.info("Loaded TorchScript model from %s on %s", ckpt_path, DEVICE)
        return
    except Exception as exc:
        LOGGER.warning("Failed to load TorchScript model: %s", exc)
        pass

    # Fallback: attempt to load a plain state_dict into ResNet18 + custom head
    try:
        state = torch.load(ckpt_path, map_location=DEVICE)
        num_classes = max(1, len(LABELS) or 1)
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features, num_classes),
        )
        sd = state.get("state_dict", state)
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        LOGGER.info(
            "Loaded state_dict into ResNet18 (classes=%s) missing=%s unexpected=%s",
            num_classes,
            len(missing),
            len(unexpected),
        )
        backbone.to(DEVICE)
        backbone.eval()
        MODEL = backbone
        return
    except Exception as exc:
        LOGGER.warning("ResNet18 load failed, trying FaceCNN_Improved: %s", exc)

    # Second fallback: custom CNN architecture used in notebook
    try:
        class FaceCNN_Improved(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm2d(128)
                self.pool = nn.MaxPool2d(2, 2)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
                self.gap = nn.AdaptiveAvgPool2d((1, 1))
                self.fc1 = nn.Linear(128, 256)
                self.fc2 = nn.Linear(256, num_classes)

            def forward(self, x):
                x = self.pool(self.relu(self.bn1(self.conv1(x))))
                x = self.pool(self.relu(self.bn2(self.conv2(x))))
                x = self.pool(self.relu(self.bn3(self.conv3(x))))
                x = self.gap(x)
                x = torch.flatten(x, 1)
                x = self.dropout(self.relu(self.fc1(x)))
                return self.fc2(x)

        num_classes = max(1, len(LABELS) or 1)
        cnn = FaceCNN_Improved(num_classes=num_classes)
        sd = torch.load(ckpt_path, map_location=DEVICE)
        sd = sd.get("state_dict", sd)
        missing, unexpected = cnn.load_state_dict(sd, strict=False)
        LOGGER.info(
            "Loaded state_dict into FaceCNN_Improved (classes=%s) missing=%s unexpected=%s",
            num_classes,
            len(missing),
            len(unexpected),
        )
        cnn.to(DEVICE)
        cnn.eval()
        MODEL = cnn
        return
    except Exception as exc2:
        LOGGER.exception("Failed to load model with fallbacks: %s", exc2)
        raise RuntimeError(
            "Failed to load model. Provide TorchScript .pth or matching state_dict for ResNet18/FaceCNN_Improved."
        ) from exc2


@APP.on_event("startup")
def _startup() -> None:
    load_labels()
    try:
        load_model()
    except Exception:
        # Defer model load failure; endpoint will use a graceful fallback
        pass


def _predict_topk(img: Image.Image, topk: int) -> List[Dict[str, Any]]:
    # Try real model inference; if unavailable, fall back to label-based results
    try:
        if MODEL is None:
            raise RuntimeError("Model is not loaded")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # [1, 3, 112, 112]
        with torch.no_grad():
            logits = MODEL(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("Model did not return a tensor output")
            
            # Validate number of classes
            num_model_classes = logits.shape[1]
            num_labels = len(LABELS)
            if num_model_classes != num_labels:
                LOGGER.warning(
                    "Mismatch: Model has %s classes but labels.json has %s classes. "
                    "This may cause incorrect predictions!",
                    num_model_classes,
                    num_labels,
                )
            
            LOGGER.info("Logits shape: %s, Model classes: %s, Labels: %s", 
                       tuple(logits.shape), num_model_classes, num_labels)
            
            probs = torch.softmax(logits, dim=1)
            values, indices = torch.topk(
                probs, k=min(topk, probs.shape[1]), dim=1
            )
        values = values.squeeze(0).cpu().numpy().tolist()
        indices = indices.squeeze(0).cpu().numpy().tolist()

        results: List[Dict[str, Any]] = []
        for score, idx in zip(values, indices):
            idx_int = int(idx)
            label_meta = LABELS.get(idx_int, {"id": idx_int, "name": f"Unknown_{idx_int}"})
            predicted_name = label_meta.get("name", str(idx_int))
            LOGGER.debug("Predicted: index=%s, name=%s, score=%.4f", idx_int, predicted_name, float(score))
            results.append(
                {
                    "id": label_meta.get("id", idx_int),
                    "name": predicted_name,
                    "score": float(score),
                }
            )
        return results
    except Exception as exc:
        LOGGER.exception("Inference failed: %s", exc)
        # Disable fallback to avoid returning actors not in database
        return []


async def _find_actor_in_db_by_name(name: str) -> Optional[Dict[str, Any]]:
    # Query main API and attempt exact name match (case-insensitive)
    params = {"q": name, "page": 0, "size": 5}
    url = f"{MOVIE_API_BASE}/actors"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return None

    items: List[Dict[str, Any]] = []
    if isinstance(data, dict) and isinstance(data.get("content"), list):
        items = data["content"]
    elif isinstance(data, list):
        items = data

    # Prefer exact, otherwise first close match
    lower = name.strip().lower()
    for it in items:
        n = str(it.get("name", "")).strip().lower()
        if n == lower:
            return it
    return items[0] if items else None


@APP.get("/api/debug/model-info")
@APP.get("/debug/model-info")
async def get_model_info():
    """Debug endpoint to check model and labels status"""
    model_info = {
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "labels_count": len(LABELS),
        "labels": {str(k): v.get("name") for k, v in LABELS.items()},
    }
    
    if MODEL is not None:
        try:
            # Test model output
            test_input = torch.randn(1, 3, 112, 112).to(DEVICE)
            with torch.no_grad():
                test_output = MODEL(test_input)
                if isinstance(test_output, (list, tuple)):
                    test_output = test_output[0]
                model_info["model_output_shape"] = list(test_output.shape)
                model_info["model_classes"] = test_output.shape[1]
                model_info["classes_match"] = test_output.shape[1] == len(LABELS)
        except Exception as exc:
            model_info["model_test_error"] = str(exc)
    
    return model_info


@APP.post("/api/debug/reload")
@APP.post("/debug/reload")
async def reload_model():
    """Reload model and labels"""
    global MODEL, LABELS
    MODEL = None
    LABELS = {}
    try:
        load_labels()
        load_model()
        return {"status": "success", "labels_count": len(LABELS), "model_loaded": MODEL is not None}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@APP.post("/api/actors/recognize")
@APP.post("/actors/recognize")  # alias for frontend proxy pointing to /ai
async def recognize(
    image: UploadFile = File(...),
    topK: int = Form(12),
    debug: int = Form(0),
    minScore: float = Form(0.3),  # Chỉ hiển thị diễn viên có điểm >= 30%
    maxResults: int = Form(3),    # Tối đa 3 kết quả
):
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image content type")
    try:
        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image")

    try:
        preds = _predict_topk(img, topK)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Debug: log raw predictions with more detail
    LOGGER.info(
        "Recognize request: topK=%s, total_labels=%s, preds=%s",
        topK,
        len(LABELS),
        [
            {"name": p.get("name"), "score": round(float(p.get("score", 0)), 4), "id": p.get("id")}
            for p in (preds or [])[:5]  # Log top 5 only
        ],
    )

    # Filter predictions against DB actors; return only existing actors
    filtered: List[Dict[str, Any]] = []
    seen_ids = set()
    for p in preds:
        # Chỉ lấy diễn viên có điểm số >= minScore
        if float(p.get("score", 0.0)) < float(minScore):
            continue
        actor = await _find_actor_in_db_by_name(str(p.get("name", "")))
        if not actor:
            continue
        actor_id = actor.get("id")
        if actor_id in seen_ids:
            continue
        seen_ids.add(actor_id)
        # Merge known fields; keep score for potential UI usage
        filtered.append({
            "id": actor.get("id"),
            "name": actor.get("name"),
            "imageUrl": actor.get("imageUrl") or actor.get("image") or actor.get("avatarUrl"),
            "movieCount": actor.get("movieCount"),
            "score": p.get("score"),
        })
    
    # Sắp xếp theo điểm số giảm dần và giới hạn số kết quả
    filtered.sort(key=lambda a: float(a.get("score", 0.0)), reverse=True)
    if maxResults > 0:
        filtered = filtered[:maxResults]

    LOGGER.info(
        "Filtered against DB: matched=%s -> %s",
        len(preds or []),
        [
            {"id": a.get("id"), "name": a.get("name"), "score": a.get("score")}
            for a in filtered
        ],
    )

    if debug:
        return {
            "content": filtered,
            "debug": {
                "labels_count": len(LABELS),
                "device": str(DEVICE),
                "preds": preds,
            },
        }

    return {"content": filtered}


def create_app() -> FastAPI:
    return APP


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:APP", host="0.0.0.0", port=8000, reload=True)


