# Movie App Backend

Backend API cho ứng dụng phân tích cảm xúc từ khuôn mặt trong phim.

## Tính năng

- Phân tích cảm xúc từ hình ảnh khuôn mặt
- API RESTful với Flask
- Hỗ trợ CORS cho frontend
- Sử dụng PyTorch model đã được fine-tuned

## Cài đặt

1. Tạo virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# hoặc
source .venv/bin/activate  # Linux/Mac
```

2. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

3. Đảm bảo file model `face_classifier_finetuned.pth` có trong thư mục gốc của project.

## Chạy ứng dụng

```bash
python app.py
```

Server sẽ chạy tại `http://localhost:5000`

## API Endpoints

### POST /predict

Phân tích cảm xúc từ hình ảnh

**Request:**

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response:**

```json
{
  "prediction": "happy",
  "confidence": 0.95,
  "success": true
}
```

### GET /health

Kiểm tra trạng thái server

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

### GET /labels

Lấy danh sách các nhãn cảm xúc

**Response:**

```json
{
  "0": "angry",
  "1": "disgust",
  "2": "fear",
  "3": "happy",
  "4": "neutral",
  "5": "sad",
  "6": "surprise"
}
```

## Các cảm xúc được hỗ trợ

- angry (tức giận)
- disgust (ghê tởm)
- fear (sợ hãi)
- happy (vui vẻ)
- neutral (trung tính)
- sad (buồn bã)
- surprise (ngạc nhiên)
