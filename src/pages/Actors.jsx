import { useEffect, useRef, useState } from "react";
import {
  Box,
  Container,
  Grid,
  InputBase,
  Paper,
  Typography,
  IconButton,
  Tooltip,
  Chip,
  Pagination,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  listActors,
  recognizeActorsByImage,
  searchActorsByNames,
} from "../api/actors";
import Header from "../components/Header";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import CameraAltIcon from "@mui/icons-material/CameraAlt";

const PageContainer = styled(Box)(() => ({
  minHeight: "100vh",
  background: "linear-gradient(180deg, #0b1220 0%, #0a0f1a 50%, #0b0d13 100%)",
  color: "#fff",
  paddingTop: 96,
}));

const SearchBar = styled(Paper)(() => ({
  display: "flex",
  alignItems: "center",
  padding: "10px 16px",
  background: "rgba(255,255,255,0.08)",
  border: "1px solid rgba(255,255,255,0.14)",
  borderRadius: 28,
  backdropFilter: "blur(8px)",
}));

const ActorCard = styled(Box)(() => ({
  background: "rgba(255,255,255,0.05)",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 16,
  overflow: "hidden",
  cursor: "pointer",
  transition: "all .28s ease",
  backdropFilter: "blur(10px)",
  position: "relative",
  "&:hover": {
    transform: "translateY(-8px)",
    boxShadow: "0 20px 40px rgba(0,0,0,0.6)",
    borderColor: "rgba(255,215,0,0.45)",
  },
}));

export default function Actors() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalElements, setTotalElements] = useState(0);
  const pageRef = useRef(0);
  const fileInputRef = useRef(null);
  const [imageSearching, setImageSearching] = useState(false);
  const [imagePreviewUrl, setImagePreviewUrl] = useState("");
  const [imageName, setImageName] = useState("");
  const q = params.get("q") || "";
  const itemsPerPage = 20;

  const debouncedQuery = useDebounce(q, 300);
  const prevQueryRef = useRef(debouncedQuery);

  // Reset to page 1 when search query changes
  useEffect(() => {
    if (!imageSearching && prevQueryRef.current !== debouncedQuery) {
      prevQueryRef.current = debouncedQuery;
      setCurrentPage(1);
      // Don't load here, let the next useEffect handle it
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedQuery]);

  // Load data when page changes (including when reset to page 1)
  useEffect(() => {
    if (!imageSearching) {
      const pageToLoad = currentPage - 1; // Convert 1-based to 0-based
      load(pageToLoad);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPage]);

  async function load(page) {
    try {
      setLoading(true);
      setError("");
      const data = await listActors({
        q: debouncedQuery,
        page,
        size: itemsPerPage,
      });
      const list = Array.isArray(data?.content)
        ? data.content
        : Array.isArray(data)
        ? data
        : [];
      setItems(list);
      pageRef.current = page;

      // Update pagination info
      if (data?.totalPages != null) {
        setTotalPages(data.totalPages);
      } else if (data?.totalElements != null) {
        setTotalPages(Math.ceil(data.totalElements / itemsPerPage));
      } else {
        // Estimate based on current page and items count
        setTotalPages(list.length < itemsPerPage ? page + 1 : page + 2);
      }

      if (data?.totalElements != null) {
        setTotalElements(data.totalElements);
      } else {
        setTotalElements(list.length);
      }
    } catch (e) {
      setError(e?.message || "Không thể tải danh sách diễn viên");
    } finally {
      setLoading(false);
    }
  }

  const handleOpen = (id) => navigate(`/actor/${id}`);

  function handlePickImage() {
    fileInputRef.current?.click();
  }

  function handleClearImageSearch() {
    setImageSearching(false);
    setImagePreviewUrl("");
    setImageName("");
    setCurrentPage(1);
    // Re-run text search
    load(0);
  }

  async function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError("");
    setImageSearching(true);
    setLoading(true);
    setImageName(file.name);
    try {
      // preview
      const url = URL.createObjectURL(file);
      setImagePreviewUrl(url);
      // call recognition API
      const data = await recognizeActorsByImage(file, { topK: 24 });
      const preds = Array.isArray(data?.content)
        ? data.content
        : Array.isArray(data)
        ? data
        : [];
      const names = preds
        .map((p) => String(p?.name || "").trim())
        .filter(Boolean);
      // eslint-disable-next-line no-console
      console.log("[AI] predicted names:", names);
      const dbActors = await searchActorsByNames(names);
      // eslint-disable-next-line no-console
      console.log("[DB] matched actors:", dbActors);
      setItems(dbActors);
      pageRef.current = 0;
      setCurrentPage(1);
      setTotalPages(1);
      setTotalElements(dbActors.length);
    } catch (e) {
      setError(e?.message || "Không thể nhận diện diễn viên từ ảnh");
    } finally {
      setLoading(false);
      // reset input value to allow same-file reselect
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  return (
    <PageContainer>
      <Header />
      <Container maxWidth="xl" sx={{ pt: 2, pb: 6 }}>
        <Box sx={{ mb: 3 }}>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 800,
              color: "#fff",
              mb: 2,
              textShadow: "0 2px 8px rgba(0,0,0,0.8)",
            }}
          >
            Diễn viên
          </Typography>
          <SearchBar>
            <InputBase
              placeholder={
                imageSearching ? "Đang tìm theo ảnh…" : "Tìm diễn viên…"
              }
              disabled={imageSearching}
              fullWidth
              value={q}
              onChange={(e) => setParams({ q: e.target.value })}
            />
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            <Tooltip title="Tìm bằng ảnh">
              <IconButton onClick={handlePickImage} sx={{ ml: 1 }}>
                <CameraAltIcon sx={{ color: "#ffd700" }} />
              </IconButton>
            </Tooltip>
          </SearchBar>

          {imageSearching && (
            <Box
              sx={{ mt: 1.5, display: "flex", alignItems: "center", gap: 2 }}
            >
              {imagePreviewUrl && (
                <Box
                  component="img"
                  src={imagePreviewUrl}
                  alt={imageName || "uploaded"}
                  sx={{
                    width: 64,
                    height: 64,
                    objectFit: "cover",
                    borderRadius: 1,
                    border: "1px solid rgba(255,255,255,0.2)",
                  }}
                />
              )}
              <Chip
                label={
                  imageName ? `Tìm theo ảnh: ${imageName}` : "Đang tìm theo ảnh"
                }
                onDelete={handleClearImageSearch}
                sx={{
                  background: "rgba(255,215,0,0.15)",
                  color: "#ffd700",
                  border: "1px solid rgba(255,215,0,0.35)",
                  fontWeight: 600,
                }}
              />
            </Box>
          )}
        </Box>

        {error && (
          <Typography sx={{ color: "#ff6b6b", mt: 2 }}>{error}</Typography>
        )}

        {loading && items.length === 0 && (
          <Typography sx={{ color: "#bbb", mt: 2 }}>Đang tải…</Typography>
        )}

        <Grid container spacing={3} sx={{ mt: 1 }}>
          {items.map((a) => (
            <Grid item xs={6} sm={4} md={3} lg={2.4} xl={2} key={a.id}>
              <ActorCard onClick={() => handleOpen(a.id)}>
                <Box
                  component="img"
                  src={a.imageUrl || a.image || a.avatarUrl}
                  alt={a.name}
                  onError={(e) => {
                    e.currentTarget.src = `https://via.placeholder.com/400x500/333/fff?text=${encodeURIComponent(
                      a.name || "Actor"
                    )}`;
                  }}
                  sx={{
                    width: "100%",
                    height: 320,
                    objectFit: "cover",
                    display: "block",
                  }}
                />

                {/* overlay actions */}
                <Box
                  sx={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 1.5,
                    background:
                      "linear-gradient(180deg, rgba(0,0,0,0) 40%, rgba(0,0,0,0.6) 100%)",
                    opacity: 0,
                    transition: "opacity .25s ease",
                    "&:hover": { opacity: 1 },
                  }}
                >
                  <Tooltip title="Xem chi tiết">
                    <IconButton
                      onClick={(e) => {
                        e.stopPropagation();
                        handleOpen(a.id);
                      }}
                      sx={{
                        background: "rgba(255,215,0,0.9)",
                        color: "#000",
                        "&:hover": { background: "#FFD700" },
                      }}
                    >
                      <InfoOutlinedIcon />
                    </IconButton>
                  </Tooltip>
                </Box>

                {/* optional badges */}
                {!!a.movieCount && (
                  <Chip
                    label={`${a.movieCount} phim`}
                    size="small"
                    sx={{
                      position: "absolute",
                      top: 8,
                      right: 8,
                      background: "rgba(255,215,0,0.9)",
                      color: "#000",
                      fontWeight: 600,
                      fontSize: "0.75rem",
                    }}
                  />
                )}
                <Box sx={{ p: 1.5 }}>
                  <Typography sx={{ fontWeight: 700 }} noWrap>
                    {a.name}
                  </Typography>
                </Box>
              </ActorCard>
            </Grid>
          ))}
        </Grid>

        {/* Pagination */}
        {!imageSearching && totalPages > 1 && (
          <Box sx={{ display: "flex", justifyContent: "center", mt: 4, mb: 2 }}>
            <Pagination
              count={totalPages}
              page={currentPage}
              onChange={(e, newPage) => {
                setCurrentPage(newPage);
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
              color="primary"
              size="large"
              sx={{
                "& .MuiPaginationItem-root": {
                  color: "rgba(255,255,255,0.7)",
                  "&.Mui-selected": {
                    background: "rgba(255,215,0,0.9)",
                    color: "#000",
                    fontWeight: 600,
                    "&:hover": {
                      background: "#FFD700",
                    },
                  },
                  "&:hover": {
                    background: "rgba(255,215,0,0.2)",
                    color: "#FFD700",
                  },
                },
                "& .MuiPaginationItem-ellipsis": {
                  color: "rgba(255,255,255,0.5)",
                },
              }}
            />
          </Box>
        )}

        {/* Pagination info */}
        {!imageSearching && totalElements > 0 && (
          <Typography
            variant="body2"
            sx={{
              color: "rgba(255,255,255,0.6)",
              textAlign: "center",
              mt: 2,
              mb: 2,
            }}
          >
            Hiển thị {items.length} / {totalElements} diễn viên
            {totalPages > 1 && ` (Trang ${currentPage}/${totalPages})`}
          </Typography>
        )}
      </Container>
    </PageContainer>
  );
}

function useDebounce(value, delayMs) {
  const [val, setVal] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setVal(value), delayMs);
    return () => clearTimeout(t);
  }, [value, delayMs]);
  return val;
}
