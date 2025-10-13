import { useMemo, useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Box,
  Container,
  Typography,
  Chip,
  Stack,
  Button,
  Card,
  CardMedia,
  Divider,
  Avatar,
  TextField,
  Snackbar,
  Alert,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import FavoriteIcon from "@mui/icons-material/Favorite";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import AddIcon from "@mui/icons-material/Add";
import ShareIcon from "@mui/icons-material/Share";
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline";
import SendIcon from "@mui/icons-material/Send";
import useMovies from "../hooks/useMovies";
import Header from "../components/Header";
import { useAuth } from "../hooks/useAuth";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  MenuItem,
} from "@mui/material";
import { listCollections, addMovieToWatchlist } from "../api/watchlist";
import { addFavorite } from "../api/favorites";
import { listActors } from "../api/actors";

export default function MovieDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { movies } = useMovies();
  const { token, isAuthenticated } = useAuth();
  const [activeTab, setActiveTab] = useState("gallery");
  const [addingFav, setAddingFav] = useState(false);
  const [snack, setSnack] = useState({
    open: false,
    message: "",
    severity: "success",
  });
  const [addOpen, setAddOpen] = useState(false);
  const [collections, setCollections] = useState([]);
  const [addForm, setAddForm] = useState({
    collectionId: "",
    notes: "",
    priority: 1,
  });
  useEffect(() => {
    (async () => {
      try {
        if (!token) return;
        const cols = await listCollections(token);
        setCollections(cols || []);
      } catch (_) {
        // ignore
      }
    })();
  }, [token]);
  // Comments only; rating removed

  const movie = useMemo(() => {
    const numericId = Number(id);
    return movies.find(
      (m) => String(m.id) === String(id) || m.id === numericId
    );
  }, [id, movies]);

  // defer early return until after hooks

  const defaultCast = [
    {
      name: "Yoona",
      country: "Hàn Quốc",
      avatar: "https://i.pravatar.cc/150?img=47",
    },
    {
      name: "Lee Chae-min",
      country: "Hàn Quốc",
      avatar: "https://i.pravatar.cc/150?img=12",
    },
    {
      name: "Kang Han-na",
      country: "Hàn Quốc",
      avatar: "https://i.pravatar.cc/150?img=32",
    },
    {
      name: "Choi Gwi-hwa",
      country: "Hàn Quốc",
      avatar: "https://i.pravatar.cc/150?img=58",
    },
    {
      name: "Oh Eui-sik",
      country: "Hàn Quốc",
      avatar: "https://i.pravatar.cc/150?img=15",
    },
    {
      name: "Seo Yi-sook",
      country: "Hàn Quốc",
      avatar: "https://i.pravatar.cc/150?img=21",
    },
    {
      name: "Park Young-woon",
      country: "Hàn Quốc",
      avatar: "https://i.pravatar.cc/150?img=13",
    },
    {
      name: "Yoon Seo-ah",
      country: "Hàn Quốc",
      avatar: "https://i.pravatar.cc/150?img=45",
    },
  ];
  const castList =
    Array.isArray(movie?.actors) && movie.actors.length > 0
      ? movie.actors.map((a) => {
          if (typeof a === "string") return { name: a };
          const name = a?.name || a?.fullName || a?.username || "";
          const avatar =
            a?.avatar ||
            a?.avatarUrl ||
            a?.imageUrl ||
            a?.image ||
            a?.profilePath;
          const id = a?.id;
          return { id, name, avatar };
        })
      : defaultCast;

  // Enrich actors when only name is available
  const [enrichedCast, setEnrichedCast] = useState([]);
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const src = castList.slice(0, 12);
        const out = await Promise.all(
          src.map(async (it) => {
            if ((it?.avatar && it?.id) || !it?.name) return it;
            try {
              const res = await listActors({ q: it.name, page: 0, size: 1 });
              const first = Array.isArray(res?.content)
                ? res.content[0]
                : Array.isArray(res)
                ? res[0]
                : null;
              if (first) {
                return {
                  id: first.id,
                  name: first.name || it.name,
                  avatar:
                    first.imageUrl ||
                    first.image ||
                    first.avatarUrl ||
                    it.avatar,
                };
              }
              return it;
            } catch (_) {
              return it;
            }
          })
        );
        if (mounted) setEnrichedCast(out);
      } catch (_) {
        if (mounted) setEnrichedCast(castList.slice(0, 12));
      }
    })();
    return () => {
      mounted = false;
    };
  }, [castList]);

  return (
    <Box
      sx={{
        background:
          "linear-gradient(180deg, #0b1220 0%, #0a0f1a 50%, #0b0d13 100%)",
        minHeight: "100vh",
        color: "#fff",
        width: "100%",
        position: "relative",

        // Ambient hero backdrop with soft blur + vignette
        "&:before": {
          content: '""',
          position: "fixed",
          inset: 0,
          background: `radial-gradient(60% 60% at 50% 20%, rgba(255,215,0,0.12) 0%, rgba(11,18,32,0.0) 60%) , url(${
            movie?.backdrop || movie?.thumb
          }) center/cover no-repeat`,
          filter: "blur(28px)",
          opacity: 0.35,
          zIndex: 0,
          transform: "translateZ(0)",
        },
        "&:after": {
          content: '""',
          position: "fixed",
          inset: 0,
          background:
            "linear-gradient(180deg, rgba(11,18,32,0.6) 0%, rgba(10,15,26,0.85) 40%, #0b0d13 100%)",
          zIndex: 0,
        },
      }}
    >
      <Header />
      {!movie && (
        <Container maxWidth="lg" sx={{ py: 6 }}>
          <Typography color="#fff" variant="h5" gutterBottom>
            Không tìm thấy phim
          </Typography>
          <Button
            variant="outlined"
            onClick={() => navigate(-1)}
            startIcon={<ArrowBackIcon />}
            sx={{ color: "#FFD700", borderColor: "#FFD700" }}
          >
            Quay lại
          </Button>
        </Container>
      )}
      {movie && (
        <>
          <Snackbar
            anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
            open={snack.open}
            autoHideDuration={2500}
            onClose={() => setSnack((s) => ({ ...s, open: false }))}
          >
            <Alert
              onClose={() => setSnack((s) => ({ ...s, open: false }))}
              severity={snack.severity}
              sx={{ width: "100%" }}
            >
              {snack.message}
            </Alert>
          </Snackbar>
          <Container
            maxWidth="xl"
            disableGutters
            sx={{
              py: 4,
              px: { xs: 2, md: 4 },
              mt: { xs: 6, md: 8 },
              position: "relative",
              zIndex: 1,
            }}
          >
            {movie && (
              <Button
                onClick={() => navigate(-1)}
                startIcon={<ArrowBackIcon />}
                sx={{ color: "#FFD700", mb: 2 }}
              >
                Quay lại
              </Button>
            )}

            <Stack
              direction={{ xs: "column", md: "row" }}
              spacing={6}
              alignItems={{ xs: "stretch", md: "flex-start" }}
            >
              {/* Left column: Poster + detailed info */}
              <Box sx={{ width: { xs: "100%", md: 360, rowGap: 2 } }}>
                <Card
                  sx={{
                    width: "100%",
                    background: "#0f1013",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 2,
                    rowGap: 2,
                    boxShadow:
                      "0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.02)",
                    backdropFilter: "blur(6px)",
                    transition: "transform 0.25s ease, box-shadow 0.25s ease",
                    ":hover": {
                      transform: "translateY(-2px)",
                      boxShadow:
                        "0 16px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.03)",
                    },
                  }}
                >
                  <CardMedia
                    component="img"
                    image={movie.thumb}
                    alt={movie.title}
                    sx={{
                      aspectRatio: "2 / 3",
                      objectFit: "cover",
                      objectPosition: "center",
                      borderRadius: 2,
                    }}
                  />
                </Card>

                <Box sx={{ mt: 3 }}>
                  <Typography
                    variant="h5"
                    fontWeight={900}
                    gutterBottom
                    sx={{
                      backgroundImage:
                        "linear-gradient(90deg, #fff 0%, #FFD700 60%, #fff 100%)",
                      WebkitBackgroundClip: "text",
                      backgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                      letterSpacing: 0.3,
                    }}
                  >
                    {movie.title}
                  </Typography>
                  {movie.englishTitle && (
                    <Typography
                      variant="subtitle2"
                      color="rgba(255,255,255,0.75)"
                      gutterBottom
                    >
                      {movie.englishTitle}
                    </Typography>
                  )}

                  <Stack
                    direction="row"
                    alignItems="center"
                    flexWrap="wrap"
                    useFlexGap
                    gap={1.5}
                    sx={{ my: 1, mt: 4 }}
                  >
                    {movie.ageRating && (
                      <Chip
                        label={movie.ageRating}
                        size="small"
                        sx={{
                          bgcolor: "rgba(255,255,255,0.08)",
                          color: "#fff",
                          border: "1px solid rgba(255,255,255,0.16)",
                        }}
                      />
                    )}
                    {movie.year && (
                      <Chip
                        label={movie.year}
                        size="small"
                        sx={{
                          bgcolor: "rgba(255,255,255,0.08)",
                          color: "#fff",
                          border: "1px solid rgba(255,255,255,0.16)",
                        }}
                      />
                    )}
                    {typeof movie.imdbRating === "number" && (
                      <Chip
                        label={`IMDb ${movie.imdbRating.toFixed(1)}`}
                        size="small"
                        sx={{
                          bgcolor: "rgba(255,215,0,0.15)",
                          color: "#FFD700",
                          border: "1px solid rgba(255,215,0,0.35)",
                        }}
                      />
                    )}
                  </Stack>

                  {Array.isArray(movie.genres) && movie.genres.length > 0 && (
                    <Stack
                      direction="row"
                      useFlexGap
                      gap={1}
                      sx={{ my: 1, flexWrap: "wrap", mt: 1 }}
                    >
                      {movie.genres.map((g, i) => (
                        <Chip
                          key={`${g}-${i}`}
                          label={g}
                          size="small"
                          sx={{
                            bgcolor: "rgba(255,255,255,0.08)",
                            color: "#fff",
                            border: "1px solid rgba(255,255,255,0.16)",
                          }}
                        />
                      ))}
                    </Stack>
                  )}

                  {/* Quốc gia moved into Giới thiệu section below */}

                  {(movie.synopsis || movie.country) && (
                    <Box sx={{ mt: 5 }}>
                      <Typography
                        variant="subtitle1"
                        fontWeight={800}
                        gutterBottom
                      >
                        Giới thiệu:
                      </Typography>
                      {movie.synopsis && (
                        <Typography
                          variant="body2"
                          sx={{ color: "rgba(255,255,255,0.85)" }}
                        >
                          {movie.synopsis}
                        </Typography>
                      )}
                      {movie.country && (
                        <Typography
                          variant="body2"
                          sx={{ mt: 1, color: "rgba(255,255,255,0.85)" }}
                        >
                          <strong>Quốc gia:</strong> {movie.country}
                        </Typography>
                      )}
                      {movie.language && (
                        <Typography
                          variant="body2"
                          sx={{ mt: 1, color: "rgba(255,255,255,0.85)" }}
                        >
                          <strong>Ngôn ngữ:</strong> {movie.language}
                        </Typography>
                      )}
                      {Array.isArray(movie.directors) &&
                        movie.directors.length > 0 && (
                          <Typography
                            variant="body2"
                            sx={{ mt: 1, color: "rgba(255,255,255,0.85)" }}
                          >
                            <strong>Đạo diễn:</strong>{" "}
                            {movie.directors.join(", ")}
                          </Typography>
                        )}
                    </Box>
                  )}

                  <Box sx={{ mt: 2 }}>
                    {movie.duration && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}>
                        <strong>Thời lượng:</strong>{" "}
                        {movie.duration ? `${movie.duration} phút` : "Duration"}
                      </Typography>
                    )}
                  </Box>

                  {castList.length > 0 && (
                    <Box sx={{ mt: 3 }}>
                      <Typography variant="h6" fontWeight={900} gutterBottom>
                        Diễn viên
                      </Typography>
                      <Box
                        sx={{
                          display: "grid",
                          gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                          gap: 2,
                          mt: 5,
                        }}
                      >
                        {(enrichedCast.length > 0 ? enrichedCast : castList)
                          .slice(0, 12)
                          .map((c, idx) => {
                            const fallbackAvatar = `https://i.pravatar.cc/150?img=${
                              (idx % 70) + 1
                            }`;
                            const item =
                              typeof c === "string" ? { name: c } : c;
                            return (
                              <Stack
                                key={`${item.name ?? idx}-${idx}`}
                                alignItems="center"
                                spacing={1}
                                onClick={() => {
                                  if (item?.id) navigate(`/actor/${item.id}`);
                                  else if (item?.name)
                                    navigate(
                                      `/actors?q=${encodeURIComponent(
                                        item.name
                                      )}`
                                    );
                                }}
                                sx={{
                                  cursor: "pointer",
                                  transition:
                                    "transform 0.2s ease, opacity 0.2s ease",
                                  ":hover": {
                                    transform: "translateY(-2px)",
                                    opacity: 0.95,
                                  },
                                }}
                              >
                                <Avatar
                                  src={item.avatar || fallbackAvatar}
                                  alt={item.name}
                                  sx={{ width: 64, height: 64 }}
                                  imgProps={{
                                    onError: (e) => {
                                      e.currentTarget.src = fallbackAvatar;
                                    },
                                  }}
                                >
                                  {(item.name || "?").charAt(0)}
                                </Avatar>
                                <Typography
                                  variant="caption"
                                  noWrap
                                  sx={{
                                    textAlign: "center",
                                    width: "100%",
                                    display: "block",
                                  }}
                                >
                                  {item.name || "Diễn viên"}
                                </Typography>
                                {/* Removed country display under each actor */}
                              </Stack>
                            );
                          })}
                      </Box>
                    </Box>
                  )}
                </Box>
              </Box>

              {/* Right column: actions, tabs, episodes, comments */}
              <Box sx={{ flex: 1 }}>
                <Stack
                  direction={{ xs: "column", sm: "row" }}
                  spacing={2}
                  sx={{ mt: 0, flexWrap: "wrap", rowGap: 1 }}
                >
                  <Button
                    variant="contained"
                    startIcon={<PlayArrowIcon />}
                    disabled={
                      !movie.isAvailable ||
                      !(movie.streamingUrl || movie.videoUrl)
                    }
                    onClick={() => navigate(`/stream/${movie.id}`)}
                    sx={{
                      bgcolor:
                        !movie.isAvailable ||
                        !(movie.streamingUrl || movie.videoUrl)
                          ? "rgba(255,255,255,0.2)"
                          : "#FFD700",
                      color:
                        !movie.isAvailable ||
                        !(movie.streamingUrl || movie.videoUrl)
                          ? "rgba(255,255,255,0.6)"
                          : "#000",
                      fontWeight: 800,
                      px: 2.5,
                      boxShadow: "0 8px 24px rgba(255,215,0,0.25)",
                      backgroundImage:
                        "linear-gradient(90deg, rgba(255,215,0,1) 0%, rgba(255,193,7,1) 100%)",
                      ":hover": {
                        bgcolor:
                          !movie.isAvailable ||
                          !(movie.streamingUrl || movie.videoUrl)
                            ? "rgba(255,255,255,0.25)"
                            : "#FFC107",
                        transform: "translateY(-1px)",
                        boxShadow: "0 10px 28px rgba(255,215,0,0.35)",
                      },
                      transition: "all 0.2s ease",
                    }}
                  >
                    {!movie.isAvailable ||
                    !(movie.streamingUrl || movie.videoUrl)
                      ? "Chưa có nguồn"
                      : "Xem ngay"}
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<FavoriteIcon />}
                    sx={{
                      color: "#fff",
                      borderColor: "rgba(255,255,255,0.2)",
                      backdropFilter: "blur(6px)",
                      ":hover": { bgcolor: "rgba(255,255,255,0.06)" },
                    }}
                    disabled={addingFav}
                    onClick={async () => {
                      if (!isAuthenticated) {
                        navigate("/login");
                        return;
                      }
                      try {
                        setAddingFav(true);
                        await addFavorite(Number(movie.id), token);
                        setSnack({
                          open: true,
                          message: "Đã thêm vào yêu thích",
                          severity: "success",
                        });
                      } catch (e) {
                        setSnack({
                          open: true,
                          message: "Không thể thêm vào yêu thích",
                          severity: "error",
                        });
                      } finally {
                        setAddingFav(false);
                      }
                    }}
                  >
                    {addingFav ? "Đang thêm..." : "Yêu thích"}
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<AddIcon />}
                    sx={{
                      color: "#fff",
                      borderColor: "rgba(255,255,255,0.2)",
                      backdropFilter: "blur(6px)",
                      ":hover": { bgcolor: "rgba(255,255,255,0.06)" },
                    }}
                    onClick={() => {
                      if (!isAuthenticated) {
                        navigate("/login");
                        return;
                      }
                      setAddOpen(true);
                    }}
                  >
                    Thêm vào
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<ShareIcon />}
                    sx={{
                      color: "#fff",
                      borderColor: "rgba(255,255,255,0.2)",
                      backdropFilter: "blur(6px)",
                      ":hover": { bgcolor: "rgba(255,255,255,0.06)" },
                    }}
                  >
                    Chia sẻ
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<ChatBubbleOutlineIcon />}
                    sx={{
                      color: "#fff",
                      borderColor: "rgba(255,255,255,0.2)",
                      backdropFilter: "blur(6px)",
                      ":hover": { bgcolor: "rgba(255,255,255,0.06)" },
                    }}
                  >
                    Bình luận
                  </Button>
                  {/* Removed rating display to keep interface focused on comments */}
                </Stack>

                {/* Trailer */}
                <Box sx={{ mt: 4 }}>
                  <Typography variant="h5" fontWeight={900} sx={{ mb: 2 }}>
                    Trailer
                  </Typography>
                  {movie.trailerUrl ? (
                    <Card
                      sx={{
                        background: "#0f1013",
                        border: "1px solid rgba(255,255,255,0.1)",
                        borderRadius: 2,
                        overflow: "hidden",
                        maxWidth: 900, // Tăng chiều rộng tối đa
                      }}
                    >
                      <Box
                        sx={{
                          width: "100%",
                          position: "relative",
                          pt: "56.25%",
                        }}
                      >
                        {/* 16:9 aspect ratio */}
                        {movie.trailerUrl.includes("youtube.com") ||
                        movie.trailerUrl.includes("youtu.be") ? (
                          // YouTube URL - use iframe
                          <iframe
                            src={movie.trailerUrl
                              .replace("watch?v=", "embed/")
                              .replace("youtu.be/", "youtube.com/embed/")}
                            title="Movie Trailer"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              border: 0,
                            }}
                          />
                        ) : (
                          // Direct video file - use video element
                          <video
                            controls
                            preload="metadata"
                            style={{
                              position: "absolute",
                              top: 0,
                              left: 0,
                              width: "100%",
                              height: "100%",
                              objectFit: "cover",
                            }}
                          >
                            <source src={movie.trailerUrl} type="video/mp4" />
                            <source src={movie.trailerUrl} type="video/webm" />
                            <source src={movie.trailerUrl} type="video/ogg" />
                            Trình duyệt của bạn không hỗ trợ video.
                          </video>
                        )}
                      </Box>
                      <Box sx={{ p: 2.5 }}>
                        {" "}
                        {/* Tăng padding */}
                        <Typography
                          variant="h5"
                          fontWeight={900}
                          gutterBottom
                          sx={{
                            backgroundImage:
                              "linear-gradient(90deg, #fff 0%, #FFD700 60%, #fff 100%)",
                            WebkitBackgroundClip: "text",
                            backgroundClip: "text",
                            WebkitTextFillColor: "transparent",
                            letterSpacing: 0.3,
                          }}
                        >
                          {movie.title}
                        </Typography>
                        {movie.englishTitle && (
                          <Typography
                            variant="subtitle1"
                            sx={{ color: "rgba(255,255,255,0.7)", mb: 1.5 }}
                          >
                            {movie.englishTitle}
                          </Typography>
                        )}
                        <Stack
                          direction="row"
                          spacing={1.5}
                          alignItems="center"
                          sx={{ flexWrap: "wrap", gap: 1 }}
                        >
                          {movie.videoQuality && (
                            <Chip
                              size="small"
                              label={movie.videoQuality.toUpperCase()}
                              sx={{
                                bgcolor: "rgba(255,255,255,0.08)",
                                color: "#fff",
                              }}
                            />
                          )}
                          {movie.year && (
                            <Chip
                              size="small"
                              label={movie.year}
                              sx={{
                                bgcolor: "rgba(255,255,255,0.08)",
                                color: "#fff",
                              }}
                            />
                          )}
                          {movie.videoDuration && (
                            <Chip
                              size="small"
                              label={`${movie.videoDuration} phút`}
                              sx={{
                                bgcolor: "rgba(255,255,255,0.08)",
                                color: "#fff",
                              }}
                            />
                          )}
                        </Stack>
                      </Box>
                    </Card>
                  ) : (
                    <Card
                      sx={{
                        background: "#0f1013",
                        border: "1px solid rgba(255,255,255,0.1)",
                        borderRadius: 2,
                        p: 4, // Tăng padding
                        maxWidth: 800,
                      }}
                    >
                      <Typography
                        variant="h6" // Tăng kích thước chữ
                        sx={{
                          color: "rgba(255,255,255,0.7)",
                          textAlign: "center",
                          fontWeight: 500,
                        }}
                      >
                        Phim chưa có Trailer
                      </Typography>
                    </Card>
                  )}
                </Box>

                <Stack
                  direction="row"
                  spacing={6}
                  sx={{
                    mt: 5,
                    flexWrap: "wrap",
                    rowGap: 2,
                    pb: 1.5,
                    borderBottom: "1px solid rgba(255,255,255,0.08)",
                  }}
                >
                  <Button
                    size="small"
                    variant="text"
                    onClick={() => setActiveTab("gallery")}
                    sx={{
                      position: "relative",
                      px: 0,
                      textTransform: "none",
                      fontWeight: 800,
                      color:
                        activeTab === "gallery"
                          ? "#FFD700"
                          : "rgba(255,255,255,0.8)",
                      ":hover": { color: "#FFD700" },
                      "&::after": {
                        content: '""',
                        position: "absolute",
                        left: 0,
                        right: 0,
                        bottom: -10,
                        height: 2,
                        backgroundColor:
                          activeTab === "gallery" ? "#FFD700" : "transparent",
                        transition: "background-color 0.2s ease",
                      },
                    }}
                  >
                    Gallery
                  </Button>
                  <Button
                    size="small"
                    variant="text"
                    onClick={() => setActiveTab("cast")}
                    sx={{
                      position: "relative",
                      px: 0,
                      textTransform: "none",
                      fontWeight: 800,
                      color:
                        activeTab === "cast"
                          ? "#FFD700"
                          : "rgba(255,255,255,0.8)",
                      ":hover": { color: "#FFD700" },
                      "&::after": {
                        content: '""',
                        position: "absolute",
                        left: 0,
                        right: 0,
                        bottom: -10,
                        height: 2,
                        backgroundColor:
                          activeTab === "cast" ? "#FFD700" : "transparent",
                        transition: "background-color 0.2s ease",
                      },
                    }}
                  >
                    Diễn viên
                  </Button>
                  <Button
                    size="small"
                    variant="text"
                    onClick={() => setActiveTab("suggest")}
                    sx={{
                      position: "relative",
                      px: 0,
                      textTransform: "none",
                      fontWeight: 800,
                      color:
                        activeTab === "suggest"
                          ? "#FFD700"
                          : "rgba(255,255,255,0.8)",
                      ":hover": { color: "#FFD700" },
                      "&::after": {
                        content: '""',
                        position: "absolute",
                        left: 0,
                        right: 0,
                        bottom: -10,
                        height: 2,
                        backgroundColor:
                          activeTab === "suggest" ? "#FFD700" : "transparent",
                        transition: "background-color 0.2s ease",
                      },
                    }}
                  >
                    Đề xuất
                  </Button>
                </Stack>
                {/* Right column content area */}
                {activeTab === "gallery" && (
                  <Box sx={{ mt: 4 }}>
                    <Typography variant="body2" color="rgba(255,255,255,0.8)">
                      Gallery sẽ hiển thị ở đây.
                    </Typography>
                  </Box>
                )}
                {activeTab === "cast" && (
                  <Box sx={{ mt: 4 }}>
                    <Typography variant="body2" color="rgba(255,255,255,0.8)">
                      Danh sách diễn viên sẽ hiển thị ở đây.
                    </Typography>
                  </Box>
                )}
                {activeTab === "suggest" && (
                  <Box sx={{ mt: 4 }}>
                    <Typography variant="body2" color="rgba(255,255,255,0.8)">
                      Các đề xuất liên quan sẽ hiển thị ở đây.
                    </Typography>
                  </Box>
                )}

                {/* Comments & Ratings Section inside right column */}
                <Box sx={{ mt: 5 }}>
                  <Stack
                    direction="row"
                    alignItems="center"
                    spacing={2}
                    sx={{ mb: 2 }}
                  >
                    <Typography variant="h6" fontWeight={800}>
                      Bình luận (3505)
                    </Typography>
                  </Stack>

                  {/* Removed review/rating box; keep comments only */}

                  <Stack
                    direction="row"
                    spacing={2}
                    alignItems="flex-start"
                    sx={{
                      p: 2,
                      bgcolor: "rgba(15,16,19,0.9)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 2,
                      boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
                      backdropFilter: "blur(8px)",
                    }}
                  >
                    <Avatar alt="Bạn" src="" sx={{ width: 40, height: 40 }} />
                    <Stack spacing={1} sx={{ flex: 1 }}>
                      <TextField
                        placeholder="Viết bình luận..."
                        multiline
                        minRows={2}
                        variant="outlined"
                        fullWidth
                        InputProps={{ sx: { color: "#fff" } }}
                        sx={{
                          "& .MuiOutlinedInput-root": {
                            color: "#fff",
                            bgcolor: "#111",
                            "& fieldset": {
                              borderColor: "rgba(255,255,255,0.2)",
                            },
                            "&:hover fieldset": {
                              borderColor: "rgba(255,255,255,0.35)",
                            },
                            "&.Mui-focused fieldset": {
                              borderColor: "#FFD700",
                            },
                          },
                        }}
                      />
                      <Stack direction="row" justifyContent="flex-end">
                        <Button
                          endIcon={<SendIcon />}
                          sx={{
                            color: "#000",
                            bgcolor: "#FFD700",
                            fontWeight: 700,
                            px: 2,
                            borderRadius: 999,
                            ":hover": { bgcolor: "#FFC107" },
                          }}
                        >
                          Gửi
                        </Button>
                      </Stack>
                    </Stack>
                  </Stack>

                  <Divider
                    sx={{ my: 3, borderColor: "rgba(255,255,255,0.08)" }}
                  />

                  <Stack spacing={2}>
                    {[1, 2, 3].map((i) => (
                      <Stack key={i} direction="row" spacing={2}>
                        <Avatar sx={{ width: 36, height: 36 }}>U{i}</Avatar>
                        <Box sx={{ flex: 1 }}>
                          <Stack
                            direction="row"
                            alignItems="center"
                            spacing={1}
                          >
                            <Typography variant="subtitle2" fontWeight={700}>
                              Người dùng {i}
                            </Typography>
                            <Typography
                              variant="caption"
                              color="rgba(255,255,255,0.6)"
                            >
                              11 phút trước
                            </Typography>
                          </Stack>
                          <Typography
                            variant="body2"
                            sx={{ color: "rgba(255,255,255,0.85)" }}
                          >
                            Bình luận mẫu để minh hoạ giao diện.
                          </Typography>
                        </Box>
                      </Stack>
                    ))}
                  </Stack>
                </Box>
              </Box>
            </Stack>
          </Container>
          <Dialog
            open={addOpen}
            onClose={() => setAddOpen(false)}
            fullWidth
            maxWidth="sm"
          >
            <DialogTitle>Thêm vào danh sách</DialogTitle>
            <DialogContent>
              <Stack spacing={2} sx={{ mt: 1 }}>
                <TextField
                  select
                  label="Bộ sưu tập"
                  value={addForm.collectionId}
                  onChange={(e) =>
                    setAddForm((p) => ({ ...p, collectionId: e.target.value }))
                  }
                >
                  {collections.map((c) => (
                    <MenuItem key={c.id} value={c.id}>
                      {c.name}
                    </MenuItem>
                  ))}
                </TextField>
                <TextField
                  label="Ghi chú"
                  value={addForm.notes}
                  onChange={(e) =>
                    setAddForm((p) => ({ ...p, notes: e.target.value }))
                  }
                />
                <TextField
                  label="Ưu tiên"
                  type="number"
                  value={addForm.priority}
                  onChange={(e) =>
                    setAddForm((p) => ({
                      ...p,
                      priority: Number(e.target.value || 0),
                    }))
                  }
                />
              </Stack>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setAddOpen(false)}>Hủy</Button>
              <Button
                onClick={async () => {
                  try {
                    await addMovieToWatchlist(Number(movie.id), addForm, token);
                    setSnack({
                      open: true,
                      message: "Đã thêm vào danh sách",
                      severity: "success",
                    });
                    setAddOpen(false);
                  } catch (_) {
                    setSnack({
                      open: true,
                      message: "Không thể thêm",
                      severity: "error",
                    });
                  }
                }}
                variant="contained"
              >
                Thêm
              </Button>
            </DialogActions>
          </Dialog>
        </>
      )}
    </Box>
  );
}
