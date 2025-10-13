import { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  Avatar,
  Link,
  IconButton,
  Button,
  Tooltip,
  Chip,
  Divider,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import {
  Favorite,
  List,
  History,
  Notifications,
  AccountCircle,
  Logout,
  DeleteOutline,
  PlayArrow,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { getFavorites, removeFavorite } from "../api/favorites";

const PageContainer = styled(Box)(() => ({
  minHeight: "100vh",
  background: "linear-gradient(180deg, #0b1220 0%, #0a0f1a 50%, #0b0d13 100%)",
  color: "#fff",
  paddingTop: 80,
}));

const Sidebar = styled(Paper)(() => ({
  background: "rgba(18, 18, 18, 0.95)",
  backdropFilter: "blur(20px)",
  border: "1px solid rgba(255, 255, 255, 0.12)",
  borderRadius: 16,
  padding: 24,
  height: "fit-content",
  position: "sticky",
  top: 100,
}));

const MainContent = styled(Paper)(() => ({
  background: "rgba(18, 18, 18, 0.95)",
  backdropFilter: "blur(20px)",
  border: "1px solid rgba(255, 255, 255, 0.12)",
  borderRadius: 16,
  padding: 24,
}));

const NavItem = styled(Box)(({ active }) => ({
  display: "flex",
  alignItems: "center",
  padding: "12px 16px",
  borderRadius: 8,
  marginBottom: 8,
  cursor: "pointer",
  transition: "all 0.2s ease",
  color: active ? "#FFD700" : "rgba(255, 255, 255, 0.8)",
  background: active ? "rgba(255, 215, 0, 0.1)" : "transparent",
  borderBottom: active ? "2px solid #FFD700" : "none",
  "&:hover": {
    background: "rgba(255, 215, 0, 0.1)",
    color: "#FFD700",
  },
  "& .MuiSvgIcon-root": {
    marginRight: 12,
    fontSize: "1.2rem",
  },
}));

function buildImageUrl(path) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  try {
    return new URL(path, window.location.origin).toString();
  } catch (_) {
    return path;
  }
}

export default function Favorites() {
  const { token, user } = useAuth();
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [page, setPage] = useState(0);
  const [size] = useState(24);
  const [totalPages, setTotalPages] = useState(1);
  const hasFetched = useRef(false);

  const navItems = useMemo(
    () => [
      {
        label: "Yêu thích",
        icon: <Favorite />,
        to: "/favorites",
        active: true,
      },
      { label: "Danh sách", icon: <List />, to: "/watchlist" },
      { label: "Xem tiếp", icon: <History />, to: "/account" },
      { label: "Thông báo", icon: <Notifications />, to: "/account" },
      { label: "Tài khoản", icon: <AccountCircle />, to: "/account" },
    ],
    []
  );

  useEffect(() => {
    if (!token || hasFetched.current) return;
    hasFetched.current = true;
    loadPage(0, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function loadPage(nextPage = 0, append = false) {
    try {
      setLoading(true);
      setError("");
      const data = await getFavorites({ page: nextPage, size }, token);
      const source = Array.isArray(data?.content) ? data.content : [];
      const mapped = source.map((fav) => ({
        id: fav.id,
        movieId: fav.movieId,
        title: fav.movieTitle,
        year: fav.movieYear,
        genres: fav.movieGenres || [],
        poster: buildImageUrl(fav.moviePosterUrl),
        addedAt: fav.addedAt,
      }));
      setItems((prev) => (append ? [...prev, ...mapped] : mapped));
      setPage(nextPage);
      setTotalPages(
        data?.totalPages ?? (mapped.length < size ? nextPage + 1 : nextPage + 2)
      );
    } catch (e) {
      setError(e?.message || "Không thể tải danh sách yêu thích");
    } finally {
      setLoading(false);
    }
  }

  const handleRemove = async (favoriteId) => {
    try {
      await removeFavorite(favoriteId, token);
      setItems((prev) => prev.filter((x) => x.id !== favoriteId));
    } catch (e) {
      setError(e?.message || "Không thể xóa khỏi yêu thích");
    }
  };

  const handlePlay = (movieId) => navigate(`/stream/${movieId}`);
  const handleOpenMovie = (movieId) => navigate(`/movie/${movieId}`);

  if (!token) {
    return (
      <PageContainer>
        <Container maxWidth="lg">
          <Typography variant="h5" sx={{ mt: 10, textAlign: "center" }}>
            Vui lòng đăng nhập để xem danh sách yêu thích
          </Typography>
        </Container>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <Container maxWidth="lg" sx={{ pb: 6 }}>
        <Grid container spacing={4} sx={{ py: 4 }}>
          <Grid item xs={12} md={3}>
            <Sidebar>
              <Typography
                variant="h5"
                sx={{ fontWeight: 800, mb: 3, color: "#fff" }}
              >
                Quản lý tài khoản
              </Typography>
              {navItems.map((item) => (
                <NavItem
                  key={item.label}
                  active={item.active}
                  onClick={() => navigate(item.to)}
                >
                  {item.icon}
                  {item.label}
                </NavItem>
              ))}

              <Divider
                sx={{ my: 3, borderColor: "rgba(255, 255, 255, 0.1)" }}
              />

              {/* User summary */}
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                }}
              >
                <Avatar
                  src={user?.avatarUrl}
                  sx={{
                    width: 60,
                    height: 60,
                    background: user?.avatarUrl
                      ? "transparent"
                      : "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
                    color: user?.avatarUrl ? "transparent" : "#000",
                    fontWeight: 700,
                    fontSize: "1.2rem",
                    mb: 2,
                  }}
                />
                <Typography
                  variant="h6"
                  sx={{
                    color: "#fff",
                    fontWeight: 700,
                    textAlign: "center",
                    mb: 0.5,
                  }}
                >
                  {user?.fullName || user?.username}
                </Typography>
                <Typography
                  variant="body2"
                  sx={{ color: "rgba(255,255,255,0.7)", mb: 2 }}
                >
                  {user?.email}
                </Typography>
                <Link
                  component="button"
                  onClick={() => navigate("/")}
                  sx={{
                    color: "rgba(255,255,255,0.7)",
                    textDecoration: "none",
                    fontSize: "0.9rem",
                  }}
                >
                  <Logout sx={{ fontSize: "1rem", mr: 0.5 }} /> Thoát
                </Link>
              </Box>
            </Sidebar>
          </Grid>

          <Grid item xs={12} md={9}>
            <MainContent>
              <Typography
                variant="h4"
                sx={{ fontWeight: 800, mb: 1, color: "#fff" }}
              >
                Yêu thích
              </Typography>
              <Typography
                variant="body2"
                sx={{ color: "rgba(255,255,255,0.7)", mb: 3 }}
              >
                Danh sách phim bạn đã thêm vào yêu thích
              </Typography>

              {error && (
                <Typography sx={{ color: "#ff6b6b", mb: 2 }}>
                  {error}
                </Typography>
              )}

              {loading && (
                <Typography sx={{ color: "#fff", mb: 2 }}>
                  Đang tải...
                </Typography>
              )}

              {!loading && items.length === 0 && (
                <Typography sx={{ color: "rgba(255,255,255,0.7)" }}>
                  Chưa có phim yêu thích nào.
                </Typography>
              )}

              <Grid container spacing={2}>
                {items.map((it) => (
                  <Grid item xs={6} sm={4} md={3} key={it.id}>
                    <Box
                      sx={{
                        position: "relative",
                        borderRadius: 2,
                        overflow: "hidden",
                        border: "1px solid rgba(255,255,255,0.12)",
                        background: "rgba(255,255,255,0.04)",
                        cursor: "pointer",
                      }}
                      onClick={() => handleOpenMovie(it.movieId)}
                    >
                      <Box
                        component="img"
                        src={it.poster}
                        alt={it.title}
                        sx={{
                          width: "100%",
                          height: "auto",
                          aspectRatio: "2 / 3",
                          objectFit: "cover",
                          display: "block",
                        }}
                        onError={(e) => {
                          e.currentTarget.src = `https://via.placeholder.com/300x420/333/fff?text=${encodeURIComponent(
                            it.title
                          )}`;
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
                        <Tooltip title="Xem phim">
                          <IconButton
                            onClick={(e) => {
                              e.stopPropagation();
                              handlePlay(it.movieId);
                            }}
                            sx={{
                              background: "rgba(255,215,0,0.9)",
                              color: "#000",
                              "&:hover": { background: "#FFD700" },
                            }}
                          >
                            <PlayArrow />
                          </IconButton>
                        </Tooltip>

                        <Tooltip title="Bỏ yêu thích">
                          <IconButton
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRemove(it.id);
                            }}
                            sx={{
                              background: "rgba(255,255,255,0.15)",
                              color: "#fff",
                              "&:hover": {
                                background: "rgba(255,255,255,0.3)",
                              },
                            }}
                          >
                            <DeleteOutline />
                          </IconButton>
                        </Tooltip>
                      </Box>

                      <Box sx={{ p: 1.5 }}>
                        <Typography
                          sx={{
                            color: "#fff",
                            fontWeight: 600,
                            fontSize: ".95rem",
                            lineHeight: 1.3,
                          }}
                          noWrap
                        >
                          {it.title}
                        </Typography>
                        <Box
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            gap: 1,
                            mt: 1,
                          }}
                        >
                          {!!it.year && (
                            <Chip
                              size="small"
                              label={it.year}
                              sx={{
                                height: 20,
                                fontSize: "0.7rem",
                                color: "rgba(255,255,255,0.85)",
                                background: "rgba(255,255,255,0.08)",
                              }}
                            />
                          )}
                          {Array.isArray(it.genres) &&
                            it.genres.slice(0, 1).map((g) => (
                              <Chip
                                key={g}
                                size="small"
                                label={g}
                                sx={{
                                  height: 20,
                                  fontSize: "0.7rem",
                                  color: "#FFD700",
                                  background: "rgba(255,215,0,0.15)",
                                }}
                              />
                            ))}
                        </Box>
                      </Box>
                    </Box>
                  </Grid>
                ))}
              </Grid>

              {page + 1 < totalPages && (
                <Box sx={{ textAlign: "center", mt: 4 }}>
                  <Button
                    variant="outlined"
                    onClick={() => loadPage(page + 1, true)}
                    disabled={loading}
                    sx={{
                      color: "rgba(255,255,255,0.9)",
                      borderColor: "rgba(255,255,255,0.3)",
                      borderRadius: 3,
                      px: 4,
                      textTransform: "none",
                      fontWeight: 600,
                      "&:hover": { borderColor: "#FFD700", color: "#FFD700" },
                    }}
                  >
                    {loading ? "Đang tải..." : "Xem thêm"}
                  </Button>
                </Box>
              )}
            </MainContent>
          </Grid>
        </Grid>
      </Container>
    </PageContainer>
  );
}
