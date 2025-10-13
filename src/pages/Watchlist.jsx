import { useEffect, useState } from "react";
import {
  Box,
  Button,
  Card,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Grid,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Stack,
  TextField,
  Typography,
  Avatar,
  Link,
  Paper,
  Tooltip,
  Chip,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import {
  Favorite,
  List as ListIcon,
  History,
  Notifications,
  AccountCircle,
  Logout,
  PlayArrow,
  DeleteOutline,
} from "@mui/icons-material";
import AddIcon from "@mui/icons-material/Add";
import EditIcon from "@mui/icons-material/Edit";
import DeleteIcon from "@mui/icons-material/Delete";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import {
  listCollections,
  createCollection,
  updateCollection,
  deleteCollection,
  getCollectionMovies,
  removeMovieFromWatchlist,
} from "../api/watchlist";

// Styled components to mirror Favorites layout
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
  "& .MuiSvgIcon-root": { marginRight: 12, fontSize: "1.2rem" },
}));

function buildImageUrl(path) {
  if (!path) return "";
  if (typeof path !== "string") return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  try {
    return new URL(path, window.location.origin).toString();
  } catch (_) {
    return path;
  }
}

export default function Watchlist() {
  const navigate = useNavigate();
  const { token, user } = useAuth();
  const [collections, setCollections] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [movies, setMovies] = useState([]);
  // lightweight page; omit loading state to simplify
  const [error, setError] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState({ name: "", description: "" });

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const cols = await listCollections(token);
        if (!mounted) return;
        setCollections(cols || []);
        const firstId = cols?.[0]?.id ?? null;
        setSelectedId(firstId);
        if (firstId) {
          const ms = await getCollectionMovies(firstId, token);
          if (mounted) setMovies(ms || []);
        }
      } catch (e) {
        if (mounted) setError(e.message || "Lỗi tải danh sách");
      } finally {
        // no-op
      }
    })();
    return () => {
      mounted = false;
    };
  }, [token]);

  const openCreate = () => {
    setEditing(null);
    setForm({ name: "", description: "" });
    setDialogOpen(true);
  };
  const openEdit = (col) => {
    setEditing(col);
    setForm({ name: col.name || "", description: col.description || "" });
    setDialogOpen(true);
  };
  const onSubmit = async () => {
    try {
      if (editing) {
        const updated = await updateCollection(editing.id, form, token);
        setCollections((prev) =>
          prev.map((c) => (c.id === editing.id ? updated : c))
        );
      } else {
        const created = await createCollection(form, token);
        setCollections((prev) => [created, ...prev]);
      }
      setDialogOpen(false);
    } catch (e) {
      setError(e.message || "Lỗi lưu");
    }
  };
  const onDelete = async (id) => {
    if (!window.confirm("Xóa bộ sưu tập này?")) return;
    try {
      await deleteCollection(id, token);
      setCollections((prev) => prev.filter((c) => c.id !== id));
      if (selectedId === id) {
        setSelectedId(null);
        setMovies([]);
      }
    } catch (e) {
      setError(e.message || "Lỗi xóa");
    }
  };
  const onSelect = async (id) => {
    try {
      setSelectedId(id);
      const ms = await getCollectionMovies(id, token);
      setMovies(ms || []);
    } catch (e) {
      setError(e.message || "Lỗi tải phim");
    }
  };

  const onRemoveMovie = async (movieId) => {
    try {
      await removeMovieFromWatchlist(movieId, token);
      setMovies((prev) => prev.filter((m) => (m.movieId ?? m.id) !== movieId));
    } catch (e) {
      setError(e.message || "Không thể xóa phim khỏi danh sách");
    }
  };

  const navItems = [
    { label: "Yêu thích", icon: <Favorite />, to: "/favorites", active: false },
    { label: "Danh sách", icon: <ListIcon />, to: "/watchlist", active: true },
    { label: "Xem tiếp", icon: <History />, to: "/account", active: false },
    {
      label: "Thông báo",
      icon: <Notifications />,
      to: "/account",
      active: false,
    },
    {
      label: "Tài khoản",
      icon: <AccountCircle />,
      to: "/account",
      active: false,
    },
  ];

  const resolvePoster = (m) => {
    const p =
      m.thumb ||
      m.poster ||
      m.image ||
      m.posterUrl ||
      m.imageUrl ||
      m.moviePosterUrl ||
      m.posterPath ||
      (m.movie &&
        (m.movie.posterUrl ||
          m.movie.poster ||
          m.movie.imageUrl ||
          m.movie.thumb));
    return buildImageUrl(p);
  };
  const resolveTitle = (m) =>
    m.title || m.movieTitle || (m.movie && m.movie.title) || "Phim";
  const movieIdOf = (m) => m.movieId ?? m.id;
  const handleOpenMovie = (m) => navigate(`/movie/${movieIdOf(m)}`);
  const handlePlay = (m) => navigate(`/stream/${movieIdOf(m)}`);

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
              <Stack
                direction="row"
                justifyContent="space-between"
                alignItems="center"
                sx={{ mb: 2 }}
              >
                <Typography
                  variant="h4"
                  sx={{ fontWeight: 800, mb: 1, color: "#fff" }}
                >
                  Danh sách
                </Typography>
                <Button
                  startIcon={<AddIcon />}
                  onClick={openCreate}
                  sx={{
                    color: "#000",
                    bgcolor: "#FFD700",
                    fontWeight: 700,
                    ":hover": { bgcolor: "#FFC107" },
                  }}
                >
                  Bộ sưu tập mới
                </Button>
              </Stack>

              {error && (
                <Typography sx={{ color: "#ff6b6b", mb: 2 }}>
                  {error}
                </Typography>
              )}

              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <Card
                    sx={{
                      background: "#0f1013",
                      border: "1px solid rgba(255,255,255,0.1)",
                      p: 2,
                    }}
                  >
                    <List>
                      {(collections || []).map((c) => (
                        <ListItem
                          key={c.id}
                          selected={c.id === selectedId}
                          onClick={() => onSelect(c.id)}
                          sx={{ cursor: "pointer", borderRadius: 1 }}
                          secondaryAction={
                            <Stack direction="row" spacing={1}>
                              <IconButton
                                size="small"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  openEdit(c);
                                }}
                                sx={{ color: "#fff" }}
                              >
                                <EditIcon fontSize="small" />
                              </IconButton>
                              <IconButton
                                size="small"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onDelete(c.id);
                                }}
                                sx={{ color: "#ff6b6b" }}
                              >
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            </Stack>
                          }
                        >
                          <ListItemText
                            primary={c.name}
                            secondary={`Phim: ${
                              c.movieCount ?? (c.movies?.length || 0)
                            }`}
                            primaryTypographyProps={{
                              sx: { color: "#fff", fontWeight: 700 },
                            }}
                            secondaryTypographyProps={{
                              sx: { color: "rgba(255,255,255,0.65)" },
                            }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Card>
                </Grid>

                <Grid item xs={12} md={8}>
                  <Card
                    sx={{
                      background: "#0f1013",
                      border: "1px solid rgba(255,255,255,0.1)",
                      p: 2,
                    }}
                  >
                    <Typography variant="h6" fontWeight={900} sx={{ mb: 1 }}>
                      Phim trong bộ sưu tập
                    </Typography>
                    <Divider
                      sx={{ borderColor: "rgba(255,255,255,0.08)", mb: 2 }}
                    />
                    <Grid container spacing={2}>
                      {(movies || []).map((m) => {
                        const title = resolveTitle(m);
                        const poster =
                          resolvePoster(m) ||
                          `https://via.placeholder.com/300x420/333/fff?text=${encodeURIComponent(
                            title
                          )}`;
                        return (
                          <Grid
                            key={m.id || m.movieId || title}
                            item
                            xs={12}
                            sm={6}
                            md={4}
                          >
                            <Card
                              sx={{
                                background: "#111723",
                                border: "1px solid rgba(255,255,255,0.08)",
                                p: 1.5,
                              }}
                            >
                              <Box
                                sx={{
                                  position: "relative",
                                  borderRadius: 1,
                                  overflow: "hidden",
                                  cursor: "pointer",
                                }}
                                onClick={() => handleOpenMovie(m)}
                              >
                                <Box
                                  component="img"
                                  src={poster}
                                  alt={title}
                                  sx={{
                                    width: "100%",
                                    aspectRatio: "2 / 3",
                                    height: "auto",
                                    objectFit: "cover",
                                    display: "block",
                                  }}
                                  onError={(e) => {
                                    e.currentTarget.src = `https://via.placeholder.com/300x420/333/fff?text=${encodeURIComponent(
                                      title
                                    )}`;
                                  }}
                                />
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
                                        handlePlay(m);
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
                                  <Tooltip title="Xóa khỏi danh sách">
                                    <IconButton
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        onRemoveMovie(m.movieId ?? m.id);
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
                              </Box>
                              <Box sx={{ p: 1.5, pt: 1 }}>
                                <Typography
                                  sx={{
                                    color: "#fff",
                                    fontWeight: 600,
                                    fontSize: ".95rem",
                                    lineHeight: 1.3,
                                  }}
                                  noWrap
                                >
                                  {title}
                                </Typography>
                                {Array.isArray(m.genres) &&
                                  m.genres.slice(0, 1).map((g) => (
                                    <Chip
                                      key={g}
                                      size="small"
                                      label={g}
                                      sx={{
                                        height: 20,
                                        fontSize: "0.7rem",
                                        ml: 0.5,
                                        color: "#FFD700",
                                        background: "rgba(255,215,0,0.15)",
                                      }}
                                    />
                                  ))}
                              </Box>
                            </Card>
                          </Grid>
                        );
                      })}
                      {(movies || []).length === 0 && (
                        <Box sx={{ p: 2 }}>
                          <Typography sx={{ color: "rgba(255,255,255,0.7)" }}>
                            Chưa có phim nào.
                          </Typography>
                        </Box>
                      )}
                    </Grid>
                  </Card>
                </Grid>
              </Grid>

              <Dialog
                open={dialogOpen}
                onClose={() => setDialogOpen(false)}
                fullWidth
                maxWidth="sm"
                PaperProps={{ sx: { background: "#0f1422", color: "#fff" } }}
              >
                <DialogTitle>
                  {editing ? "Chỉnh sửa bộ sưu tập" : "Tạo bộ sưu tập"}
                </DialogTitle>
                <DialogContent>
                  <Stack spacing={2} sx={{ mt: 1 }}>
                    <TextField
                      label="Tên"
                      fullWidth
                      value={form.name}
                      onChange={(e) =>
                        setForm((p) => ({ ...p, name: e.target.value }))
                      }
                      InputLabelProps={{
                        sx: { color: "rgba(255,255,255,0.7)" },
                      }}
                      InputProps={{ sx: { color: "#fff" } }}
                    />
                    <TextField
                      label="Mô tả"
                      fullWidth
                      value={form.description}
                      onChange={(e) =>
                        setForm((p) => ({ ...p, description: e.target.value }))
                      }
                      InputLabelProps={{
                        sx: { color: "rgba(255,255,255,0.7)" },
                      }}
                      InputProps={{ sx: { color: "#fff" } }}
                    />
                  </Stack>
                </DialogContent>
                <DialogActions>
                  <Button
                    onClick={() => setDialogOpen(false)}
                    sx={{ color: "#fff" }}
                  >
                    Hủy
                  </Button>
                  <Button
                    onClick={onSubmit}
                    sx={{
                      color: "#000",
                      bgcolor: "#FFD700",
                      fontWeight: 700,
                      ":hover": { bgcolor: "#FFC107" },
                    }}
                  >
                    Lưu
                  </Button>
                </DialogActions>
              </Dialog>
            </MainContent>
          </Grid>
        </Grid>
      </Container>
    </PageContainer>
  );
}
