import { useState, useEffect, useMemo } from "react";
import {
  Box,
  Container,
  Typography,
  Button,
  Avatar,
  IconButton,
  Divider,
  Paper,
  Grid,
  CircularProgress,
  Tooltip,
  Chip,
  Menu,
  MenuItem,
  Link,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import {
  Favorite,
  List,
  History,
  Notifications,
  AccountCircle,
  Logout,
  PlayArrow,
  Delete,
  ClearAll,
} from "@mui/icons-material";
import { useAuth } from "../hooks/useAuth";
import { useNavigate } from "react-router-dom";
import { getRecentWatchHistory, getIncompleteWatchHistory, getWatchHistory, deleteWatchHistoryItem, clearWatchHistory } from "../api/watchHistory";

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
  padding: 32,
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

const ProgressFill = styled(Box)(({ percentage }) => ({
  width: `${percentage}%`,
  height: "100%",
  background: "linear-gradient(90deg, #FFD700 0%, #FFA500 100%)",
  transition: "width 0.3s ease",
}));

export default function WatchHistory() {
  const { user, token } = useAuth();
  const navigate = useNavigate();
  const [watchHistory, setWatchHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [anchorMenu, setAnchorMenu] = useState(null);
  const [selectedItem, setSelectedItem] = useState(null);

  const navItems = useMemo(
    () => [
      { label: "Yêu thích", icon: <Favorite />, to: "/favorites" },
      { label: "Danh sách", icon: <List />, to: "/watchlist" },
      {
        label: "Xem tiếp",
        icon: <History />,
        to: "/watch-history",
        active: true,
      },
      { label: "Thông báo", icon: <Notifications />, to: "/account" },
      { label: "Tài khoản", icon: <AccountCircle />, to: "/account" },
    ],
    []
  );

  useEffect(() => {
    const loadWatchHistory = async () => {
      if (!token) {
        navigate("/login");
        return;
      }
      try {
        setLoading(true);
        // Get watch history and filter incomplete items
        const data = await getWatchHistory(0, 20, token);
        console.log("Watch history API response:", data);
        
        // Parse response - handle both array and object with content property
        let items = [];
        if (Array.isArray(data)) {
          items = data;
        } else if (data?.content && Array.isArray(data.content)) {
          items = data.content;
        } else if (data?.content) {
          // In case content is not an array
          items = Array.isArray(data.content) ? data.content : [];
        }
        
        // Filter to only show incomplete items (isCompleted: false) for "continue watching"
        items = items.filter((item) => item.isCompleted === false);
        
        console.log("Parsed watch history items (incomplete only):", items);
        setWatchHistory(items);
      } catch (err) {
        console.error("Error loading watch history:", err);
        setWatchHistory([]);
      } finally {
        setLoading(false);
      }
    };
    loadWatchHistory();
  }, [token, navigate]);

  const handleMovieClick = (movieId, watchDurationSeconds) => {
    navigate(`/stream/${movieId}`, {
      state: {
        startPosition: watchDurationSeconds || 0,
      },
    });
  };

  const handleMenuOpen = (event, item) => {
    event.stopPropagation();
    setAnchorMenu(event.currentTarget);
    setSelectedItem(item);
  };

  const handleMenuClose = () => {
    setAnchorMenu(null);
    setSelectedItem(null);
  };

  const handleDelete = async () => {
    if (!selectedItem || !token) return;
    try {
      await deleteWatchHistoryItem(selectedItem.id, token);
      setWatchHistory((prev) => prev.filter((item) => item.id !== selectedItem.id));
      handleMenuClose();
    } catch (err) {
      console.error("Error deleting watch history item:", err);
    }
  };

  const handleClearAll = async () => {
    if (!token) return;
    if (!window.confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử xem?")) return;
    try {
      await clearWatchHistory(token);
      setWatchHistory([]);
    } catch (err) {
      console.error("Error clearing watch history:", err);
    }
  };

  const formatTime = (seconds) => {
    if (!seconds) return "0:00";
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hours > 0) {
      return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    }
    return `${minutes}:${String(secs).padStart(2, "0")}`;
  };

  const formatDate = (dateString) => {
    if (!dateString) return "";
    const date = new Date(dateString);
    return date.toLocaleDateString("vi-VN", {
      day: "numeric",
      month: "long",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const buildPosterUrl = (url) => {
    if (!url) return null;
    if (url.startsWith("http")) return url;
    if (url.startsWith("/api")) return url;
    return `/api${url}`;
  };

  if (!token) {
    return (
      <PageContainer>
        <Container maxWidth="lg">
          <Typography variant="h5" sx={{ mt: 10, textAlign: "center" }}>
            Vui lòng đăng nhập để xem lịch sử xem
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
                >
                  {!user?.avatarUrl &&
                    (user?.fullName || user?.username || "U")
                      .split(" ")
                      .map((w) => w.charAt(0))
                      .join("")
                      .toUpperCase()
                      .slice(0, 2)}
                </Avatar>
                <Typography
                  variant="h6"
                  sx={{
                    color: "#fff",
                    fontWeight: 700,
                    textAlign: "center",
                    mb: 0.5,
                  }}
                >
                  {user?.fullName || user?.username} ∞
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
              <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
                <Box>
                  <Typography
                    variant="h4"
                    sx={{ fontWeight: 800, mb: 1, color: "#fff" }}
                  >
                    Xem tiếp
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{ color: "rgba(255,255,255,0.7)", mb: 3 }}
                  >
                    Tiếp tục xem các phim bạn đã bắt đầu
                  </Typography>
                </Box>
                {watchHistory.length > 0 && (
                  <Button
                    startIcon={<ClearAll />}
                    onClick={handleClearAll}
                    sx={{
                      color: "rgba(255, 255, 255, 0.8)",
                      textTransform: "none",
                      "&:hover": {
                        background: "rgba(255, 215, 0, 0.1)",
                        color: "#FFD700",
                      },
                    }}
                  >
                    Xóa tất cả
                  </Button>
                )}
              </Box>

              {loading && (
                <Typography sx={{ color: "#fff", mb: 2 }}>
                  Đang tải...
                </Typography>
              )}

              {!loading && watchHistory.length === 0 && (
                <Typography sx={{ color: "rgba(255,255,255,0.7)" }}>
                  Chưa có lịch sử xem.
                </Typography>
              )}

              {!loading && watchHistory.length > 0 && (
                <Grid container spacing={2}>
                  {watchHistory.map((item) => (
                    <Grid item xs={6} sm={4} md={3} key={item.id}>
                      <Box
                        sx={{
                          position: "relative",
                          borderRadius: 2,
                          overflow: "hidden",
                          border: "1px solid rgba(255,255,255,0.12)",
                          background: "rgba(255,255,255,0.04)",
                          cursor: "pointer",
                        }}
                        onClick={() => handleMovieClick(item.movieId, item.watchDurationSeconds)}
                      >
                        <Box
                          component="img"
                          src={
                            buildPosterUrl(item.moviePosterUrl) ||
                            `https://via.placeholder.com/300x420/333/fff?text=${encodeURIComponent(
                              item.movieTitle || "Movie"
                            )}`
                          }
                          alt={item.movieTitle}
                          sx={{
                            width: "100%",
                            height: "auto",
                            aspectRatio: "2 / 3",
                            objectFit: "cover",
                            display: "block",
                          }}
                          onError={(e) => {
                            e.currentTarget.src = `https://via.placeholder.com/300x420/333/fff?text=${encodeURIComponent(
                              item.movieTitle || "Movie"
                            )}`;
                          }}
                        />

                        {/* Progress overlay */}
                        {item.watchPercentage && (
                          <Box
                            sx={{
                              position: "absolute",
                              bottom: 0,
                              left: 0,
                              right: 0,
                              height: 4,
                              background: "rgba(0, 0, 0, 0.5)",
                            }}
                          >
                            <ProgressFill percentage={item.watchPercentage || 0} />
                          </Box>
                        )}

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
                                handleMovieClick(item.movieId, item.watchDurationSeconds);
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

                          <Tooltip title="Xóa khỏi lịch sử">
                            <IconButton
                              onClick={(e) => {
                                e.stopPropagation();
                                handleMenuOpen(e, item);
                              }}
                              sx={{
                                background: "rgba(255,255,255,0.15)",
                                color: "#fff",
                                "&:hover": {
                                  background: "rgba(255,255,255,0.3)",
                                },
                              }}
                            >
                              <Delete />
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
                            {item.movieTitle}
                          </Typography>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 1,
                              mt: 1,
                            }}
                          >
                            {item.watchPercentage && (
                              <Chip
                                label={`${Math.round(item.watchPercentage)}%`}
                                size="small"
                                sx={{
                                  height: 20,
                                  fontSize: "0.7rem",
                                  color: "#FFD700",
                                  background: "rgba(255,215,0,0.15)",
                                }}
                              />
                            )}
                            <Typography
                              variant="caption"
                              sx={{
                                color: "rgba(255,255,255,0.6)",
                                fontSize: "0.7rem",
                              }}
                            >
                              {formatTime(item.watchDurationSeconds)} / {formatTime(item.totalDurationSeconds)}
                            </Typography>
                          </Box>
                        </Box>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              )}
            </MainContent>
          </Grid>
        </Grid>
      </Container>

      {/* Menu for actions */}
      <Menu
        anchorEl={anchorMenu}
        open={Boolean(anchorMenu)}
        onClose={handleMenuClose}
        PaperProps={{
          sx: {
            background: "rgba(18, 18, 18, 0.95)",
            border: "1px solid rgba(255, 255, 255, 0.12)",
            color: "#fff",
          },
        }}
      >
        <MenuItem
          onClick={handleDelete}
          sx={{
            color: "rgba(255, 255, 255, 0.9)",
            "&:hover": {
              background: "rgba(255, 215, 0, 0.1)",
              color: "#FFD700",
            },
          }}
        >
          <Delete sx={{ mr: 1, fontSize: 18 }} />
          Xóa khỏi lịch sử
        </MenuItem>
      </Menu>
    </PageContainer>
  );
}

