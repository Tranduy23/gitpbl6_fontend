import { useState, useEffect, useRef } from "react";
import {
  Box,
  Container,
  Typography,
  Button,
  Avatar,
  IconButton,
  Link,
  Divider,
  Paper,
  Grid,
  TextField,
  Input,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import {
  Favorite,
  List,
  History,
  Notifications,
  AccountCircle,
  Logout,
  PhotoCamera,
  Edit,
  Lock,
} from "@mui/icons-material";
import { useAuth } from "../hooks/useAuth";
import { useNavigate } from "react-router-dom";

const PageContainer = styled(Box)(() => ({
  minHeight: "100vh",
  background: "linear-gradient(180deg, #0b1220 0%, #0a0f1a 50%, #0b0d13 100%)",
  color: "#fff",
  paddingTop: 80, // Account for fixed header
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

const UpdateButton = styled(Button)(() => ({
  background: "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
  color: "#000",
  fontWeight: 700,
  borderRadius: 12,
  padding: "12px 24px",
  textTransform: "none",
  fontSize: "1rem",
  "&:hover": {
    background: "linear-gradient(135deg, #FFA500 0%, #FF8C00 100%)",
    boxShadow: "0 8px 24px rgba(255, 215, 0, 0.3)",
  },
}));

const AvatarContainer = styled(Box)(() => ({
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: 16,
  "& .avatar": {
    width: 120,
    height: 120,
    border: "3px solid rgba(255, 215, 0, 0.3)",
    "&:hover": {
      borderColor: "rgba(255, 215, 0, 0.6)",
    },
  },
  "& .avatar-text": {
    color: "rgba(255, 255, 255, 0.7)",
    fontSize: "0.9rem",
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
}));

const StyledTextField = styled(TextField)(() => ({
  "& .MuiOutlinedInput-root": {
    borderRadius: 12,
    background: "rgba(255, 255, 255, 0.05)",
    border: "1px solid rgba(255, 255, 255, 0.15)",
    color: "#fff",
    "&:hover": {
      borderColor: "rgba(255, 215, 0, 0.5)",
    },
    "&.Mui-focused": {
      borderColor: "#FFD700",
      boxShadow: "0 0 0 2px rgba(255, 215, 0, 0.15)",
    },
    "& fieldset": { border: "none" },
    "& input": {
      color: "#fff",
      "&::placeholder": { color: "rgba(255, 255, 255, 0.5)" },
    },
  },
  "& .MuiInputLabel-root": {
    color: "rgba(255, 255, 255, 0.7)",
    "&.Mui-focused": { color: "#FFD700" },
  },
}));

const HiddenInput = styled(Input)(() => ({
  display: "none",
}));

export default function Account() {
  const { updateUser, token, loading: authLoading, user } = useAuth();
  const navigate = useNavigate();
  const [accountData, setAccountData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const hasFetched = useRef(false);
  const [formData, setFormData] = useState({
    fullName: "",
    birthday: "",
  });
  const [avatarFile, setAvatarFile] = useState(null);
  const [avatarPreview, setAvatarPreview] = useState(null);
  const [passwordData, setPasswordData] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: "",
  });
  const [passwordError, setPasswordError] = useState("");
  const [passwordSuccess, setPasswordSuccess] = useState("");
  const [changePasswordOpen, setChangePasswordOpen] = useState(false);

  useEffect(() => {
    // Wait for auth context to finish loading
    if (authLoading) {
      return;
    }

    // Prevent multiple API calls
    if (hasFetched.current) {
      return;
    }

    const fetchAccountData = async () => {
      console.log("Fetching account data...", {
        token: token ? "exists" : "missing",
      });

      if (!token) {
        console.log("No token found, setting loading to false");
        setLoading(false);
        return;
      }

      hasFetched.current = true;

      try {
        setLoading(true);
        setError(null);

        const API_BASE_URL = import.meta.env.VITE_API_URL || "";
        const url = `${API_BASE_URL}/api/auth/me`;
        console.log("Making request to:", url);

        const response = await fetch(url, {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        });

        console.log("Response status:", response.status);

        if (!response.ok) {
          const errorText = await response.text();
          console.error("API Error:", errorText);
          throw new Error(
            `HTTP error! status: ${response.status} - ${errorText}`
          );
        }

        const data = await response.json();
        console.log("Account data received:", data);
        setAccountData(data);

        // Update user context with fresh data
        // Note: updateUser is stable from context, no need to include in deps
        updateUser(data);
      } catch (err) {
        console.error("Error fetching account data:", err);
        setError(`Không thể tải thông tin tài khoản: ${err.message}`);
        hasFetched.current = false; // Reset on error to allow retry
      } finally {
        setLoading(false);
      }
    };

    fetchAccountData();
  }, [token, authLoading, updateUser]);

  // Reset hasFetched when token changes
  useEffect(() => {
    hasFetched.current = false;
  }, [token]);

  // Update form data when account data changes
  useEffect(() => {
    if (accountData) {
      setFormData({
        fullName: accountData.fullName || "",
        birthday: accountData.birthday || "",
      });
    }
  }, [accountData]);

  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handlePasswordChange = (e) => {
    setPasswordData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
    // Clear errors when user starts typing
    if (passwordError) setPasswordError("");
    if (passwordSuccess) setPasswordSuccess("");
  };

  const handleAvatarChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAvatarFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setAvatarPreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const API_BASE_URL = import.meta.env.VITE_API_URL || "";

      // First, update user info if there are changes
      if (
        formData.fullName !== accountData.fullName ||
        formData.birthday !== accountData.birthday
      ) {
        const updateResponse = await fetch(
          `${API_BASE_URL}/api/users/${accountData.id}`,
          {
            method: "PUT",
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              fullName: formData.fullName,
              birthday: formData.birthday,
            }),
          }
        );

        if (!updateResponse.ok) {
          throw new Error(`Failed to update user: ${updateResponse.status}`);
        }

        const updatedUser = await updateResponse.json();
        setAccountData(updatedUser);
        updateUser(updatedUser);
      }

      // Then, upload avatar if there's a new file
      if (avatarFile) {
        const formData = new FormData();
        formData.append("file", avatarFile);

        const avatarResponse = await fetch(
          `${API_BASE_URL}/api/users/${accountData.id}/avatar`,
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${token}`,
            },
            body: formData,
          }
        );

        if (!avatarResponse.ok) {
          throw new Error(`Failed to upload avatar: ${avatarResponse.status}`);
        }

        const userWithAvatar = await avatarResponse.json();
        setAccountData(userWithAvatar);
        updateUser(userWithAvatar);
        setAvatarFile(null);
        setAvatarPreview(null);
      }

      console.log("User updated successfully");
    } catch (error) {
      console.error("Error updating user:", error);
      setError(`Lỗi cập nhật: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleChangePassword = async (e) => {
    e.preventDefault();
    setPasswordError("");
    setPasswordSuccess("");

    // Validation
    if (!passwordData.currentPassword) {
      setPasswordError("Vui lòng nhập mật khẩu hiện tại");
      return;
    }

    if (!passwordData.newPassword) {
      setPasswordError("Vui lòng nhập mật khẩu mới");
      return;
    }

    if (passwordData.newPassword.length < 6) {
      setPasswordError("Mật khẩu mới phải có ít nhất 6 ký tự");
      return;
    }

    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setPasswordError("Mật khẩu xác nhận không khớp");
      return;
    }

    if (passwordData.currentPassword === passwordData.newPassword) {
      setPasswordError("Mật khẩu mới phải khác mật khẩu hiện tại");
      return;
    }

    setLoading(true);

    try {
      const API_BASE_URL = import.meta.env.VITE_API_URL || "";

      const response = await fetch(
        `${API_BASE_URL}/api/users/${accountData.id}/change-password`,
        {
          method: "PUT",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            currentPassword: passwordData.currentPassword,
            newPassword: passwordData.newPassword,
            confirmPassword: passwordData.confirmPassword,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.message || `Failed to change password: ${response.status}`
        );
      }

      const updatedUser = await response.json();
      setAccountData(updatedUser);
      updateUser(updatedUser);

      setPasswordSuccess("Đổi mật khẩu thành công!");
      setPasswordData({
        currentPassword: "",
        newPassword: "",
        confirmPassword: "",
      });

      // Close dialog after 2 seconds
      setTimeout(() => {
        handleCloseChangePassword();
      }, 2000);
    } catch (error) {
      console.error("Error changing password:", error);
      setPasswordError(`Lỗi đổi mật khẩu: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenChangePassword = () => {
    setChangePasswordOpen(true);
    setPasswordError("");
    setPasswordSuccess("");
    setPasswordData({
      currentPassword: "",
      newPassword: "",
      confirmPassword: "",
    });
  };

  const handleCloseChangePassword = () => {
    setChangePasswordOpen(false);
    setPasswordError("");
    setPasswordSuccess("");
    setPasswordData({
      currentPassword: "",
      newPassword: "",
      confirmPassword: "",
    });
  };

  const handleLogout = () => {
    navigate("/");
  };

  const getInitials = (name) => {
    if (!name) return "U";
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const navItems = [
    {
      label: "Yêu thích",
      icon: <Favorite />,
      active: false,
      to: "/favorites",
    },
    { label: "Danh sách", icon: <List />, active: false, to: "/watchlist" },
    {
      label: "Xem tiếp",
      icon: <History />,
      active: false,
      to: "/watch-history",
    },
    { label: "Thông báo", icon: <Notifications />, active: false },
    { label: "Tài khoản", icon: <AccountCircle />, active: true },
  ];

  if (!token) {
    return (
      <PageContainer>
        <Container maxWidth="lg">
          <Typography variant="h4" textAlign="center" sx={{ mt: 8 }}>
            Vui lòng đăng nhập để xem tài khoản
          </Typography>
        </Container>
      </PageContainer>
    );
  }

  if (authLoading || loading) {
    return (
      <PageContainer>
        <Container maxWidth="lg">
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              minHeight: "50vh",
            }}
          >
            <Typography variant="h5" sx={{ color: "#fff" }}>
              {authLoading
                ? "Đang khởi tạo..."
                : "Đang tải thông tin tài khoản..."}
            </Typography>
          </Box>
        </Container>
      </PageContainer>
    );
  }

  if (error) {
    return (
      <PageContainer>
        <Container maxWidth="lg">
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              minHeight: "50vh",
            }}
          >
            <Typography variant="h5" sx={{ color: "#ff6b6b" }}>
              {error}
            </Typography>
          </Box>
        </Container>
      </PageContainer>
    );
  }

  // Use accountData if available, otherwise fallback to user from context
  const displayData = accountData || user;

  if (!displayData) {
    return (
      <PageContainer>
        <Container maxWidth="lg">
          <Typography variant="h4" textAlign="center" sx={{ mt: 8 }}>
            Không có dữ liệu tài khoản
          </Typography>
        </Container>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <Container maxWidth="lg">
        <Grid container spacing={4} sx={{ py: 4 }}>
          {/* Sidebar */}
          <Grid item xs={12} md={3}>
            <Sidebar>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 800,
                  mb: 3,
                  color: "#fff",
                  fontSize: "1.5rem",
                }}
              >
                Quản lý tài khoản
              </Typography>

              {navItems.map((item) => (
                <NavItem
                  key={item.label}
                  active={item.active}
                  onClick={() => item.to && navigate(item.to)}
                >
                  {item.icon}
                  {item.label}
                </NavItem>
              ))}

              <Divider
                sx={{ my: 3, borderColor: "rgba(255, 255, 255, 0.1)" }}
              />

              {/* User Profile Summary */}
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                }}
              >
                <Avatar
                  src={displayData.avatarUrl}
                  sx={{
                    width: 60,
                    height: 60,
                    background: displayData.avatarUrl
                      ? "transparent"
                      : "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
                    color: displayData.avatarUrl ? "transparent" : "#000",
                    fontWeight: 700,
                    fontSize: "1.2rem",
                    mb: 2,
                  }}
                >
                  {!displayData.avatarUrl &&
                    getInitials(displayData.fullName || displayData.username)}
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
                  {displayData.fullName || displayData.username} ∞
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    color: "rgba(255, 255, 255, 0.7)",
                    textAlign: "center",
                    mb: 2,
                    fontSize: "0.85rem",
                  }}
                >
                  {displayData.email}
                </Typography>
                <Link
                  component="button"
                  onClick={handleLogout}
                  sx={{
                    color: "rgba(255, 255, 255, 0.7)",
                    textDecoration: "none",
                    fontSize: "0.9rem",
                    display: "flex",
                    alignItems: "center",
                    gap: 1,
                    "&:hover": { color: "#FFD700" },
                  }}
                >
                  <Logout sx={{ fontSize: "1rem" }} />
                  Thoát
                </Link>
              </Box>

              {/* Change Password Link */}
              <Box sx={{ mt: 4, textAlign: "center" }}>
                <Typography
                  variant="body2"
                  sx={{
                    color: "rgba(255, 255, 255, 0.7)",
                    cursor: "pointer",
                    "&:hover": {
                      color: "#FFD700",
                    },
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 1,
                  }}
                  onClick={handleOpenChangePassword}
                >
                  <Lock sx={{ fontSize: "1rem" }} />
                  Đổi mật khẩu, nhấn vào đây
                </Typography>
              </Box>
            </Sidebar>
          </Grid>

          {/* Main Content */}
          <Grid item xs={12} md={9}>
            <MainContent>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 800,
                  mb: 1,
                  color: "#fff",
                  fontSize: "2rem",
                }}
              >
                Tài khoản
              </Typography>
              <Typography
                variant="body1"
                sx={{
                  color: "rgba(255, 255, 255, 0.7)",
                  mb: 4,
                }}
              >
                Cập nhật thông tin tài khoản
              </Typography>

              <Grid container spacing={4}>
                {/* Form Fields */}
                <Grid item xs={12} md={8}>
                  <Box component="form" onSubmit={handleSubmit}>
                    <Box sx={{ mb: 3 }}>
                      <Typography
                        variant="body1"
                        sx={{
                          color: "#fff",
                          fontWeight: 600,
                          mb: 1,
                        }}
                      >
                        Email
                      </Typography>
                      <Typography
                        variant="body1"
                        sx={{
                          color: "rgba(255, 255, 255, 0.7)",
                          background: "rgba(255, 255, 255, 0.05)",
                          padding: 2,
                          borderRadius: 2,
                          border: "1px solid rgba(255, 255, 255, 0.1)",
                        }}
                      >
                        {displayData.email} (Không thể thay đổi)
                      </Typography>
                    </Box>

                    <Box sx={{ mb: 3 }}>
                      <Typography
                        variant="body1"
                        sx={{
                          color: "#fff",
                          fontWeight: 600,
                          mb: 1,
                        }}
                      >
                        Tên hiển thị
                      </Typography>
                      <StyledTextField
                        fullWidth
                        name="fullName"
                        value={formData.fullName}
                        onChange={handleChange}
                        placeholder="Nhập tên hiển thị"
                        variant="outlined"
                      />
                    </Box>

                    <Box sx={{ mb: 3 }}>
                      <Typography
                        variant="body1"
                        sx={{
                          color: "#fff",
                          fontWeight: 600,
                          mb: 1,
                        }}
                      >
                        Ngày sinh
                      </Typography>
                      <StyledTextField
                        fullWidth
                        name="birthday"
                        type="date"
                        value={formData.birthday}
                        onChange={handleChange}
                        variant="outlined"
                        InputLabelProps={{
                          shrink: true,
                        }}
                      />
                    </Box>

                    <Box sx={{ mb: 3 }}>
                      <Typography
                        variant="body1"
                        sx={{
                          color: "#fff",
                          fontWeight: 600,
                          mb: 1,
                        }}
                      >
                        Ngày tạo tài khoản
                      </Typography>
                      <Typography
                        variant="body1"
                        sx={{
                          color: "#fff",
                          background: "rgba(255, 255, 255, 0.05)",
                          padding: 2,
                          borderRadius: 2,
                          border: "1px solid rgba(255, 255, 255, 0.1)",
                        }}
                      >
                        {displayData.createdAt
                          ? new Date(displayData.createdAt).toLocaleDateString(
                              "vi-VN",
                              {
                                year: "numeric",
                                month: "long",
                                day: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                              }
                            )
                          : "Không có thông tin"}
                      </Typography>
                    </Box>

                    <Box sx={{ mb: 4 }}>
                      <Typography
                        variant="body1"
                        sx={{
                          color: "#fff",
                          fontWeight: 600,
                          mb: 1,
                        }}
                      >
                        Cập nhật lần cuối
                      </Typography>
                      <Typography
                        variant="body1"
                        sx={{
                          color: "#fff",
                          background: "rgba(255, 255, 255, 0.05)",
                          padding: 2,
                          borderRadius: 2,
                          border: "1px solid rgba(255, 255, 255, 0.1)",
                        }}
                      >
                        {displayData.updatedAt
                          ? new Date(displayData.updatedAt).toLocaleDateString(
                              "vi-VN",
                              {
                                year: "numeric",
                                month: "long",
                                day: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                              }
                            )
                          : "Không có thông tin"}
                      </Typography>
                    </Box>

                    {error && (
                      <Typography
                        variant="body2"
                        sx={{
                          color: "#ff6b6b",
                          mb: 2,
                          padding: 2,
                          background: "rgba(255, 107, 107, 0.1)",
                          borderRadius: 2,
                          border: "1px solid rgba(255, 107, 107, 0.3)",
                        }}
                      >
                        {error}
                      </Typography>
                    )}

                    <UpdateButton
                      type="submit"
                      variant="contained"
                      disabled={loading}
                      sx={{ mb: 2 }}
                    >
                      {loading ? "Đang cập nhật..." : "Cập nhật"}
                    </UpdateButton>
                  </Box>
                </Grid>

                {/* Avatar Section */}
                <Grid item xs={12} md={4}>
                  <AvatarContainer>
                    <Avatar
                      src={avatarPreview || displayData.avatarUrl}
                      className="avatar"
                      sx={{
                        background:
                          avatarPreview || displayData.avatarUrl
                            ? "transparent"
                            : "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
                        color:
                          avatarPreview || displayData.avatarUrl
                            ? "transparent"
                            : "#000",
                        fontWeight: 700,
                        fontSize: "2rem",
                      }}
                    >
                      {!(avatarPreview || displayData.avatarUrl) &&
                        getInitials(
                          displayData.fullName || displayData.username
                        )}
                    </Avatar>
                    <Typography className="avatar-text">
                      {avatarFile ? "Ảnh mới" : "Ảnh có sẵn"}
                      <Edit sx={{ fontSize: "1rem" }} />
                    </Typography>
                    <HiddenInput
                      accept="image/*"
                      id="avatar-upload"
                      type="file"
                      onChange={handleAvatarChange}
                    />
                    <label htmlFor="avatar-upload">
                      <IconButton
                        component="span"
                        sx={{
                          color: "#FFD700",
                          background: "rgba(255, 215, 0, 0.1)",
                          "&:hover": {
                            background: "rgba(255, 215, 0, 0.2)",
                          },
                        }}
                      >
                        <PhotoCamera />
                      </IconButton>
                    </label>
                    {avatarFile && (
                      <Typography
                        variant="caption"
                        sx={{
                          color: "#FFD700",
                          textAlign: "center",
                          fontSize: "0.8rem",
                        }}
                      >
                        {avatarFile.name}
                      </Typography>
                    )}
                  </AvatarContainer>
                </Grid>
              </Grid>
            </MainContent>
          </Grid>
        </Grid>
      </Container>

      {/* Change Password Dialog */}
      <Dialog
        open={changePasswordOpen}
        onClose={handleCloseChangePassword}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
            borderRadius: 3,
            border: "1px solid rgba(255, 255, 255, 0.1)",
          },
        }}
      >
        <DialogTitle
          sx={{
            color: "#fff",
            fontWeight: 700,
            fontSize: "1.5rem",
            textAlign: "center",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 1,
          }}
        >
          <Lock sx={{ fontSize: "1.5rem" }} />
          Đổi mật khẩu
        </DialogTitle>

        <DialogContent>
          <Box component="form" onSubmit={handleChangePassword}>
            <Box sx={{ mb: 3 }}>
              <Typography
                variant="body2"
                sx={{
                  color: "rgba(255, 255, 255, 0.8)",
                  mb: 1,
                  fontSize: "0.9rem",
                  fontWeight: 600,
                }}
              >
                Mật khẩu hiện tại
              </Typography>
              <StyledTextField
                fullWidth
                name="currentPassword"
                type="password"
                value={passwordData.currentPassword}
                onChange={handlePasswordChange}
                placeholder="Nhập mật khẩu hiện tại"
                variant="outlined"
              />
            </Box>

            <Box sx={{ mb: 3 }}>
              <Typography
                variant="body2"
                sx={{
                  color: "rgba(255, 255, 255, 0.8)",
                  mb: 1,
                  fontSize: "0.9rem",
                  fontWeight: 600,
                }}
              >
                Mật khẩu mới
              </Typography>
              <StyledTextField
                fullWidth
                name="newPassword"
                type="password"
                value={passwordData.newPassword}
                onChange={handlePasswordChange}
                placeholder="Nhập mật khẩu mới"
                variant="outlined"
              />
            </Box>

            <Box sx={{ mb: 3 }}>
              <Typography
                variant="body2"
                sx={{
                  color: "rgba(255, 255, 255, 0.8)",
                  mb: 1,
                  fontSize: "0.9rem",
                  fontWeight: 600,
                }}
              >
                Xác nhận mật khẩu mới
              </Typography>
              <StyledTextField
                fullWidth
                name="confirmPassword"
                type="password"
                value={passwordData.confirmPassword}
                onChange={handlePasswordChange}
                placeholder="Nhập lại mật khẩu mới"
                variant="outlined"
              />
            </Box>

            {passwordError && (
              <Typography
                variant="body2"
                sx={{
                  color: "#ff6b6b",
                  mb: 2,
                  padding: 2,
                  background: "rgba(255, 107, 107, 0.1)",
                  borderRadius: 2,
                  border: "1px solid rgba(255, 107, 107, 0.3)",
                  fontSize: "0.9rem",
                }}
              >
                {passwordError}
              </Typography>
            )}

            {passwordSuccess && (
              <Typography
                variant="body2"
                sx={{
                  color: "#4caf50",
                  mb: 2,
                  padding: 2,
                  background: "rgba(76, 175, 80, 0.1)",
                  borderRadius: 2,
                  border: "1px solid rgba(76, 175, 80, 0.3)",
                  fontSize: "0.9rem",
                }}
              >
                {passwordSuccess}
              </Typography>
            )}
          </Box>
        </DialogContent>

        <DialogActions sx={{ p: 3, gap: 2 }}>
          <Button
            onClick={handleCloseChangePassword}
            sx={{
              color: "rgba(255, 255, 255, 0.7)",
              border: "1px solid rgba(255, 255, 255, 0.2)",
              "&:hover": {
                borderColor: "rgba(255, 255, 255, 0.4)",
                background: "rgba(255, 255, 255, 0.05)",
              },
            }}
          >
            Hủy
          </Button>
          <Button
            onClick={handleChangePassword}
            variant="contained"
            disabled={loading}
            sx={{
              background: "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
              color: "#000",
              fontWeight: 600,
              borderRadius: 2,
              textTransform: "none",
              px: 3,
              "&:hover": {
                background: "linear-gradient(135deg, #FFA500 0%, #FF8C00 100%)",
                boxShadow: "0 4px 12px rgba(255, 215, 0, 0.3)",
              },
              "&:disabled": {
                background: "rgba(255, 255, 255, 0.1)",
                color: "rgba(255, 255, 255, 0.5)",
              },
            }}
          >
            {loading ? "Đang xử lý..." : "Đổi mật khẩu"}
          </Button>
        </DialogActions>
      </Dialog>
    </PageContainer>
  );
}
