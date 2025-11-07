import { useState } from "react";
import {
  Avatar,
  Box,
  Menu,
  MenuItem,
  Typography,
  Divider,
  IconButton,
  Tooltip,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import {
  Favorite,
  List,
  History,
  AccountCircle,
  Logout,
  AdminPanelSettings,
} from "@mui/icons-material";
import { useAuth } from "../../hooks/useAuth";
import { Link } from "react-router-dom";

const StyledMenu = styled(Menu)(({ theme }) => ({
  "& .MuiPaper-root": {
    background: "rgba(18, 18, 18, 0.95)",
    backdropFilter: "blur(20px)",
    border: "1px solid rgba(255, 255, 255, 0.12)",
    borderRadius: 16,
    minWidth: 280,
    marginTop: theme.spacing(1),
    boxShadow: "0 20px 40px rgba(0, 0, 0, 0.4)",
  },
}));

const UserInfo = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  "& .user-name": {
    color: "#fff",
    fontSize: "1.1rem",
    fontWeight: 700,
    marginBottom: theme.spacing(0.5),
  },
}));

const MenuItemStyled = styled(MenuItem)(({ theme }) => ({
  color: "rgba(255, 255, 255, 0.9)",
  padding: theme.spacing(1.5, 2),
  borderRadius: 8,
  margin: theme.spacing(0.5, 1),
  "&:hover": {
    background: "rgba(255, 215, 0, 0.1)",
    color: "#FFD700",
  },
  "& .MuiSvgIcon-root": {
    marginRight: theme.spacing(1.5),
    fontSize: "1.2rem",
  },
}));

const UserAvatar = () => {
  const { user, logout } = useAuth();
  const [anchorEl, setAnchorEl] = useState(null);
  const open = Boolean(anchorEl);

  // Check if user is admin
  const isAdmin = user?.roles?.includes("ADMIN");

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    handleClose();
  };

  if (!user) {
    return null;
  }

  const getInitials = (name) => {
    if (!name) return "U";
    return name
      .split(" ")
      .map((word) => word.charAt(0))
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <>
      <Tooltip title="Tài khoản">
        <IconButton
          onClick={handleClick}
          sx={{
            p: 0,
            "&:hover": {
              transform: "scale(1.05)",
            },
            transition: "transform 0.2s ease",
          }}
        >
          <Avatar
            src={user.avatarUrl}
            sx={{
              width: 40,
              height: 40,
              background: user.avatarUrl
                ? "transparent"
                : "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
              color: user.avatarUrl ? "transparent" : "#000",
              fontWeight: 700,
              fontSize: "0.9rem",
              border: "2px solid rgba(255, 215, 0, 0.3)",
              "&:hover": {
                borderColor: "rgba(255, 215, 0, 0.6)",
              },
            }}
          >
            {!user.avatarUrl && getInitials(user.fullName || user.username)}
          </Avatar>
        </IconButton>
      </Tooltip>

      <StyledMenu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "right",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "right",
        }}
      >
        <UserInfo>
          <Typography className="user-name">
            {user.fullName || user.username} ∞
          </Typography>
        </UserInfo>

        <Divider sx={{ borderColor: "rgba(255, 255, 255, 0.1)" }} />

        <MenuItemStyled
          component={Link}
          to="/favorites"
          onClick={handleClose}
        >
          <Favorite />
          Yêu thích
        </MenuItemStyled>

        <MenuItemStyled
          component={Link}
          to="/watchlist"
          onClick={handleClose}
        >
          <List />
          Danh sách
        </MenuItemStyled>

        <MenuItemStyled onClick={handleClose}>
          <History />
          Xem tiếp
        </MenuItemStyled>

        <MenuItemStyled component={Link} to="/account" onClick={handleClose}>
          <AccountCircle />
          Tài khoản
        </MenuItemStyled>

        {isAdmin && (
          <MenuItemStyled component={Link} to="/admin" onClick={handleClose}>
            <AdminPanelSettings />
            Quản lý
          </MenuItemStyled>
        )}

        <Divider sx={{ borderColor: "rgba(255, 255, 255, 0.1)" }} />

        <MenuItemStyled onClick={handleLogout}>
          <Logout />
          Thoát
        </MenuItemStyled>
      </StyledMenu>
    </>
  );
};

export default UserAvatar;
