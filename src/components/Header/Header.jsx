import { useState, useEffect } from "react";
import {
  Box,
  Container,
  IconButton,
  InputBase,
  Paper,
  Button,
  Badge,
  Tooltip,
  Popover,
  Chip,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import { Link, useNavigate } from "react-router-dom";
import MovieIcon from "@mui/icons-material/Movie";
import SearchIcon from "@mui/icons-material/Search";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import NotificationsIcon from "@mui/icons-material/Notifications";
import MenuIcon from "@mui/icons-material/Menu";

import { NAV_ITEMS, GENRES } from "../../constants";
import { useAuth } from "../../hooks/useAuth";
import UserAvatar from "../UserAvatar";
import { quickSearch, getSearchSuggestions } from "../../api/streaming";

const StyledHeader = styled("header", {
  shouldForwardProp: (prop) => prop !== "scrolled",
})(({ scrolled }) => ({
  position: "fixed",
  inset: "0 0 auto 0",
  zIndex: 1000,
  background: scrolled ? "rgba(0,0,0,0.9)" : "rgba(0,0,0,0.35)",
  backdropFilter: scrolled ? "blur(14px)" : "blur(10px)",
  borderBottom: "1px solid rgba(255,255,255,0.12)",
  transition: "all .28s ease",
}));

const Logo = styled("div")(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  gap: theme.spacing(1.5),
  cursor: "pointer",
  transition: "transform 0.2s ease",
  "&:hover": {
    transform: "scale(1.02)",
  },
  "& .logo-icon": {
    width: 48,
    height: 48,
    borderRadius: "12px",
    background:
      "linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FF8C00 100%)",
    display: "grid",
    placeItems: "center",
    color: "#000",
    fontWeight: 800,
    boxShadow:
      "0 8px 24px rgba(255, 215, 0, .5), 0 0 0 1px rgba(255, 255, 255, .1)",
    position: "relative",
    overflow: "hidden",
    "&::before": {
      content: '""',
      position: "absolute",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background:
        "linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.2) 50%, transparent 70%)",
      transform: "translateX(-100%)",
      transition: "transform 0.6s ease",
    },
    "&:hover::before": {
      transform: "translateX(100%)",
    },
  },
  "& .main-text": {
    color: "#fff",
    fontSize: "1.9rem",
    fontWeight: 900,
    letterSpacing: "-0.5px",
    textShadow: "0 2px 12px rgba(0,0,0,.8), 0 0 20px rgba(255,215,0,.3)",
    lineHeight: 1,
    background: "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)",
    backgroundClip: "text",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
  },
  "& .sub-text": {
    color: "rgba(255,255,255,.85)",
    fontSize: ".9rem",
    letterSpacing: ".3px",
    marginTop: 2,
    fontWeight: 500,
    textShadow: "0 1px 4px rgba(0,0,0,.6)",
  },
}));

const SearchBar = styled(Paper)(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  background: "rgba(255,255,255,.10)",
  borderRadius: 28,
  padding: theme.spacing(0.75, 2),
  maxWidth: 380,
  border: "1 px solid rgba(255,255,255,.16)",
  backdropFilter: "blur(10px)",
  transition: "all .25s ease",
  "&:hover": {
    background: "rgba(255,255,255,.14)",
    borderColor: "rgba(255,215,0,.45)",
  },
  "&:focus-within": {
    background: "rgba(255,255,255,.18)",
    borderColor: "rgba(255,215,0,.65)",
    boxShadow: "0 0 18px rgba(255,215,0,.28)",
  },
  "& .MuiInputBase-input": {
    color: "rgba(255,255,255,.95)",
    fontSize: ".98rem",
    "&::placeholder": { color: "rgba(255,255,255,.64)", opacity: 1 },
  },
  "& .MuiSvgIcon-root": { color: "rgba(255,255,255,.82)" },
}));

const NavLink = styled("a")(({ theme }) => ({
  color: "rgba(255,255,255,.9)",
  textDecoration: "none",
  padding: theme.spacing(1, 1.6),
  borderRadius: 12,
  display: "flex",
  alignItems: "center",
  gap: 6,
  fontSize: ".95rem",
  fontWeight: 600,
  transition: "all .22s ease",
  "&:hover": {
    color: "#FFD700",
    background: "rgba(255,215,0,.14)",
    transform: "translateY(-1px)",
  },
}));

const Header = () => {
  const [scrolled, setScrolled] = useState(false);
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const [query, setQuery] = useState("");
  const [genreAnchor, setGenreAnchor] = useState(null);
  const [countryAnchor, setCountryAnchor] = useState(null);
  const [dbGenres, setDbGenres] = useState([]);
  const [dbCountries, setDbCountries] = useState([]);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Fetch genres and countries from database by sampling movies
  useEffect(() => {
    let aborted = false;
    (async () => {
      try {
        const res = await fetch(`/api/movies?page=0&size=500&sort=title,asc`, {
          method: "GET",
          headers: { Accept: "*/*", "Content-Type": "application/json" },
        });
        if (!res.ok) throw new Error("failed");
        const data = await res.json();
        const list = Array.isArray(data) ? data : data?.content || [];
        const categories = new Set();
        const countries = new Set();
        list.forEach((m) => {
          (m?.categories || []).forEach((c) => categories.add(c));
          if (m?.country) countries.add(m.country);
        });
        if (!aborted) {
          setDbGenres(Array.from(categories));
          setDbCountries(Array.from(countries));
        }
      } catch (_) {
        if (!aborted) {
          setDbGenres([]);
          setDbCountries([]);
        }
      }
    })();
    return () => {
      aborted = true;
    };
  }, []);

  // Debounced background search calls (no UI yet)
  useEffect(() => {
    if (!query || query.trim().length < 2) return;
    const handle = setTimeout(async () => {
      try {
        await Promise.all([
          quickSearch(query, 5).catch(() => undefined),
          getSearchSuggestions(query, 6).catch(() => undefined),
        ]);
      } catch (_) {
        // ignore
      }
    }, 300);
    return () => {
      clearTimeout(handle);
    };
  }, [query]);

  const submitSearch = () => {
    const q = query.trim();
    if (!q) return;
    const params = new URLSearchParams({ query: q });
    navigate(`/movies?${params.toString()}`);
  };

  const openGenres = (event) => setGenreAnchor(event.currentTarget);
  const closeGenres = () => setGenreAnchor(null);
  const genresOpen = Boolean(genreAnchor);
  const genresToShow = (dbGenres && dbGenres.length > 0 ? dbGenres : GENRES)
    .slice()
    .sort((a, b) => a.localeCompare(b, "vi"));

  const openCountries = (event) => setCountryAnchor(event.currentTarget);
  const closeCountries = () => setCountryAnchor(null);
  const countriesOpen = Boolean(countryAnchor);
  const countriesToShow = (dbCountries || [])
    .slice()
    .sort((a, b) => String(a).localeCompare(String(b), "vi"));

  const handleSelectGenre = (name) => {
    closeGenres();
    navigate(`/movies?genre=${encodeURIComponent(name)}`);
  };
  const handleSelectCountry = (name) => {
    closeCountries();
    navigate(`/movies?country=${encodeURIComponent(name)}`);
  };

  return (
    <StyledHeader scrolled={scrolled}>
      <Container maxWidth={false} sx={{ px: { xs: 2, md: 4, lg: 6 } }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            py: 2,
            gap: 2,
          }}
        >
          {/* Logo */}
          <Logo onClick={() => navigate("/")}>
            <div className="logo-icon">
              <MovieIcon sx={{ fontSize: 26 }} />
            </div>
            <div>
              <div className="main-text">PhimNhaLam</div>
              <div className="sub-text">Phim hay nhà dịch</div>
            </div>
          </Logo>

          {/* Search */}
          <SearchBar component="label" htmlFor="search" sx={{ ml: 2 }}>
            <SearchIcon sx={{ mr: 1.25 }} onClick={submitSearch} />
            <InputBase
              id="search"
              placeholder="Tìm kiếm phim, diễn viên…"
              sx={{ flex: 1 }}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  submitSearch();
                }
              }}
            />
          </SearchBar>

          {/* Nav */}
          <Box
            sx={{
              display: { xs: "none", lg: "flex" },
              alignItems: "center",
              gap: 2,
              flex: 1,
              justifyContent: "center",
            }}
          >
            {NAV_ITEMS.map((item) => {
              const isGenre = item.label === "Thể loại";
              const isCountry = item.label === "Quốc gia";
              const isActors = item.label === "Diễn Viên";
              return (
                <NavLink
                  key={item.label}
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    if (item.label === "Phim hay") {
                      navigate("/movies");
                    } else if (isGenre) {
                      openGenres(e);
                    } else if (isCountry) {
                      openCountries(e);
                    } else if (isActors) {
                      navigate("/actors");
                    } else if (item.label === "Lịch chiếu") {
                      navigate("/showtime");
                    }
                  }}
                >
                  {item.label}{" "}
                  {item.hasDropdown && (
                    <KeyboardArrowDownIcon sx={{ fontSize: 18 }} />
                  )}
                </NavLink>
              );
            })}
          </Box>

          <Box sx={{ display: "flex", alignItems: "center" }}>
            <Tooltip title="Thông báo">
              <IconButton sx={{ color: "rgba(255,255,255,.9)", mr: 2 }}>
                <Badge color="warning" variant="dot">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
            </Tooltip>

            {isAuthenticated ? (
              <UserAvatar />
            ) : (
              <Button
                component={Link}
                to="/login"
                variant="outlined"
                sx={{
                  color: "rgba(255,255,255,.9)",
                  borderColor: "rgba(255,255,255,.22)",
                  "&:hover": {
                    borderColor: "rgba(255,215,0,.38)",
                    color: "#FFD700",
                  },
                  borderRadius: 2,
                  px: 2,
                  py: 0.9,
                  fontWeight: 600,
                  textTransform: "none",
                  mr: 1,
                }}
              >
                Thành Viên
              </Button>
            )}

            <IconButton
              sx={{ color: "rgba(255,255,255,.9)", display: { lg: "none" } }}
            >
              <MenuIcon />
            </IconButton>
          </Box>
        </Box>
      </Container>

      {/* Genres Popover */}
      <Popover
        open={genresOpen}
        anchorEl={genreAnchor}
        onClose={closeGenres}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
        transformOrigin={{ vertical: "top", horizontal: "center" }}
        PaperProps={{
          sx: {
            mt: 1,
            background: "rgba(20,20,20,0.98)",
            border: "1px solid rgba(255,255,255,0.1)",
            boxShadow: "0 20px 40px rgba(0,0,0,0.6)",
            p: 2,
            borderRadius: 2,
            minWidth: 680,
            maxWidth: 920,
            maxHeight: 420,
            overflow: "auto",
          },
        }}
        onMouseLeave={closeGenres}
      >
        <Box
          sx={{
            display: "flex",
            flexWrap: "wrap",
            gap: 0.75,
            width: "100%",
          }}
        >
          {genresToShow.map((g) => (
            <Chip
              key={g}
              label={g}
              onClick={() => handleSelectGenre(g)}
              clickable
              sx={{
                color: "rgba(255,255,255,0.9)",
                background: "rgba(255,255,255,0.08)",
                border: "1px solid rgba(255,255,255,0.12)",
                "&:hover": {
                  background: "rgba(255,215,0,0.12)",
                  color: "#FFD700",
                  borderColor: "rgba(255,215,0,0.4)",
                },
              }}
            />
          ))}
        </Box>
      </Popover>

      {/* Countries Popover */}
      <Popover
        open={countriesOpen}
        anchorEl={countryAnchor}
        onClose={closeCountries}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
        transformOrigin={{ vertical: "top", horizontal: "center" }}
        PaperProps={{
          sx: {
            mt: 1,
            background: "rgba(20,20,20,0.98)",
            border: "1px solid rgba(255,255,255,0.1)",
            boxShadow: "0 20px 40px rgba(0,0,0,0.6)",
            p: 2,
            borderRadius: 2,
            minWidth: 480,
            maxWidth: 800,
            maxHeight: 420,
            overflow: "auto",
          },
        }}
        onMouseLeave={closeCountries}
      >
        <Box
          sx={{
            display: "flex",
            flexWrap: "wrap",
            gap: 0.75,
            width: "100%",
          }}
        >
          {countriesToShow.map((c) => (
            <Chip
              key={c}
              label={c}
              onClick={() => handleSelectCountry(c)}
              clickable
              sx={{
                color: "rgba(255,255,255,0.9)",
                background: "rgba(255,255,255,0.08)",
                border: "1px solid rgba(255,255,255,0.12)",
                "&:hover": {
                  background: "rgba(255,215,0,0.12)",
                  color: "#FFD700",
                  borderColor: "rgba(255,215,0,0.4)",
                },
              }}
            />
          ))}
        </Box>
      </Popover>
    </StyledHeader>
  );
};

export default Header;
