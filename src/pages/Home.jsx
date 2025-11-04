import { useState, useCallback, useEffect, useRef } from "react";
import { Box, Snackbar, Alert } from "@mui/material";
import { useNavigate } from "react-router-dom";

// Components
import Header from "../components/Header";
import Hero from "../components/Hero";
import MovieThumbnails from "../components/MovieThumbnails";
import Background from "../components/Background";
import FeaturedMovies from "../components/FeaturedMovies";
import CatalogSection from "../components/CatalogSection";
import TopComments from "../components/TopComments";
import CallToAction from "../components/CallToAction";
import Footer from "../components/Footer";

// Hooks
import useMovies from "../hooks/useMovies";

// Utils
import { toHero } from "../utils/imageUtils";

// API
import { getFeaturedMovies, getTrendingMovies } from "../api/streaming";

export default function Home() {
  const { movies } = useMovies();
  const [activeIndex, setActiveIndex] = useState(0);
  const [trendingMovies, setTrendingMovies] = useState([]);
  const displayMovies = trendingMovies; // show thumbnails only for trending
  const activeMovie =
    displayMovies && displayMovies.length > 0
      ? displayMovies[activeIndex]
      : undefined;
  const navigate = useNavigate();
  const [featuredMovies, setFeaturedMovies] = useState([]);

  // Ẩn/hiện thumbnails khi rời Hero
  const [showThumb, setShowThumb] = useState(true);
  const [toast, setToast] = useState({
    open: false,
    message: "",
    severity: "success",
  });

  // Tham chiếu tới section Hero để đo chiều cao
  const heroRef = useRef(null);

  const handleMovieSelect = useCallback((index) => setActiveIndex(index), []);

  const handleSignUp = useCallback(() => {
    console.log("Sign up clicked");
  }, []);
  // Fetch trending movies for thumbnails/Hero
  useEffect(() => {
    const fetchTrending = async () => {
      try {
        const data = await getTrendingMovies(10);
        const list = Array.isArray(data)
          ? data
          : Array.isArray(data?.items)
          ? data.items
          : Array.isArray(data?.data)
          ? data.data
          : [];

        const transformed = list.map((movie, i) => {
          let posterUrl =
            movie.posterUrl || movie.thumbnailUrl || movie.thumb || "";

          // Normalize and prefix relative paths
          if (posterUrl && posterUrl.startsWith("/uploads")) {
            posterUrl = `/api${posterUrl}`;
          } else if (
            posterUrl &&
            !posterUrl.startsWith("http") &&
            !posterUrl.startsWith("/api")
          ) {
            posterUrl = `/api${posterUrl}`;
          }

          // Ensure non-empty src so thumbnails render even without images
          if (!posterUrl) {
            posterUrl = `https://via.placeholder.com/300x300/333/fff?text=${encodeURIComponent(
              movie.title || `Movie ${i + 1}`
            )}`;
          }

          return {
            id: movie.id ?? i + 1,
            title: movie.title || movie.englishTitle || `Movie ${i + 1}`,
            thumb: posterUrl,
            rating:
              movie.imdbRating?.toString() ||
              movie.averageRating?.toString() ||
              "8.5",
            year: movie.year?.toString() || "2024",
            genres: movie.categories || movie.genres || [],
            synopsis: movie.synopsis || movie.overview || "",
            imdbRating: movie.imdbRating,
          };
        });
        setTrendingMovies(transformed);
        setActiveIndex(0);
      } catch (err) {
        console.error("Error fetching trending movies:", err);
      }
    };
    fetchTrending();
  }, []);

  const handlePlay = useCallback(
    (movie) => {
      if (!movie?.id) return;
      navigate(`/stream/${movie.id}`);
    },
    [navigate]
  );

  const handleInfo = useCallback(
    (movie) => {
      if (!movie?.id) return;
      navigate(`/movie/${movie.id}`);
    },
    [navigate]
  );

  const handleFavorite = useCallback(async (movie) => {
    try {
      if (!movie?.id) return;
      const { addFavorite } = await import("../api/favorites.js");
      await addFavorite(movie.id);
      window.dispatchEvent(
        new CustomEvent("app:toast", {
          detail: { message: "Đã thêm vào Yêu thích", severity: "success" },
        })
      );
    } catch (err) {
      window.dispatchEvent(
        new CustomEvent("app:toast", {
          detail: { message: "Không thể thêm Yêu thích", severity: "error" },
        })
      );
    }
  }, []);

  useEffect(() => {
    const onScroll = () => {
      const heroEl = heroRef.current;
      if (!heroEl) return;

      // Lấy vị trí đáy của Hero so với viewport
      const rect = heroEl.getBoundingClientRect();
      // Khi đáy Hero đi lên khỏi viewport (<= 0) thì ẩn
      // Thêm buffer 120px để ẩn sớm, tránh đè nội dung
      const buffer = 120;
      const shouldShow = rect.bottom > buffer;

      setShowThumb(shouldShow);
    };

    // Gọi 1 lần khi mount để đồng bộ trạng thái ban đầu
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  // Fetch featured movies from API
  useEffect(() => {
    const fetchFeaturedMovies = async () => {
      try {
        console.log("Fetching featured movies...");
        const data = await getFeaturedMovies(10);
        console.log("Featured movies API response:", data);

        // Transform API response to match FeaturedMovies component format
        const transformed = (data || []).map((movie) => {
          // Build full poster URL if it's a relative path
          let posterUrl = movie.posterUrl || "";
          if (
            posterUrl &&
            !posterUrl.startsWith("http") &&
            !posterUrl.startsWith("/")
          ) {
            posterUrl = `/api${posterUrl}`;
          }
          // If it already starts with /api, use it as is

          return {
            id: movie.id,
            title: movie.title,
            thumb: posterUrl,
            rating:
              movie.imdbRating?.toString() ||
              movie.averageRating?.toString() ||
              "8.5",
            year: movie.year?.toString() || "2024",
            genres: movie.categories || [],
            synopsis: movie.synopsis || "",
            imdbRating: movie.imdbRating,
            averageRating: movie.averageRating,
            isFavorite: movie.isFavorite || false,
            isInWatchlist: movie.isInWatchlist || false,
          };
        });

        console.log("Transformed featured movies:", transformed);
        setFeaturedMovies(transformed);
      } catch (err) {
        console.error("Error fetching featured movies:", err);
        // On error, fallback to regular movies filtered by isFeatured
        const fallback = movies
          .filter((m) => m.isFeatured)
          .slice(0, 10)
          .map((m) => ({
            id: m.id,
            title: m.title,
            thumb: m.thumb || "",
            rating:
              m.imdbRating?.toString() || m.averageRating?.toString() || "8.5",
            year: m.year || "2024",
            genres: m.genres || [],
            synopsis: m.synopsis || "",
          }));
        console.log("Using fallback featured movies:", fallback);
        setFeaturedMovies(fallback);
      }
    };

    fetchFeaturedMovies();
  }, [movies]); // Include movies for fallback

  // Lắng nghe sự kiện toast toàn cục
  useEffect(() => {
    const handler = (e) => {
      const detail = e?.detail || {};
      setToast({
        open: true,
        message: detail.message || "Thao tác thành công",
        severity: detail.severity || "success",
      });
    };
    window.addEventListener("app:toast", handler);
    return () => window.removeEventListener("app:toast", handler);
  }, []);

  return (
    <Box sx={{ background: "#000" }}>
      {/* Hero Section with Background */}
      <Box ref={heroRef} sx={{ position: "relative", minHeight: "100vh" }}>
        <Background backgroundImage={toHero(activeMovie?.thumb)} />
        <Header />
        <Hero
          movie={activeMovie}
          onPlay={handlePlay}
          onInfo={handleInfo}
          onFavorite={handleFavorite}
        />

        {displayMovies.length > 0 && (
          <Box
            sx={{
              position: "absolute",
              left: 0,
              right: 0,
              bottom: 24,
              zIndex: 5,
              opacity: showThumb ? 1 : 0,
              pointerEvents: showThumb ? "auto" : "none",
              transform: showThumb ? "translateY(0)" : "translateY(8px)",
              transition: "opacity .25s ease, transform .25s ease",
            }}
          >
            <MovieThumbnails
              movies={displayMovies}
              activeIndex={activeIndex}
              onMovieSelect={handleMovieSelect}
            />
          </Box>
        )}
      </Box>

      {/* Featured Movies Section */}
      {featuredMovies.length > 0 && (
        <FeaturedMovies
          movies={featuredMovies}
          title="PHIM NỔI BẬT"
          subtitle="Khám phá những bộ phim hay nhất được yêu thích"
        />
      )}

      {/* Catalog sections similar to screenshot */}
      <CatalogSection
        title="Phim Hàn Quốc mới"
        movies={movies.map((m, i) => ({
          id: m.id || i + 1,
          title: m.title || `Movie ${i + 1}`,
          thumb: m.thumb,
          subtitle: m.subtitle || m.enTitle || m.engTitle,
          episode: m.ep || m.episode,
          pd: m.pd,
          tm: m.tm,
        }))}
      />

      <CatalogSection
        title="Phim Trung Quốc mới"
        movies={movies.map((m, i) => ({
          id: m.id || i + 1,
          title: m.title || `Movie ${i + 1}`,
          thumb: m.thumb,
          subtitle: m.subtitle || m.enTitle || m.engTitle,
          episode: m.ep || m.episode,
          pd: m.pd,
          tm: m.tm,
        }))}
      />

      <CatalogSection
        title="Phim US-UK mới"
        movies={movies.map((m, i) => ({
          id: m.id || i + 1,
          title: m.title || `Movie ${i + 1}`,
          thumb: m.thumb,
          subtitle: m.subtitle || m.enTitle || m.engTitle,
          episode: m.ep || m.episode,
          pd: m.pd,
          tm: m.tm,
        }))}
      />
      {/* Top Comments Section */}
      <TopComments />

      <CallToAction onSignUp={handleSignUp} />

      {/* Footer */}
      <Footer />

      {/* Toast */}
      <Snackbar
        open={toast.open}
        autoHideDuration={3000}
        onClose={() => setToast((t) => ({ ...t, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
      >
        <Alert
          onClose={() => setToast((t) => ({ ...t, open: false }))}
          severity={toast.severity}
          variant="filled"
          sx={{ width: "100%" }}
        >
          {toast.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
