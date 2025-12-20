import { useState, useEffect, useMemo } from "react";
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardMedia,
  CardContent,
  Chip,
  Stack,
  IconButton,
  Tooltip,
  Collapse,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Pagination,
} from "@mui/material";
import {
  PlayArrow as PlayIcon,
  FilterList as FilterIcon,
  KeyboardArrowDown as ArrowDownIcon,
} from "@mui/icons-material";
import { useNavigate, useLocation } from "react-router-dom";
import Header from "../components/Header";
import {
  searchMovies,
  searchByActor,
  searchByDirector,
  searchByGenre,
  getNowShowingMovies,
} from "../api/streaming";

const Movies = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // State
  const [allMovies, setAllMovies] = useState([]); // All movies from initial load
  const [filteredMovies, setFilteredMovies] = useState([]); // Filtered results
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showFilter, setShowFilter] = useState(false);
  const [isSearching, setIsSearching] = useState(false);

  // Filter state
  const [filters, setFilters] = useState({
    genre: "",
    year: "",
    search: "",
    country: "",
    actor: "",
    director: "",
  });

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 24;

  // Metadata for dropdowns
  const [dbGenres, setDbGenres] = useState([]);
  const [dbYears, setDbYears] = useState([]);
  const [dbCountries, setDbCountries] = useState([]);

  // =============================================================================
  // STEP 1: Load metadata (genres, years, countries) from API
  // =============================================================================
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
        const years = new Set();
        const countries = new Set();

        list.forEach((m) => {
          (m?.categories || m?.genres || []).forEach((c) => {
            if (c) categories.add(String(c).trim());
          });
          if (m?.year) years.add(String(m.year));
          if (m?.country || m?.countryName) {
            countries.add(String(m.country || m.countryName).trim());
          }
        });

        if (!aborted) {
          setDbGenres(
            Array.from(categories)
              .filter(Boolean)
              .sort((a, b) => String(a).localeCompare(String(b), "vi"))
          );
          setDbYears(
            Array.from(years)
              .filter(Boolean)
              .sort((a, b) => b.localeCompare(a))
          );
          setDbCountries(
            Array.from(countries)
              .filter(Boolean)
              .sort((a, b) => String(a).localeCompare(String(b), "vi"))
          );
        }
      } catch (err) {
        console.error("Error loading metadata:", err);
      }
    })();
    return () => {
      aborted = true;
    };
  }, []);

  // =============================================================================
  // STEP 2: Sync filters with URL on mount and URL change
  // =============================================================================
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const newFilters = {
      search: params.get("query") || "",
      genre: params.get("genre") || "",
      country: params.get("country") || "",
      year: params.get("year") || "",
      actor: params.get("actor") || "",
      director: params.get("director") || "",
    };
    setFilters(newFilters);
  }, [location.search]);

  // =============================================================================
  // STEP 3: Load initial movies from API (only once on mount)
  // =============================================================================
  useEffect(() => {
    const loadMovies = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await getNowShowingMovies(100);
        const moviesData = Array.isArray(response)
          ? response
          : response?.content || response?.movies || response?.data || [];

        const transformedMovies = moviesData.map((movie) => ({
          id: movie.id,
          title: movie.title,
          englishTitle: movie.title,
          ageRating: movie.ageRating || null,
          imdbRating: movie.imdbRating ?? null,
          averageRating: movie.averageRating ?? null,
          totalRatings: movie.totalRatings ?? 0,
          year: movie.year != null ? String(movie.year) : undefined,
          duration: movie.videoDuration || undefined,
          genres: (movie.categories || movie.genres || [])
            .map((c) => String(c).trim())
            .filter(Boolean),
          synopsis: movie.synopsis || "",
          thumb: movie.posterUrl || movie.thumbnailUrl || undefined,
          posterUrl: movie.posterUrl || undefined,
          videoUrl: movie.videoUrl || undefined,
          trailerUrl: movie.trailerUrl || undefined,
          streamingUrl: movie.streamingUrl || undefined,
          isAvailable: Boolean(movie.isAvailable),
          actors: movie.actors || [],
          directors: movie.directors || [],
          country: movie.country || movie.countryName || undefined,
          language: movie.language || undefined,
          viewCount: movie.viewCount ?? 0,
          likeCount: movie.likeCount ?? 0,
          dislikeCount: movie.dislikeCount ?? 0,
          isFeatured: Boolean(movie.isFeatured),
          isTrending: Boolean(movie.isTrending),
          releaseDate: movie.releaseDate || undefined,
          downloadEnabled: Boolean(movie.downloadEnabled),
          availableQualities: movie.availableQualities || [],
        }));

        setAllMovies(transformedMovies);
      } catch (err) {
        console.error("Error loading movies:", err);
        setError(err.message || "Failed to load movies");
      } finally {
        setLoading(false);
      }
    };

    loadMovies();
  }, []);

  // =============================================================================
  // STEP 4: Determine if we need server search
  // =============================================================================
  const needsServerSearch = useMemo(() => {
    return Boolean(
      filters.genre || filters.director || filters.actor || filters.search
    );
  }, [filters.genre, filters.director, filters.actor, filters.search]);

  // =============================================================================
  // STEP 5: Apply filters (client-side or server-side)
  // =============================================================================
  useEffect(() => {
    const applyFilters = async () => {
      // Case 1: Need server search (genre, director, actor, or search)
      if (needsServerSearch) {
        setIsSearching(true);
        try {
          let data;

          // Call appropriate server search API
          if (filters.genre) {
            data = await searchByGenre(filters.genre, 100);
          } else if (filters.director) {
            data = await searchByDirector(filters.director, 100);
          } else if (filters.actor) {
            data = await searchByActor(filters.actor, 0, 100);
          } else if (filters.search) {
            data = await searchMovies({
              query: filters.search,
              page: 0,
              size: 100,
              sortBy: "title",
              sortOrder: "asc",
            });
          }

          // Transform server results
          const source = Array.isArray(data?.data)
            ? data.data
            : Array.isArray(data?.content)
            ? data.content
            : Array.isArray(data?.movies)
            ? data.movies
            : Array.isArray(data)
            ? data
            : [];

          let results = source.map((movie) => ({
            id: movie.id,
            title: movie.title,
            englishTitle: movie.title,
            ageRating: movie.ageRating || null,
            imdbRating: movie.imdbRating ?? null,
            averageRating: movie.averageRating ?? null,
            totalRatings: movie.totalRatings ?? 0,
            year: movie.year != null ? String(movie.year) : undefined,
            duration: movie.videoDuration || undefined,
            genres: (movie.categories || movie.genres || [])
              .map((c) => String(c).trim())
              .filter(Boolean),
            synopsis: movie.synopsis || "",
            thumb: movie.posterUrl || undefined,
            posterUrl: movie.posterUrl || undefined,
            videoUrl: movie.videoUrl || undefined,
            trailerUrl: movie.trailerUrl || undefined,
            streamingUrl: movie.streamingUrl || undefined,
            isAvailable: Boolean(movie.isAvailable),
            actors: movie.actors || [],
            directors: movie.directors || [],
            country: movie.country || undefined,
            language: movie.language || undefined,
            viewCount: movie.viewCount ?? 0,
            likeCount: movie.likeCount ?? 0,
            dislikeCount: movie.dislikeCount ?? 0,
            isFeatured: Boolean(movie.isFeatured),
            isTrending: Boolean(movie.isTrending),
            releaseDate: movie.releaseDate || undefined,
            downloadEnabled: Boolean(movie.downloadEnabled),
            availableQualities: movie.availableQualities || [],
          }));

          // Apply client-side filters on server results (year, country)
          if (filters.year) {
            const yearStr = String(filters.year).trim();
            results = results.filter(
              (m) => String(m.year || "").trim() === yearStr
            );
          }
          if (filters.country) {
            const countryStr = String(filters.country).toLowerCase().trim();
            results = results.filter(
              (m) =>
                String(m.country || "")
                  .toLowerCase()
                  .trim() === countryStr
            );
          }

          setFilteredMovies(results);
        } catch (err) {
          console.error("Server search error:", err);
          setFilteredMovies([]);
        } finally {
          setIsSearching(false);
        }
      }
      // Case 2: Client-side filtering only (year, country, or no filters)
      else {
        let results = [...allMovies];

        if (filters.year) {
          const yearStr = String(filters.year).trim();
          results = results.filter(
            (m) => String(m.year || "").trim() === yearStr
          );
        }
        if (filters.country) {
          const countryStr = String(filters.country).toLowerCase().trim();
          results = results.filter(
            (m) =>
              String(m.country || "")
                .toLowerCase()
                .trim() === countryStr
          );
        }

        setFilteredMovies(results);
      }

      // Reset to page 1 when filters change
      setCurrentPage(1);
    };

    // Debounce for search/director/actor to avoid too many API calls
    const shouldDebounce = filters.search || filters.director || filters.actor;
    const timeoutId = setTimeout(applyFilters, shouldDebounce ? 500 : 0);

    return () => clearTimeout(timeoutId);
  }, [
    allMovies,
    filters.genre,
    filters.year,
    filters.country,
    filters.search,
    filters.director,
    filters.actor,
    needsServerSearch,
  ]);

  // =============================================================================
  // Handlers
  // =============================================================================
  const handleFilterChange = (filterType, value) => {
    const newFilters = {
      ...filters,
      [filterType]: value,
    };
    setFilters(newFilters);

    // Update URL
    const params = new URLSearchParams();
    if (newFilters.search) params.set("query", newFilters.search);
    if (newFilters.genre) params.set("genre", newFilters.genre);
    if (newFilters.country) params.set("country", newFilters.country);
    if (newFilters.year) params.set("year", newFilters.year);
    if (newFilters.actor) params.set("actor", newFilters.actor);
    if (newFilters.director) params.set("director", newFilters.director);

    const newUrl = params.toString()
      ? `${location.pathname}?${params.toString()}`
      : location.pathname;
    navigate(newUrl, { replace: true });
  };

  const clearFilters = () => {
    setFilters({
      genre: "",
      year: "",
      search: "",
      country: "",
      actor: "",
      director: "",
    });
    navigate(location.pathname, { replace: true });
  };

  const handleMovieClick = (movieId) => {
    navigate(`/movie/${movieId}`);
  };

  const handlePlayClick = (movieId) => {
    navigate(`/stream/${movieId}`);
  };

  // =============================================================================
  // Pagination
  // =============================================================================
  const totalPages = Math.ceil(filteredMovies.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedMovies = filteredMovies.slice(startIndex, endIndex);

  // =============================================================================
  // Dropdown options
  // =============================================================================
  const uniqueGenres = dbGenres.length > 0 ? dbGenres : [];
  const uniqueYears = dbYears.length > 0 ? dbYears : [];
  const uniqueCountries = dbCountries.length > 0 ? dbCountries : [];

  // =============================================================================
  // Render
  // =============================================================================
  if (loading) {
    return (
      <Box
        sx={{
          minHeight: "100vh",
          background: "#000",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Typography variant="h6" sx={{ color: "#fff" }}>
          Đang tải phim...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box
        sx={{
          minHeight: "100vh",
          background: "#000",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Typography variant="h6" sx={{ color: "#ff6b6b" }}>
          Lỗi tải phim: {error}
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background:
          "linear-gradient(180deg, #0b1220 0%, #0a0f1a 50%, #0b0d13 100%)",
      }}
    >
      <Header />

      <Container maxWidth="xl" sx={{ pt: 12, pb: 6 }}>
        {/* Page Header */}
        <Box sx={{ mb: 4 }}>
          <Typography
            variant="h4"
            sx={{
              color: "#fff",
              fontWeight: 700,
              mb: 2,
              textShadow: "0 2px 8px rgba(0,0,0,0.8)",
            }}
          >
            Phim lẻ
          </Typography>

          <Button
            variant="outlined"
            startIcon={<FilterIcon />}
            endIcon={<ArrowDownIcon />}
            onClick={() => setShowFilter(!showFilter)}
            sx={{
              color: "rgba(255,255,255,0.9)",
              borderColor: "rgba(255,255,255,0.3)",
              borderRadius: 2,
              px: 3,
              py: 1,
              textTransform: "none",
              fontWeight: 600,
              "&:hover": {
                borderColor: "#FFD700",
                color: "#FFD700",
                background: "rgba(255,215,0,0.1)",
              },
            }}
          >
            Bộ lọc
          </Button>
        </Box>

        {/* Filter Panel */}
        <Collapse in={showFilter}>
          <Box
            sx={{
              background: "rgba(255,255,255,0.05)",
              borderRadius: 3,
              p: 3,
              mb: 4,
              border: "1px solid rgba(255,255,255,0.1)",
              backdropFilter: "blur(10px)",
            }}
          >
            <Grid container spacing={3} alignItems="center">
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Tìm kiếm phim"
                  value={filters.search}
                  onChange={(e) => handleFilterChange("search", e.target.value)}
                  sx={{
                    "& .MuiOutlinedInput-root": {
                      color: "#fff",
                      "& fieldset": {
                        borderColor: "rgba(255,255,255,0.3)",
                      },
                      "&:hover fieldset": {
                        borderColor: "rgba(255,215,0,0.5)",
                      },
                      "&.Mui-focused fieldset": {
                        borderColor: "#FFD700",
                      },
                    },
                    "& .MuiInputLabel-root": {
                      color: "rgba(255,255,255,0.7)",
                      "&.Mui-focused": {
                        color: "#FFD700",
                      },
                    },
                  }}
                />
              </Grid>

              <Grid item xs={12} md={2}>
                <FormControl fullWidth>
                  <InputLabel sx={{ color: "rgba(255,255,255,0.7)" }}>
                    Thể loại
                  </InputLabel>
                  <Select
                    value={filters.genre}
                    onChange={(e) =>
                      handleFilterChange("genre", e.target.value)
                    }
                    label="Thể loại"
                    MenuProps={{
                      PaperProps: {
                        sx: {
                          bgcolor: "rgba(20, 20, 30, 0.95)",
                          backdropFilter: "blur(10px)",
                          border: "1px solid rgba(255,255,255,0.1)",
                          "& .MuiMenuItem-root": {
                            color: "rgba(255,255,255,0.9)",
                            "&:hover": {
                              bgcolor: "rgba(255,215,0,0.2)",
                              color: "#FFD700",
                            },
                            "&.Mui-selected": {
                              bgcolor: "rgba(255,215,0,0.3)",
                              color: "#FFD700",
                              "&:hover": {
                                bgcolor: "rgba(255,215,0,0.4)",
                              },
                            },
                          },
                        },
                      },
                    }}
                    sx={{
                      color: "#fff",
                      "& .MuiOutlinedInput-notchedOutline": {
                        borderColor: "rgba(255,255,255,0.3)",
                      },
                      "&:hover .MuiOutlinedInput-notchedOutline": {
                        borderColor: "rgba(255,215,0,0.5)",
                      },
                      "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                        borderColor: "#FFD700",
                      },
                      "& .MuiSvgIcon-root": {
                        color: "rgba(255,255,255,0.7)",
                      },
                    }}
                  >
                    <MenuItem value="">
                      <em>Tất cả</em>
                    </MenuItem>
                    {uniqueGenres.map((genre) => (
                      <MenuItem key={genre} value={genre}>
                        {genre}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={2}>
                <FormControl fullWidth>
                  <InputLabel sx={{ color: "rgba(255,255,255,0.7)" }}>
                    Năm
                  </InputLabel>
                  <Select
                    value={filters.year}
                    onChange={(e) => handleFilterChange("year", e.target.value)}
                    label="Năm"
                    sx={{
                      color: "#fff",
                      "& .MuiOutlinedInput-notchedOutline": {
                        borderColor: "rgba(255,255,255,0.3)",
                      },
                      "&:hover .MuiOutlinedInput-notchedOutline": {
                        borderColor: "rgba(255,215,0,0.5)",
                      },
                      "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                        borderColor: "#FFD700",
                      },
                      "& .MuiSvgIcon-root": {
                        color: "rgba(255,255,255,0.7)",
                      },
                    }}
                  >
                    <MenuItem value="">
                      <em>Tất cả</em>
                    </MenuItem>
                    {uniqueYears.map((year) => (
                      <MenuItem key={year} value={year}>
                        {year}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={2}>
                <FormControl fullWidth>
                  <InputLabel sx={{ color: "rgba(255,255,255,0.7)" }}>
                    Quốc gia
                  </InputLabel>
                  <Select
                    value={filters.country}
                    onChange={(e) =>
                      handleFilterChange("country", e.target.value)
                    }
                    label="Quốc gia"
                    sx={{
                      color: "#fff",
                      "& .MuiOutlinedInput-notchedOutline": {
                        borderColor: "rgba(255,255,255,0.3)",
                      },
                      "&:hover .MuiOutlinedInput-notchedOutline": {
                        borderColor: "rgba(255,215,0,0.5)",
                      },
                      "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                        borderColor: "#FFD700",
                      },
                      "& .MuiSvgIcon-root": {
                        color: "rgba(255,255,255,0.7)",
                      },
                    }}
                  >
                    <MenuItem value="">
                      <em>Tất cả</em>
                    </MenuItem>
                    {uniqueCountries.map((country) => (
                      <MenuItem key={country} value={country}>
                        {country}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={2}>
                <Button
                  fullWidth
                  variant="outlined"
                  onClick={clearFilters}
                  sx={{
                    color: "rgba(255,255,255,0.7)",
                    borderColor: "rgba(255,255,255,0.3)",
                    height: "56px",
                    "&:hover": {
                      borderColor: "#ff6b6b",
                      color: "#ff6b6b",
                    },
                  }}
                >
                  Xóa bộ lọc
                </Button>
              </Grid>
            </Grid>
          </Box>
        </Collapse>

        {/* Results count */}
        <Box sx={{ mb: 3 }}>
          <Typography
            variant="body1"
            sx={{
              color: "rgba(255,255,255,0.7)",
              fontSize: "0.9rem",
            }}
          >
            {isSearching ? (
              "Đang tìm kiếm..."
            ) : (
              <>
                Hiển thị {paginatedMovies.length} / {filteredMovies.length} phim
                {Object.values(filters).some((v) => v) && " (đã lọc)"}
                {totalPages > 1 && ` - Trang ${currentPage}/${totalPages}`}
              </>
            )}
          </Typography>
        </Box>

        {/* Movies Grid */}
        {isSearching ? (
          <Box sx={{ textAlign: "center", py: 8 }}>
            <Typography variant="h6" sx={{ color: "#fff" }}>
              Đang tìm kiếm...
            </Typography>
          </Box>
        ) : paginatedMovies.length === 0 ? (
          <Box sx={{ textAlign: "center", py: 8 }}>
            <Typography variant="h6" sx={{ color: "rgba(255,255,255,0.7)" }}>
              Không tìm thấy phim nào
            </Typography>
          </Box>
        ) : (
          <Grid container spacing={3}>
            {paginatedMovies.map((movie) => (
              <Grid item xs={6} sm={4} md={3} lg={2.4} xl={2} key={movie.id}>
                <Card
                  sx={{
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 3,
                    overflow: "hidden",
                    cursor: "pointer",
                    transition: "all 0.3s ease",
                    backdropFilter: "blur(10px)",
                    "&:hover": {
                      transform: "translateY(-8px)",
                      boxShadow: "0 20px 40px rgba(0,0,0,0.6)",
                      borderColor: "rgba(255,215,0,0.5)",
                    },
                  }}
                  onClick={() => handleMovieClick(movie.id)}
                >
                  <Box sx={{ position: "relative" }}>
                    <CardMedia
                      component="img"
                      height="280"
                      image={movie.thumb}
                      alt={movie.title}
                      sx={{
                        objectFit: "cover",
                        transition: "transform 0.3s ease",
                        "&:hover": {
                          transform: "scale(1.05)",
                        },
                      }}
                    />

                    {/* Overlay with play button */}
                    <Box
                      sx={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: "rgba(0,0,0,0.4)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        opacity: 0,
                        transition: "opacity 0.3s ease",
                        "&:hover": {
                          opacity: 1,
                        },
                      }}
                    >
                      <Tooltip title="Xem phim">
                        <IconButton
                          onClick={(e) => {
                            e.stopPropagation();
                            handlePlayClick(movie.id);
                          }}
                          sx={{
                            background: "rgba(255,215,0,0.9)",
                            color: "#000",
                            width: 60,
                            height: 60,
                            "&:hover": {
                              background: "#FFD700",
                              transform: "scale(1.1)",
                            },
                          }}
                        >
                          <PlayIcon sx={{ fontSize: 30 }} />
                        </IconButton>
                      </Tooltip>
                    </Box>

                    {/* Premium Badge */}
                    <Chip
                      label="P.Đề"
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
                  </Box>

                  <CardContent sx={{ p: 2, "&:last-child": { pb: 2 } }}>
                    <Typography
                      variant="h6"
                      sx={{
                        color: "#fff",
                        fontWeight: 600,
                        fontSize: "0.95rem",
                        lineHeight: 1.3,
                        mb: 0.5,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        display: "-webkit-box",
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: "vertical",
                      }}
                    >
                      {movie.title}
                    </Typography>

                    <Typography
                      variant="body2"
                      sx={{
                        color: "rgba(255,255,255,0.7)",
                        fontSize: "0.8rem",
                        fontStyle: "italic",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {movie.englishTitle || movie.title}
                    </Typography>

                    <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                      {movie.year && (
                        <Chip
                          label={movie.year}
                          size="small"
                          sx={{
                            background: "rgba(255,255,255,0.1)",
                            color: "rgba(255,255,255,0.8)",
                            fontSize: "0.7rem",
                            height: 20,
                          }}
                        />
                      )}
                      {movie.imdbRating && (
                        <Chip
                          label={movie.imdbRating}
                          size="small"
                          sx={{
                            background: "rgba(255,215,0,0.2)",
                            color: "#FFD700",
                            fontSize: "0.7rem",
                            height: 20,
                          }}
                        />
                      )}
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}

        {/* Pagination */}
        {totalPages > 1 && !isSearching && (
          <Box sx={{ display: "flex", justifyContent: "center", mt: 6, mb: 2 }}>
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
      </Container>
    </Box>
  );
};

export default Movies;
