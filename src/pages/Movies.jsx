import { useState, useEffect } from "react";
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
  getNowShowingMovies,
} from "../api/streaming";

const Movies = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showFilter, setShowFilter] = useState(false);
  const [filteredMovies, setFilteredMovies] = useState([]);
  const [filters, setFilters] = useState({
    genre: "",
    year: "",
    search: "",
    country: "",
    actor: "",
    director: "",
  });
  const [page, setPage] = useState(0);
  const [size] = useState(24);
  const [isSearching, setIsSearching] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [searchMode, setSearchMode] = useState(null); // 'movie' | 'actor' | 'director'

  // Load movies from now-showing API
  useEffect(() => {
    const loadNowShowingMovies = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await getNowShowingMovies(100); // Load more movies
        const moviesData = Array.isArray(response) 
          ? response 
          : response?.content || response?.movies || response?.data || [];
        
        // Transform API response to match expected format
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
          genres: movie.categories || movie.genres || [],
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
        
        setMovies(transformedMovies);
        setFilteredMovies(transformedMovies);
      } catch (err) {
        console.error("Error loading now-showing movies:", err);
        setError(err.message || "Failed to load movies");
        setMovies([]);
        setFilteredMovies([]);
      } finally {
        setLoading(false);
      }
    };

    loadNowShowingMovies();
  }, []);

  // Initialize from URL query
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const q = params.get("query") || "";
    const g = params.get("genre") || "";
    const c = params.get("country") || "";
    const a = params.get("actor") || "";
    const d = params.get("director") || "";
    const next = { ...filters };
    if (q) next.search = q;
    if (g) next.genre = g;
    if (c) next.country = c;
    if (a) next.actor = a;
    if (d) next.director = d;
    setFilters(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.search]);

  // Local fallback filtering when no server search (empty filters)
  useEffect(() => {
    setFilteredMovies(movies);
  }, [movies]);

  useEffect(() => {
    let current = movies;

    if (filters.search) {
      current = current.filter(
        (movie) =>
          movie.title.toLowerCase().includes(filters.search.toLowerCase()) ||
          (movie.englishTitle &&
            movie.englishTitle
              .toLowerCase()
              .includes(filters.search.toLowerCase()))
      );
    }

    if (filters.genre) {
      current = current.filter((movie) =>
        movie.genres.some((genre) =>
          genre.toLowerCase().includes(filters.genre.toLowerCase())
        )
      );
    }

    if (filters.year) {
      current = current.filter((movie) => movie.year === filters.year);
    }

    if (filters.country) {
      current = current.filter(
        (movie) =>
          movie.country &&
          String(movie.country).toLowerCase() === filters.country.toLowerCase()
      );
    }

    if (filters.actor) {
      const needle = filters.actor.toLowerCase();
      current = current.filter(
        (movie) =>
          Array.isArray(movie.actors) &&
          movie.actors.some((n) => String(n).toLowerCase().includes(needle))
      );
    }

    if (filters.director) {
      const needle = filters.director.toLowerCase();
      current = current.filter(
        (movie) =>
          Array.isArray(movie.directors) &&
          movie.directors.some((n) => String(n).toLowerCase().includes(needle))
      );
    }

    setFilteredMovies(current);
  }, [movies, filters]);

  // Disable server-side search - only show now-showing movies
  const shouldServerSearch = false;

  const transformSearchResults = (data) => {
    const source = Array.isArray(data?.content)
      ? data.content
      : Array.isArray(data?.movies)
      ? data.movies
      : Array.isArray(data)
      ? data
      : [];
    const items = source;
    return items.map((movie) => ({
      id: movie.id,
      title: movie.title,
      englishTitle: movie.title,
      ageRating: movie.ageRating || null,
      imdbRating: movie.imdbRating ?? null,
      averageRating: movie.averageRating ?? null,
      totalRatings: movie.totalRatings ?? 0,
      year: movie.year != null ? String(movie.year) : undefined,
      duration: movie.videoDuration || undefined,
      genres: movie.categories || movie.genres || [],
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
  };

  const fetchAdvanced = async (nextPage = 0, append = false) => {
    setIsSearching(true);
    try {
      let data;
      let mode = searchMode;

      // Determine mode for this request
      if (filters.actor) mode = "actor";
      else if (filters.director) mode = "director";

      if (!mode) {
        // Try title search first
        const body = {
          query: filters.search || undefined,
          page: nextPage,
          size,
          sortBy: "title",
          sortOrder: "asc",
        };
        const movieRes = await searchMovies(body);
        const movieItems = transformSearchResults(movieRes);
        if (movieItems.length > 0 || !filters.search) {
          data = movieRes;
          mode = "movie";
        } else {
          // Fallback to actor, then director
          const actorRes = await searchByActor(
            filters.search,
            nextPage,
            size
          ).catch(() => null);
          const actorItems = transformSearchResults(actorRes || {});
          if (actorItems.length > 0) {
            data = actorRes;
            mode = "actor";
            if (nextPage === 0) {
              setFilters((prev) => ({
                ...prev,
                actor: prev.search,
                director: "",
                search: "",
              }));
            }
          } else {
            const dirRes = await searchByDirector(
              filters.search,
              nextPage,
              size
            ).catch(() => null);
            data = dirRes || { content: [] };
            mode = "director";
            if (nextPage === 0) {
              setFilters((prev) => ({
                ...prev,
                director: prev.search,
                actor: "",
                search: "",
              }));
            }
          }
        }
      } else if (mode === "actor") {
        data = await searchByActor(
          filters.actor || filters.search,
          nextPage,
          size
        );
      } else if (mode === "director") {
        data = await searchByDirector(
          filters.director || filters.search,
          nextPage,
          size
        );
      } else {
        const body = {
          query: filters.search || undefined,
          page: nextPage,
          size,
          sortBy: "title",
          sortOrder: "asc",
        };
        data = await searchMovies(body);
      }
      const items = transformSearchResults(data);
      const totalPages =
        data?.totalPages ?? (items.length < size ? nextPage + 1 : nextPage + 2);
      setHasMore(nextPage + 1 < totalPages);
      setFilteredMovies((prev) => (append ? [...prev, ...items] : items));
      setPage(nextPage);
      setSearchMode(mode);
    } catch (e) {
      // fallback: keep local filtered
      setHasMore(false);
    } finally {
      setIsSearching(false);
    }
  };

  // Trigger server search when keyword changes
  useEffect(() => {
    if (shouldServerSearch) {
      fetchAdvanced(0, false);
    } else {
      setPage(0);
      setHasMore(true);
      setSearchMode(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters.search, filters.actor, filters.director]);

  const handleMovieClick = (movieId) => {
    navigate(`/movie/${movieId}`);
  };

  const handlePlayClick = (movieId) => {
    navigate(`/stream/${movieId}`);
  };

  const handleFilterChange = (filterType, value) => {
    setFilters((prev) => ({
      ...prev,
      [filterType]: value,
    }));
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
    setPage(0);
    setHasMore(true);
    setSearchMode(null);
  };

  // Get unique genres, years, countries for filter options
  const uniqueGenres = [...new Set(movies.flatMap((movie) => movie.genres))];
  const uniqueYears = [...new Set(movies.map((movie) => movie.year))].sort(
    (a, b) => b - a
  );
  const uniqueCountries = [
    ...new Set(movies.map((movie) => movie.country).filter(Boolean)),
  ].sort((a, b) => String(a).localeCompare(String(b), "vi"));

  if (loading && !shouldServerSearch) {
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

  if (error && !shouldServerSearch) {
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
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      const value = String(e.currentTarget.value || "").trim();
                      if (value) {
                        setFilters((prev) => ({
                          ...prev,
                          actor: value,
                          director: "",
                          search: "",
                        }));
                      }
                    }
                  }}
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

              <Grid item xs={12} md={3}>
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
                      <em>Tất cả thể loại</em>
                    </MenuItem>
                    {uniqueGenres.map((genre) => (
                      <MenuItem key={genre} value={genre}>
                        {genre}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={3}>
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
                      <em>Tất cả năm</em>
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
                      <em>Tất cả quốc gia</em>
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
                <TextField
                  fullWidth
                  label="Diễn viên"
                  value={filters.actor}
                  onChange={(e) => handleFilterChange("actor", e.target.value)}
                  sx={{
                    "& .MuiOutlinedInput-root": {
                      color: "#fff",
                      "& fieldset": { borderColor: "rgba(255,255,255,0.3)" },
                      "&:hover fieldset": {
                        borderColor: "rgba(255,215,0,0.5)",
                      },
                      "&.Mui-focused fieldset": { borderColor: "#FFD700" },
                    },
                    "& .MuiInputLabel-root": {
                      color: "rgba(255,255,255,0.7)",
                      "&.Mui-focused": { color: "#FFD700" },
                    },
                  }}
                />
              </Grid>

              <Grid item xs={12} md={2}>
                <TextField
                  fullWidth
                  label="Đạo diễn"
                  value={filters.director}
                  onChange={(e) =>
                    handleFilterChange("director", e.target.value)
                  }
                  sx={{
                    "& .MuiOutlinedInput-root": {
                      color: "#fff",
                      "& fieldset": { borderColor: "rgba(255,255,255,0.3)" },
                      "&:hover fieldset": {
                        borderColor: "rgba(255,215,0,0.5)",
                      },
                      "&.Mui-focused fieldset": { borderColor: "#FFD700" },
                    },
                    "& .MuiInputLabel-root": {
                      color: "rgba(255,255,255,0.7)",
                      "&.Mui-focused": { color: "#FFD700" },
                    },
                  }}
                />
              </Grid>

              <Grid item xs={12} md={2}>
                <Button
                  variant="outlined"
                  onClick={clearFilters}
                  sx={{
                    color: "rgba(255,255,255,0.7)",
                    borderColor: "rgba(255,255,255,0.3)",
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
            Hiển thị {filteredMovies.length} phim
            {(filters.search ||
              filters.genre ||
              filters.year ||
              filters.country) &&
              " (đã lọc)"}
          </Typography>
        </Box>

        {/* Movies Grid */}
        <Grid container spacing={3}>
          {filteredMovies.map((movie) => (
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

                  {/* Premium/Subtitle Badge */}
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

                  {movie.englishTitle && (
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
                      {movie.englishTitle}
                    </Typography>
                  )}

                  <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
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
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Load More Button */}
        {shouldServerSearch && hasMore && (
          <Box sx={{ textAlign: "center", mt: 6 }}>
            <Button
              variant="outlined"
              onClick={() => fetchAdvanced(page + 1, true)}
              disabled={isSearching}
              sx={{
                color: "rgba(255,255,255,0.9)",
                borderColor: "rgba(255,255,255,0.3)",
                borderRadius: 3,
                px: 4,
                py: 1.5,
                textTransform: "none",
                fontWeight: 600,
                fontSize: "1rem",
                "&:hover": {
                  borderColor: "#FFD700",
                  color: "#FFD700",
                  background: "rgba(255,215,0,0.1)",
                },
              }}
            >
              {isSearching ? "Đang tải..." : "Xem thêm phim"}
            </Button>
          </Box>
        )}
      </Container>
    </Box>
  );
};

export default Movies;
