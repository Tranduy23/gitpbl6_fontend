import React, { useState, useEffect, useCallback } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Box,
  Grid,
  Card,
  CardMedia,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Chip,
  IconButton,
  InputAdornment,
  Pagination,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material";
import {
  Search as SearchIcon,
  Close as CloseIcon,
  Star as StarIcon,
  CalendarToday as CalendarIcon,
  Language as LanguageIcon,
  Movie as MovieIcon,
  Download as DownloadIcon,
} from "@mui/icons-material";
import tmdbAPI from "../api/tmdb";

const TmdbMovieSearch = ({ open, onClose, onSelectMovie }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchType, setSearchType] = useState("search"); // search, popular, top_rated, now_playing, upcoming
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [downloadingImages, setDownloadingImages] = useState(false);

  // Debounced search function
  const debouncedSearch = useCallback(
    (() => {
      let timeoutId;
      return (query, page = 1) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(async () => {
          if (query.trim().length > 0) {
            await performSearch(query, page);
          }
        }, 500);
      };
    })(),
    [searchType]
  );

  // Perform search based on type
  const performSearch = async (query, page = 1) => {
    setLoading(true);
    setError("");

    try {
      let response;

      switch (searchType) {
        case "popular":
          response = await tmdbAPI.getPopularMovies(page);
          break;
        case "top_rated":
          response = await tmdbAPI.getTopRatedMovies(page);
          break;
        case "now_playing":
          response = await tmdbAPI.getNowPlayingMovies(page);
          break;
        case "upcoming":
          response = await tmdbAPI.getUpcomingMovies(page);
          break;
        default:
          response = await tmdbAPI.searchMovies(query, page);
      }

      setSearchResults(response.results || []);
      setTotalPages(response.total_pages || 1);
      setCurrentPage(page);
    } catch (err) {
      setError(err.message || "Failed to search movies");
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  // Handle search input change
  const handleSearchChange = (event) => {
    const query = event.target.value;
    setSearchQuery(query);

    if (searchType === "search") {
      debouncedSearch(query);
    }
  };

  // Handle search type change
  const handleSearchTypeChange = (event) => {
    const newType = event.target.value;
    setSearchType(newType);
    setSearchQuery("");
    setSearchResults([]);

    if (newType !== "search") {
      performSearch("", 1);
    }
  };

  // Handle page change
  const handlePageChange = (event, page) => {
    if (searchType === "search") {
      performSearch(searchQuery, page);
    } else {
      performSearch("", page);
    }
  };

  // Handle movie selection
  const handleSelectMovie = async (movie) => {
    setSelectedMovie(movie);
    setDownloadingImages(true);

    try {
      // Get detailed movie information
      const movieDetails = await tmdbAPI.getMovieDetails(movie.id);

      // Convert to app format
      const appMovieData = tmdbAPI.convertTmdbMovieToAppFormat(movieDetails);

      // Download images
      const images = await tmdbAPI.downloadMovieImages(movieDetails);

      // Merge image files with movie data
      const finalMovieData = {
        ...appMovieData,
        ...images,
      };

      onSelectMovie(finalMovieData);
      onClose();
    } catch (err) {
      setError(err.message || "Failed to get movie details");
    } finally {
      setDownloadingImages(false);
    }
  };

  // Load initial data when dialog opens
  useEffect(() => {
    if (open && searchType !== "search") {
      performSearch("", 1);
    }
  }, [open, searchType]);

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setSearchQuery("");
      setSearchResults([]);
      setError("");
      setCurrentPage(1);
      setTotalPages(1);
      setSelectedMovie(null);
    }
  }, [open]);

  const formatDate = (dateString) => {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleDateString();
  };

  const formatRating = (rating) => {
    if (!rating) return "N/A";
    return `${rating.toFixed(1)}/10`;
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { height: "90vh" },
      }}
    >
      <DialogTitle>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Typography
            variant="h6"
            sx={{ display: "flex", alignItems: "center", gap: 1 }}
          >
            <MovieIcon color="primary" />
            Import Movie from TMDB
          </Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent>
        <Box sx={{ mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Search Type</InputLabel>
                <Select
                  value={searchType}
                  onChange={handleSearchTypeChange}
                  label="Search Type"
                >
                  <MenuItem value="search">Search by Title</MenuItem>
                  <MenuItem value="popular">Popular Movies</MenuItem>
                  <MenuItem value="top_rated">Top Rated</MenuItem>
                  <MenuItem value="now_playing">Now Playing</MenuItem>
                  <MenuItem value="upcoming">Upcoming</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            {searchType === "search" && (
              <Grid item xs={12} md={8}>
                <TextField
                  fullWidth
                  placeholder="Search for movies..."
                  value={searchQuery}
                  onChange={handleSearchChange}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <SearchIcon />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>
            )}
          </Grid>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loading && (
          <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
            <CircularProgress />
          </Box>
        )}

        {!loading && searchResults.length > 0 && (
          <>
            <Grid container spacing={2}>
              {searchResults.map((movie) => (
                <Grid item xs={12} sm={6} md={4} key={movie.id}>
                  <Card
                    sx={{
                      height: "100%",
                      cursor: "pointer",
                      transition: "transform 0.2s",
                      "&:hover": {
                        transform: "translateY(-4px)",
                        boxShadow: 3,
                      },
                    }}
                    onClick={() => handleSelectMovie(movie)}
                  >
                    <CardMedia
                      component="img"
                      height="300"
                      image={tmdbAPI.buildPosterUrl(movie.poster_path)}
                      alt={movie.title}
                      sx={{ objectFit: "cover" }}
                    />
                    <CardContent>
                      <Typography variant="h6" component="h3" noWrap>
                        {movie.title}
                      </Typography>

                      <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ mb: 1 }}
                      >
                        {movie.original_title !== movie.title && (
                          <Box component="span" sx={{ fontStyle: "italic" }}>
                            {movie.original_title}
                          </Box>
                        )}
                      </Typography>

                      <Box
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          gap: 1,
                          mb: 1,
                        }}
                      >
                        <StarIcon
                          sx={{ fontSize: 16, color: "warning.main" }}
                        />
                        <Typography variant="body2">
                          {formatRating(movie.vote_average)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          ({movie.vote_count} votes)
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          gap: 1,
                          mb: 1,
                        }}
                      >
                        <CalendarIcon
                          sx={{ fontSize: 16, color: "text.secondary" }}
                        />
                        <Typography variant="body2" color="text.secondary">
                          {formatDate(movie.release_date)}
                        </Typography>
                      </Box>

                      <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{
                          display: "-webkit-box",
                          WebkitLineClamp: 3,
                          WebkitBoxOrient: "vertical",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        }}
                      >
                        {movie.overview}
                      </Typography>

                      {movie.genre_ids && movie.genre_ids.length > 0 && (
                        <Box
                          sx={{
                            mt: 1,
                            display: "flex",
                            flexWrap: "wrap",
                            gap: 0.5,
                          }}
                        >
                          {movie.genre_ids.slice(0, 3).map((genreId) => (
                            <Chip
                              key={genreId}
                              label={`Genre ${genreId}`}
                              size="small"
                              variant="outlined"
                            />
                          ))}
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>

            {totalPages > 1 && (
              <Box sx={{ display: "flex", justifyContent: "center", mt: 3 }}>
                <Pagination
                  count={totalPages}
                  page={currentPage}
                  onChange={handlePageChange}
                  color="primary"
                />
              </Box>
            )}
          </>
        )}

        {!loading && searchResults.length === 0 && searchQuery && (
          <Box sx={{ textAlign: "center", p: 3 }}>
            <Typography variant="body1" color="text.secondary">
              No movies found for "{searchQuery}"
            </Typography>
          </Box>
        )}

        {downloadingImages && (
          <Box
            sx={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: "rgba(255, 255, 255, 0.8)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 1000,
            }}
          >
            <Box sx={{ textAlign: "center" }}>
              <CircularProgress />
              <Typography variant="body1" sx={{ mt: 2 }}>
                Downloading movie images...
              </Typography>
            </Box>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
      </DialogActions>
    </Dialog>
  );
};

export default TmdbMovieSearch;
