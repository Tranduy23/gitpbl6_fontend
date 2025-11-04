// TMDB API service for movie data integration
// The Movie Database (TMDB) API client

const TMDB_API_BASE = "https://api.themoviedb.org/3";
const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p";

// Get TMDB API key from environment variables
const TMDB_API_KEY =
  import.meta?.env?.VITE_TMDB_API_KEY || "42bfbdaddb3be91c42877e3fcd619ef2";

if (!TMDB_API_KEY) {
  console.warn(
    "TMDB API key not found. Please set VITE_TMDB_API_KEY in your environment variables."
  );
}

/**
 * Make a request to TMDB API
 */
async function tmdbRequest(endpoint, params = {}) {
  if (!TMDB_API_KEY) {
    throw new Error(
      "TMDB API key is required. Please set VITE_TMDB_API_KEY in your environment variables."
    );
  }

  const url = new URL(`${TMDB_API_BASE}${endpoint}`);
  url.searchParams.append("api_key", TMDB_API_KEY);
  url.searchParams.append("language", "en-US");

  // Add additional parameters
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.append(key, value);
    }
  });

  try {
    const response = await fetch(url.toString());

    if (!response.ok) {
      throw new Error(
        `TMDB API error: ${response.status} ${response.statusText}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error("TMDB API request failed:", error);
    throw error;
  }
}

/**
 * Search for movies by title
 */
export async function searchMovies(query, page = 1) {
  if (!query || query.trim().length === 0) {
    throw new Error("Search query is required");
  }

  return tmdbRequest("/search/movie", {
    query: query.trim(),
    page,
    include_adult: false,
  });
}

/**
 * Get movie details by TMDB ID
 */
export async function getMovieDetails(tmdbId) {
  if (!tmdbId) {
    throw new Error("TMDB ID is required");
  }

  return tmdbRequest(`/movie/${tmdbId}`, {
    append_to_response: "credits,videos,images",
  });
}

/**
 * Get popular movies
 */
export async function getPopularMovies(page = 1) {
  return tmdbRequest("/movie/popular", { page });
}

/**
 * Get top rated movies
 */
export async function getTopRatedMovies(page = 1) {
  return tmdbRequest("/movie/top_rated", { page });
}

/**
 * Get now playing movies
 */
export async function getNowPlayingMovies(page = 1) {
  return tmdbRequest("/movie/now_playing", { page });
}

/**
 * Get upcoming movies
 */
export async function getUpcomingMovies(page = 1) {
  return tmdbRequest("/movie/upcoming", { page });
}

/**
 * Get movie genres
 */
export async function getMovieGenres() {
  return tmdbRequest("/genre/movie/list");
}

/**
 * Get movie configuration (image sizes, etc.)
 */
export async function getMovieConfiguration() {
  return tmdbRequest("/configuration");
}

/**
 * Build image URL for TMDB images
 */
export function buildImageUrl(path, size = "w500") {
  if (!path) return null;

  // Ensure path starts with /
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  return `${TMDB_IMAGE_BASE}/${size}${normalizedPath}`;
}

/**
 * Build backdrop image URL
 */
export function buildBackdropUrl(path, size = "w1280") {
  if (!path) return null;
  return `${TMDB_IMAGE_BASE}/${size}${path}`;
}

/**
 * Build poster image URL
 */
export function buildPosterUrl(path, size = "w500") {
  return buildImageUrl(path, size);
}

/**
 * Build thumbnail image URL
 */
export function buildThumbnailUrl(path, size = "w300") {
  return buildImageUrl(path, size);
}

/**
 * Convert TMDB movie data to our application format
 */
export function convertTmdbMovieToAppFormat(tmdbMovie) {
  const credits = tmdbMovie.credits || {};
  const videos = tmdbMovie.videos || {};
  const images = tmdbMovie.images || {};

  // Extract cast (actors)
  const actors = credits.cast
    ? credits.cast
        .slice(0, 10)
        .map((actor) => actor.name)
        .join(", ")
    : "";

  // Extract crew (directors)
  const directors = credits.crew
    ? credits.crew
        .filter((person) => person.job === "Director")
        .map((director) => director.name)
        .join(", ")
    : "";

  // Extract genres
  const categories = tmdbMovie.genres
    ? tmdbMovie.genres.map((genre) => genre.name).join(", ")
    : "";

  // Find trailer video
  const trailer = videos.results
    ? videos.results.find(
        (video) => video.type === "Trailer" && video.site === "YouTube"
      )
    : null;

  // Get poster and backdrop images
  const posterPath = tmdbMovie.poster_path;
  const backdropPath = tmdbMovie.backdrop_path;

  return {
    // Basic info
    title: tmdbMovie.title || "",
    synopsis: tmdbMovie.overview || "",
    year: tmdbMovie.release_date
      ? new Date(tmdbMovie.release_date).getFullYear()
      : new Date().getFullYear(),
    releaseDate: tmdbMovie.release_date || "",

    // People
    actors: actors,
    directors: directors,

    // Categories and metadata
    categories: categories,
    country:
      tmdbMovie.production_countries &&
      tmdbMovie.production_countries.length > 0
        ? tmdbMovie.production_countries[0].name
        : "",
    language: tmdbMovie.original_language || "en",
    ageRating: tmdbMovie.adult ? "R" : "PG-13",

    // Ratings and status
    imdbRating: tmdbMovie.vote_average ? tmdbMovie.vote_average.toFixed(1) : "",
    isAvailable: true,
    isFeatured: tmdbMovie.vote_average > 7.0,
    isTrending: tmdbMovie.popularity > 100,
    downloadEnabled: true,
    maxDownloadQuality: "1080p",

    // Media URLs
    trailerUrl: trailer ? `https://www.youtube.com/watch?v=${trailer.key}` : "",

    // Image URLs
    posterUrl: buildPosterUrl(posterPath),
    backdropUrl: buildBackdropUrl(backdropPath),
    thumbnailUrl: buildThumbnailUrl(posterPath),

    // TMDB metadata
    tmdbId: tmdbMovie.id,
    tmdbData: {
      id: tmdbMovie.id,
      originalTitle: tmdbMovie.original_title,
      popularity: tmdbMovie.popularity,
      voteCount: tmdbMovie.vote_count,
      budget: tmdbMovie.budget,
      revenue: tmdbMovie.revenue,
      runtime: tmdbMovie.runtime,
      status: tmdbMovie.status,
      tagline: tmdbMovie.tagline,
    },
  };
}

/**
 * Download image from URL and convert to File object
 */
export async function downloadImageAsFile(imageUrl, filename) {
  try {
    const response = await fetch(imageUrl);
    if (!response.ok) {
      throw new Error(`Failed to download image: ${response.statusText}`);
    }

    const blob = await response.blob();
    return new File([blob], filename, { type: blob.type });
  } catch (error) {
    console.error("Failed to download image:", error);
    throw error;
  }
}

/**
 * Download poster and backdrop images for a movie
 */
export async function downloadMovieImages(tmdbMovie) {
  const downloadedImages = {};

  try {
    // Download poster
    if (tmdbMovie.poster_path) {
      const posterUrl = buildPosterUrl(tmdbMovie.poster_path);
      downloadedImages.poster = await downloadImageAsFile(
        posterUrl,
        `poster_${tmdbMovie.id}.jpg`
      );
    }

    // Download backdrop as thumbnail
    if (tmdbMovie.backdrop_path) {
      const backdropUrl = buildBackdropUrl(tmdbMovie.backdrop_path);
      downloadedImages.thumbnail = await downloadImageAsFile(
        backdropUrl,
        `backdrop_${tmdbMovie.id}.jpg`
      );
    }

    return downloadedImages;
  } catch (error) {
    console.error("Failed to download movie images:", error);
    return downloadedImages; // Return partial results
  }
}

export default {
  searchMovies,
  getMovieDetails,
  getPopularMovies,
  getTopRatedMovies,
  getNowPlayingMovies,
  getUpcomingMovies,
  getMovieGenres,
  getMovieConfiguration,
  buildImageUrl,
  buildBackdropUrl,
  buildPosterUrl,
  buildThumbnailUrl,
  convertTmdbMovieToAppFormat,
  downloadImageAsFile,
  downloadMovieImages,
};
