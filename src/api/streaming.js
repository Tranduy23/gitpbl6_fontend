// Streaming API client for the movie app
// Uses native fetch available in browsers. Centralizes base URL and error handling.

/**
 * Resolve API base URL from environment or default
 */
const API_BASE_URL = import.meta?.env?.VITE_API_BASE_URL || "/api";
const API_KEY = import.meta?.env?.VITE_API_KEY || "";
const STATUS_BASE_URL =
  import.meta?.env?.VITE_MOVIE_STATUS_BASE_URL ||
  (API_BASE_URL.endsWith("/api")
    ? API_BASE_URL.replace(/\/api$/, "")
    : API_BASE_URL);

function buildUrl(path) {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE_URL}${path}`;
}

async function handleResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson
    ? await response.json().catch(() => ({}))
    : await response.text();

  // Check if response is an error object (even with 200 OK status)
  if (payload?.error) {
    const error = new Error(
      payload?.message ||
        payload?.error ||
        response.statusText ||
        "Request failed"
    );
    error.status = response.status || 500;
    error.details = payload;
    throw error;
  }

  if (!response.ok) {
    const error = new Error(
      payload?.message || response.statusText || "Request failed"
    );
    error.status = response.status;
    error.details = payload;
    throw error;
  }
  return payload;
}

function jsonFetch(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  const init = { ...options, headers };
  return fetch(buildUrl(path), init).then(handleResponse);
}

function withApiKey(params = {}) {
  const searchParams =
    params instanceof URLSearchParams ? params : new URLSearchParams(params);
  if (API_KEY && !searchParams.has("Key") && !searchParams.has("key")) {
    searchParams.append("Key", API_KEY);
  }
  return searchParams;
}

function statusFetch(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  const init = { ...options, headers };
  const base = STATUS_BASE_URL || "";
  const isAbsolute = path.startsWith("http://") || path.startsWith("https://");
  const sanitizedBase = base.endsWith("/") ? base.slice(0, -1) : base;
  const sanitizedPath = path.startsWith("/") || isAbsolute ? path : `/${path}`;
  const url = isAbsolute ? path : `${sanitizedBase}${sanitizedPath}`;
  return fetch(url, init).then(handleResponse);
}

// auth header util kept for backward compatibility (in case used elsewhere)
function getAuthHeader(token) {
  try {
    const bearer =
      token ||
      (typeof localStorage !== "undefined"
        ? localStorage.getItem("auth_token")
        : null);
    return bearer ? { Authorization: `Bearer ${bearer}` } : {};
  } catch (_) {
    return token ? { Authorization: `Bearer ${token}` } : {};
  }
}

// GET /api/streaming/movie/{id}
export function getStreamingInfo(movieId) {
  if (movieId == null) throw new Error("movieId is required");
  return jsonFetch(`/streaming/movie/${encodeURIComponent(movieId)}`, {
    method: "GET",
  });
}

// POST /api/streaming/start
// body: StreamingAPI.StartStreamingRequest
export function startStreaming(body) {
  if (!body || typeof body.movieId !== "number")
    throw new Error("Valid body with movieId is required");
  return jsonFetch("/streaming/start", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// PUT /api/streaming/quality
// body: StreamingAPI.UpdateQualityRequest
export function updateStreamingQuality(body) {
  if (!body || typeof body.movieId !== "number")
    throw new Error("Valid body with movieId is required");
  return jsonFetch("/streaming/quality", {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

// PUT /api/streaming/progress/{movieId}?currentTime={seconds}&totalTime={seconds}
export function updateStreamingProgress(movieId, currentTime, totalTime) {
  if (movieId == null) throw new Error("movieId is required");
  const params = new URLSearchParams({
    currentTime: String(Math.round(currentTime || 0)),
    totalTime: String(Math.round(totalTime || 0)),
  });
  return jsonFetch(
    `/streaming/progress/${encodeURIComponent(movieId)}?${params.toString()}`,
    {
      method: "PUT",
      headers: {
        ...getAuthHeader(),
      },
    }
  );
}

// GET /api/streaming/movie/{id}/subtitles
export function getAvailableSubtitles(movieId) {
  if (movieId == null) throw new Error("movieId is required");
  return jsonFetch(
    `/streaming/movie/${encodeURIComponent(movieId)}/subtitles`,
    { method: "GET" }
  );
}

// GET /api/streaming/statistics
export function getStreamingStatistics() {
  return jsonFetch("/streaming/statistics", { method: "GET" });
}

// SEARCH APIs
// POST /api/search/movies
export function searchMovies(body) {
  const payload = body || {};
  return jsonFetch("/search/movies", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// GET /api/search/quick?query=&limit=
export function quickSearch(query, limit = 10) {
  if (!query || !query.trim()) return Promise.resolve([]);
  const params = new URLSearchParams({
    query: query.trim(),
    limit: String(limit),
  });
  return jsonFetch(`/search/quick?${params.toString()}`, { method: "GET" });
}

// GET /api/search/movies/genre/{name}?limit=
export function searchByGenre(name, limit = 24) {
  if (!name) throw new Error("genre name is required");
  const params = new URLSearchParams({
    limit: String(limit),
  });
  return jsonFetch(
    `/search/movies/genre/${encodeURIComponent(name)}?${params.toString()}`,
    {
      method: "GET",
    }
  );
}

// GET /api/search/actor/{name}?page=&size=
export function searchByActor(name, page = 1, size = 24) {
  if (!name) throw new Error("actor name is required");
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
  });
  return jsonFetch(
    `/search/actor/${encodeURIComponent(name)}?${params.toString()}`,
    {
      method: "GET",
    }
  );
}

// GET /api/search/director/{name}?limit=
export function searchByDirector(name, limit = 24) {
  if (!name) throw new Error("director name is required");
  const params = new URLSearchParams({
    limit: String(limit),
  });
  return jsonFetch(
    `/search/director/${encodeURIComponent(name)}?${params.toString()}`,
    { method: "GET" }
  );
}

// GET /api/search/year/{year}?page=&size=
export function searchByYear(year, page = 1, size = 24) {
  if (year == null) throw new Error("year is required");
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
  });
  return jsonFetch(
    `/search/year/${encodeURIComponent(year)}?${params.toString()}`,
    { method: "GET" }
  );
}

// GET /api/search/suggestions?query=&limit=
export function getSearchSuggestions(query, limit = 8) {
  if (!query || !query.trim())
    return Promise.resolve({
      movies: [],
      genres: [],
      actors: [],
      directors: [],
    });
  const params = new URLSearchParams({
    query: query.trim(),
    limit: String(limit),
  });
  return jsonFetch(`/search/suggestions?${params.toString()}`, {
    method: "GET",
  });
}

// GET /api/search/popular?limit=
export function getPopularSearches(limit = 10) {
  const params = new URLSearchParams({ limit: String(limit) });
  return jsonFetch(`/search/popular?${params.toString()}`, { method: "GET" });
}

// GET /api/movies/featured?limit=
export function getFeaturedMovies(limit = 10) {
  const params = withApiKey({ limit: String(limit) });
  return statusFetch(`/movies/featured?${params.toString()}`, {
    method: "GET",
  });
}

// GET /api/movies/now-showing?limit=
export function getNowShowingMovies(limit = 10) {
  const params = withApiKey({ limit: String(limit) });
  return statusFetch(`/movies/now-showing?${params.toString()}`, {
    method: "GET",
  });
}

// GET /api/movies/upcoming?limit=
export function getUpcomingMovies(limit = 10) {
  const params = withApiKey({ limit: String(limit) });
  return statusFetch(`/movies/upcoming?${params.toString()}`, {
    method: "GET",
  });
}

// GET /api/movies/trending?limit=
export function getTrendingMovies(limit = 10) {
  const params = withApiKey({ limit: String(limit) });
  return statusFetch(`/movies/trending?${params.toString()}`, {
    method: "GET",
  });
}

// GET /api/movies/recent-ratings?limit=
export function getRecentRatings(limit = 5) {
  const params = new URLSearchParams({ limit: String(limit) });
  return jsonFetch(`/movies/recent-ratings?${params.toString()}`, {
    method: "GET",
  });
}

// Favorites APIs moved to src/api/favorites.js

// Helper: build HLS/DASH player source from startStreaming response
export function toPlayerSource(startResponse) {
  if (!startResponse || !startResponse.streamingUrl) return null;
  return {
    src: startResponse.streamingUrl,
    type: startResponse.streamingUrl.endsWith(".m3u8")
      ? "application/x-mpegURL"
      : "application/dash+xml",
    drm: startResponse.drm || undefined,
  };
}

export default {
  getStreamingInfo,
  startStreaming,
  updateStreamingQuality,
  updateStreamingProgress,
  getAvailableSubtitles,
  getStreamingStatistics,
  // search exports
  searchMovies,
  quickSearch,
  searchByGenre,
  searchByActor,
  searchByDirector,
  searchByYear,
  getSearchSuggestions,
  getPopularSearches,
  // movies exports
  getFeaturedMovies,
  getNowShowingMovies,
  getUpcomingMovies,
  getTrendingMovies,
  getRecentRatings,
  toPlayerSource,
};
