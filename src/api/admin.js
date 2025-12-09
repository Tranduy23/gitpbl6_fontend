// Admin API client for movie management
// Handles CRUD operations for movies, users, and other admin functions

/**
 * Resolve API base URL from environment or default
 */
const API_BASE_URL = import.meta?.env?.VITE_API_BASE_URL || "/api";

function buildUrl(path) {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE_URL}${path}`;
}

function getAuthHeader() {
  try {
    const token = localStorage.getItem("auth_token");
    return token ? { Authorization: `Bearer ${token}` } : {};
  } catch (_) {
    return {};
  }
}

async function handleResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson
    ? await response.json().catch(() => ({}))
    : await response.text();

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

// Helper function for JSON requests
function jsonFetch(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...getAuthHeader(),
    ...(options.headers || {}),
  };
  const init = { ...options, headers };
  return fetch(buildUrl(path), init).then(handleResponse);
}

// Helper function for multipart form data requests
function formDataFetch(path, formData, options = {}) {
  const headers = {
    ...getAuthHeader(),
    ...(options.headers || {}),
  };
  // Don't set Content-Type for FormData, let browser set it with boundary
  const init = {
    ...options,
    headers,
    body: formData,
  };
  return fetch(buildUrl(path), init).then(handleResponse);
}

// Helper for application/x-www-form-urlencoded requests
function formEncodedFetch(path, params, options = {}) {
  const headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    ...getAuthHeader(),
    ...(options.headers || {}),
  };
  const init = {
    ...options,
    headers,
    body: params instanceof URLSearchParams ? params.toString() : params,
  };
  return fetch(buildUrl(path), init).then(handleResponse);
}

// ===== USER MANAGEMENT =====

// GET /api/admin/users
export function getUsers(page = 0, size = 10, sort = "createdAt,desc") {
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
    sort: String(sort),
  });
  return jsonFetch(`/admin/users?${params.toString()}`, { method: "GET" });
}

// GET /api/admin/users/{id}
export function getUserById(userId) {
  if (!userId) throw new Error("userId is required");
  return jsonFetch(`/admin/users/${encodeURIComponent(userId)}`, {
    method: "GET",
  });
}

// PUT /api/admin/users/{id}
export function updateUser(userId, userData) {
  if (!userId) throw new Error("userId is required");
  if (!userData) throw new Error("userData is required");
  return jsonFetch(`/admin/users/${encodeURIComponent(userId)}`, {
    method: "PUT",
    body: JSON.stringify(userData),
  });
}

// DELETE /api/admin/users/{id}
export function deleteUser(userId) {
  if (!userId) throw new Error("userId is required");
  return jsonFetch(`/admin/users/${encodeURIComponent(userId)}`, {
    method: "DELETE",
  });
}

// PUT /api/admin/users/{id}/status
export function updateUserStatus(userId, enabled) {
  if (!userId) throw new Error("userId is required");
  if (enabled === undefined) throw new Error("enabled status is required");
  return jsonFetch(
    `/admin/users/${encodeURIComponent(userId)}/status?enabled=${enabled}`,
    {
      method: "PUT",
    }
  );
}

// PUT /api/admin/users/{id}/roles
export function updateUserRoles(userId, roles) {
  if (!userId) throw new Error("userId is required");
  if (!roles) throw new Error("roles are required");
  return jsonFetch(`/admin/users/${encodeURIComponent(userId)}/roles`, {
    method: "PUT",
    body: JSON.stringify(roles),
  });
}

// POST /api/admin/users/bulk-disable
export function bulkDisableUsers(userIds, reason = "Policy violation") {
  if (!userIds || !Array.isArray(userIds) || userIds.length === 0) {
    throw new Error("userIds array is required");
  }
  return jsonFetch(
    `/admin/users/bulk-disable?reason=${encodeURIComponent(reason)}`,
    {
      method: "POST",
      body: JSON.stringify(userIds),
    }
  );
}

// ===== MOVIE MANAGEMENT =====

// GET /api/admin/movies
export function getMovies(page = 0, size = 10, sort = "createdAt,desc") {
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
    sort: String(sort),
  });
  return jsonFetch(`/admin/movies?${params.toString()}`, { method: "GET" });
}

// GET /api/admin/movies/{id}
export function getMovieById(movieId) {
  if (!movieId) throw new Error("movieId is required");
  return jsonFetch(`/admin/movies/${encodeURIComponent(movieId)}`, {
    method: "GET",
  });
}

// PUBLIC: GET /api/movies/{id}
export function getPublicMovieById(movieId) {
  if (!movieId) throw new Error("movieId is required");
  return jsonFetch(`/movies/${encodeURIComponent(movieId)}`, { method: "GET" });
}

// POST /api/movies/create-form (multipart form data)
export function createMovie(movieData) {
  if (!movieData) throw new Error("movieData is required");

  const formData = new FormData();

  // Add text fields
  if (movieData.title) formData.append("title", movieData.title);
  if (movieData.synopsis) formData.append("synopsis", movieData.synopsis);
  if (movieData.year) formData.append("year", String(movieData.year));
  if (movieData.categories) formData.append("categories", movieData.categories);
  if (movieData.actors) formData.append("actors", movieData.actors);
  if (movieData.directors) {
    formData.append("directors", movieData.directors);
    // Also provide singular directorName if backend expects it
    try {
      const firstDirector = String(movieData.directors)
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)[0];
      if (firstDirector) formData.append("directorName", firstDirector);
    } catch (_) {
      // Ignore parsing errors
    }
  }
  if (movieData.directorName)
    formData.append("directorName", movieData.directorName);
  if (movieData.country) formData.append("country", movieData.country);
  if (movieData.language) formData.append("language", movieData.language);
  if (movieData.ageRating) formData.append("ageRating", movieData.ageRating);
  if (movieData.trailerUrl) formData.append("trailerUrl", movieData.trailerUrl);
  if (movieData.videoUrl) formData.append("videoUrl", movieData.videoUrl);
  if (movieData.imdbRating)
    formData.append("imdbRating", String(movieData.imdbRating));
  if (movieData.releaseDate)
    formData.append("releaseDate", movieData.releaseDate);
  if (movieData.maxDownloadQuality)
    formData.append("maxDownloadQuality", movieData.maxDownloadQuality);
  if (movieData.isAvailable !== undefined)
    formData.append("isAvailable", String(movieData.isAvailable));
  if (movieData.isFeatured !== undefined)
    formData.append("isFeatured", String(movieData.isFeatured));
  if (movieData.isTrending !== undefined)
    formData.append("isTrending", String(movieData.isTrending));
  if (movieData.downloadEnabled !== undefined)
    formData.append("downloadEnabled", String(movieData.downloadEnabled));

  // Add file fields
  if (movieData.poster) formData.append("poster", movieData.poster);
  if (movieData.thumbnail) formData.append("thumbnail", movieData.thumbnail);
  if (movieData.video) formData.append("video", movieData.video);
  if (movieData.subtitle) formData.append("subtitle", movieData.subtitle);

  return formDataFetch("/movies/create-form", formData, { method: "POST" });
}

// PUT /api/admin/movies/{id}
export function updateMovie(movieId, movieData) {
  if (!movieId) throw new Error("movieId is required");
  if (!movieData) throw new Error("movieData is required");
  return jsonFetch(`/admin/movies/${encodeURIComponent(movieId)}`, {
    method: "PUT",
    body: JSON.stringify(movieData),
  });
}

// PUT /api/admin/movies/{id} (with FormData support)
export function updateMovieWithFormData(movieId, movieData) {
  if (!movieId) throw new Error("movieId is required");
  if (!movieData) throw new Error("movieData is required");

  const formData = new FormData();

  // Add text fields
  if (movieData.title) formData.append("title", movieData.title);
  if (movieData.synopsis) formData.append("synopsis", movieData.synopsis);
  if (movieData.year) formData.append("year", String(movieData.year));
  if (movieData.categories) formData.append("categories", movieData.categories);
  if (movieData.actors) formData.append("actors", movieData.actors);
  if (movieData.directors) {
    formData.append("directors", movieData.directors);
    // Also provide singular directorName if backend expects it
    try {
      const firstDirector = String(movieData.directors)
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)[0];
      if (firstDirector) formData.append("directorName", firstDirector);
    } catch (_) {
      // Ignore parsing errors
    }
  }
  if (movieData.directorName)
    formData.append("directorName", movieData.directorName);
  if (movieData.country) formData.append("country", movieData.country);
  if (movieData.language) formData.append("language", movieData.language);
  if (movieData.ageRating) formData.append("ageRating", movieData.ageRating);
  if (movieData.trailerUrl) formData.append("trailerUrl", movieData.trailerUrl);
  if (movieData.videoUrl) formData.append("videoUrl", movieData.videoUrl);
  if (movieData.imdbRating)
    formData.append("imdbRating", String(movieData.imdbRating));
  if (movieData.releaseDate)
    formData.append("releaseDate", movieData.releaseDate);
  if (movieData.maxDownloadQuality)
    formData.append("maxDownloadQuality", movieData.maxDownloadQuality);
  if (movieData.isAvailable !== undefined)
    formData.append("isAvailable", String(movieData.isAvailable));
  if (movieData.isFeatured !== undefined)
    formData.append("isFeatured", String(movieData.isFeatured));
  if (movieData.isTrending !== undefined)
    formData.append("isTrending", String(movieData.isTrending));
  if (movieData.downloadEnabled !== undefined)
    formData.append("downloadEnabled", String(movieData.downloadEnabled));

  // Add file fields
  if (movieData.poster) formData.append("poster", movieData.poster);
  if (movieData.thumbnail) formData.append("thumbnail", movieData.thumbnail);
  if (movieData.video) formData.append("video", movieData.video);
  if (movieData.trailer) formData.append("trailer", movieData.trailer);
  if (movieData.subtitle) formData.append("subtitle", movieData.subtitle);

  return formDataFetch(
    `/admin/movies/${encodeURIComponent(movieId)}`,
    formData,
    { method: "PUT" }
  );
}

// DELETE /api/admin/movies/{id}
export function deleteMovie(movieId) {
  if (!movieId) throw new Error("movieId is required");
  return jsonFetch(`/admin/movies/${encodeURIComponent(movieId)}`, {
    method: "DELETE",
  });
}

// PUT /api/admin/movies/{id}/availability
export function toggleMovieAvailability(movieId, available) {
  if (!movieId) throw new Error("movieId is required");
  if (available === undefined) throw new Error("available status is required");
  return jsonFetch(
    `/admin/movies/${encodeURIComponent(
      movieId
    )}/availability?available=${available}`,
    {
      method: "PUT",
    }
  );
}

// POST /api/admin/movies/{id}/poster
export function uploadMoviePoster(movieId, posterFile) {
  if (!movieId) throw new Error("movieId is required");
  if (!posterFile) throw new Error("posterFile is required");

  const formData = new FormData();
  formData.append("poster", posterFile);

  return formDataFetch(
    `/admin/movies/${encodeURIComponent(movieId)}/poster`,
    formData,
    {
      method: "POST",
    }
  );
}

// POST /api/admin/movies/{id}/trailer
export function uploadMovieTrailer(movieId, trailerFile) {
  if (!movieId) throw new Error("movieId is required");
  if (!trailerFile) throw new Error("trailerFile is required");

  const formData = new FormData();
  formData.append("trailer", trailerFile);

  return formDataFetch(
    `/admin/movies/${encodeURIComponent(movieId)}/trailer`,
    formData,
    {
      method: "POST",
    }
  );
}

// POST /api/admin/movies/{id}/subtitle
export function uploadMovieSubtitle(movieId, subtitleFile, subtitleMeta = {}) {
  if (!movieId) throw new Error("movieId is required");
  if (!subtitleFile) throw new Error("subtitleFile is required");

  const formData = new FormData();
  formData.append("subtitle", subtitleFile);

  // Some backends expect subtitle metadata as request params in multipart
  const languageCode = subtitleMeta.languageCode || "vi";
  const languageName = subtitleMeta.languageName || "Vietnamese";
  const isDefault =
    subtitleMeta.isDefault !== undefined ? subtitleMeta.isDefault : true;

  formData.append("languageCode", languageCode);
  formData.append("languageName", languageName);
  formData.append("isDefault", String(isDefault));

  return formDataFetch(
    `/admin/movies/${encodeURIComponent(movieId)}/subtitle`,
    formData,
    {
      method: "POST",
    }
  );
}

// ===== COUNTRY MANAGEMENT =====

// GET /api/admin/countries
export function getCountries() {
  return jsonFetch("/admin/countries", { method: "GET" });
}

// GET /api/admin/countries/{id}
export function getCountryById(countryId) {
  if (!countryId) throw new Error("countryId is required");
  return jsonFetch(`/admin/countries/${encodeURIComponent(countryId)}`, {
    method: "GET",
  });
}

// POST /api/admin/countries
export function createCountry(countryData) {
  if (!countryData) throw new Error("countryData is required");

  const formData = new FormData();
  if (countryData.name) formData.append("name", countryData.name);
  if (countryData.flag) formData.append("flag", countryData.flag);
  if (countryData.isActive !== undefined)
    formData.append("isActive", String(countryData.isActive));

  return formDataFetch("/admin/countries", formData, {
    method: "POST",
  });
}

// PUT /api/admin/countries/{id}
export function updateCountry(countryId, countryData) {
  if (!countryId) throw new Error("countryId is required");
  if (!countryData) throw new Error("countryData is required");

  const formData = new FormData();
  if (countryData.name) formData.append("name", countryData.name);
  if (countryData.flag) formData.append("flag", countryData.flag);
  if (countryData.isActive !== undefined)
    formData.append("isActive", String(countryData.isActive));

  return formDataFetch(
    `/admin/countries/${encodeURIComponent(countryId)}`,
    formData,
    {
      method: "PUT",
    }
  );
}

// DELETE /api/admin/countries/{id}
export function deleteCountry(countryId) {
  if (!countryId) throw new Error("countryId is required");
  return jsonFetch(`/admin/countries/${encodeURIComponent(countryId)}`, {
    method: "DELETE",
  });
}

// PUT /api/admin/countries/{id}/activate
export function activateCountry(countryId) {
  if (!countryId) throw new Error("countryId is required");
  return jsonFetch(
    `/admin/countries/${encodeURIComponent(countryId)}/activate`,
    {
      method: "PUT",
    }
  );
}

// PUT /api/admin/countries/{id}/deactivate
export function deactivateCountry(countryId) {
  if (!countryId) throw new Error("countryId is required");
  return jsonFetch(
    `/admin/countries/${encodeURIComponent(countryId)}/deactivate`,
    {
      method: "PUT",
    }
  );
}

// ===== ACTOR MANAGEMENT =====

// GET /api/actors?page=&size=
export function getActors(page = 0, size = 20) {
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
  });
  return jsonFetch(`/actors?${params.toString()}`, { method: "GET" });
}

// GET /api/actors/{id}
export function getActorById(actorId) {
  if (!actorId) throw new Error("actorId is required");
  return jsonFetch(`/actors/${encodeURIComponent(actorId)}`, { method: "GET" });
}

// POST /api/admin/actors/form (multipart)
export function createActor(actorData) {
  if (!actorData) throw new Error("actorData is required");
  const formData = new FormData();
  if (actorData.name) formData.append("name", actorData.name);
  if (actorData.dob) formData.append("dob", actorData.dob);
  if (actorData.birthDate) formData.append("birthDate", actorData.birthDate);
  if (!actorData.birthDate && actorData.dob)
    formData.append("birthDate", actorData.dob);
  if (actorData.description)
    formData.append("description", actorData.description);
  if (actorData.biography) formData.append("biography", actorData.biography);
  if (!actorData.biography && actorData.description)
    formData.append("biography", actorData.description);
  if (actorData.nationality)
    formData.append("nationality", actorData.nationality);
  if (actorData.file) formData.append("file", actorData.file);
  return formDataFetch(`/admin/actors/form`, formData, { method: "POST" });
}

// PUT /api/admin/actors/{id}
export function updateActor(actorId, actorData = {}) {
  if (!actorId) throw new Error("actorId is required");
  if (!actorData) throw new Error("actorData is required");

  const formData = new FormData();
  if (actorData.name) formData.append("name", actorData.name);
  if (actorData.dob) formData.append("dob", actorData.dob);
  if (actorData.birthDate) formData.append("birthDate", actorData.birthDate);
  if (!actorData.birthDate && actorData.dob)
    formData.append("birthDate", actorData.dob);
  if (actorData.description)
    formData.append("description", actorData.description);
  if (actorData.biography) formData.append("biography", actorData.biography);
  if (!actorData.biography && actorData.description)
    formData.append("biography", actorData.description);
  if (actorData.nationality)
    formData.append("nationality", actorData.nationality);
  if (actorData.file) formData.append("file", actorData.file);

  return formDataFetch(
    `/admin/actors/${encodeURIComponent(actorId)}`,
    formData,
    {
      method: "PUT",
    }
  );
}

// PUT /api/admin/actors/{id} (multipart form data for image updates)
export function updateActorWithFormData(actorId, actorData) {
  return updateActor(actorId, actorData);
}

// DELETE /api/admin/actors/{id}
export function deleteActor(actorId) {
  if (!actorId) throw new Error("actorId is required");
  return jsonFetch(`/admin/actors/${encodeURIComponent(actorId)}`, {
    method: "DELETE",
  });
}

// ===== DIRECTOR MANAGEMENT =====

// GET /api/directors?page=&size=&sort=&direction=
export function getDirectors(
  page = 0,
  size = 10,
  sort = "name",
  direction = "asc"
) {
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
    sort: String(sort),
    direction: String(direction),
  });
  return jsonFetch(`/directors?${params.toString()}`, { method: "GET" });
}

// GET /api/directors/all
export function getAllDirectors() {
  return jsonFetch(`/directors/all`, { method: "GET" });
}

// GET /api/directors/{id}
export function getDirectorById(directorId) {
  if (!directorId) throw new Error("directorId is required");
  return jsonFetch(`/directors/${encodeURIComponent(directorId)}`, {
    method: "GET",
  });
}

// POST /api/directors
export function createDirector(directorData) {
  if (!directorData) throw new Error("directorData is required");
  const formData = new FormData();
  if (directorData.name) formData.append("name", directorData.name);
  if (directorData.biography)
    formData.append("biography", directorData.biography);
  if (directorData.birthDate)
    formData.append("birthDate", directorData.birthDate);
  if (directorData.nationality)
    formData.append("nationality", directorData.nationality);
  if (directorData.photo) formData.append("photo", directorData.photo);
  return formDataFetch(`/directors`, formData, { method: "POST" });
}

// PUT /api/directors/{id}
export function updateDirector(directorId, directorData = {}) {
  if (!directorId) throw new Error("directorId is required");
  if (!directorData) throw new Error("directorData is required");

  const formData = new FormData();
  if (directorData.name) formData.append("name", directorData.name);
  if (directorData.biography)
    formData.append("biography", directorData.biography);
  if (directorData.birthDate)
    formData.append("birthDate", directorData.birthDate);
  if (directorData.nationality)
    formData.append("nationality", directorData.nationality);
  if (directorData.photo) formData.append("photo", directorData.photo);

  return formDataFetch(
    `/directors/${encodeURIComponent(directorId)}`,
    formData,
    {
      method: "PUT",
    }
  );
}

// PUT /api/directors/{id} (multipart for photo updates)
export function updateDirectorWithFormData(directorId, directorData) {
  return updateDirector(directorId, directorData);
}

// DELETE /api/directors/{id}
export function deleteDirector(directorId) {
  if (!directorId) throw new Error("directorId is required");
  return jsonFetch(`/directors/${encodeURIComponent(directorId)}`, {
    method: "DELETE",
  });
}

// ===== CATEGORY MANAGEMENT =====

// GET /api/admin/categories?page=&size=
export function getCategories(page = 0, size = 20) {
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
  });
  return jsonFetch(`/admin/categories?${params.toString()}`, { method: "GET" });
}

// GET /api/admin/categories/{name}
export function getCategoryByName(name) {
  if (!name) throw new Error("name is required");
  return jsonFetch(`/admin/categories/${encodeURIComponent(name)}`, {
    method: "GET",
  });
}

// POST /api/admin/categories
export function createCategory(categoryData = {}) {
  if (!categoryData) throw new Error("categoryData is required");
  // Many backends (e.g., Spring @RequestParam) expect params instead of JSON body
  const params = new URLSearchParams();
  if (categoryData.name) params.append("name", categoryData.name);
  if (categoryData.displayName)
    params.append("displayName", categoryData.displayName);
  if (categoryData.description)
    params.append("description", categoryData.description);
  if (categoryData.icon) params.append("icon", categoryData.icon);
  return formEncodedFetch(`/admin/categories`, params, {
    method: "POST",
  });
}

// PUT /api/admin/categories/{name}
export function updateCategory(name, categoryData = {}) {
  if (!name) throw new Error("name is required");
  if (!categoryData) throw new Error("categoryData is required");
  // Support backends that expect request params rather than JSON
  const params = new URLSearchParams();
  if (categoryData.displayName)
    params.append("displayName", categoryData.displayName);
  if (categoryData.description)
    params.append("description", categoryData.description);
  if (categoryData.icon) params.append("icon", categoryData.icon);
  // Optionally allow renaming if provided via categoryData.name
  if (categoryData.name && categoryData.name !== name) {
    params.append("newName", categoryData.name);
  }
  return formEncodedFetch(
    `/admin/categories/${encodeURIComponent(name)}`,
    params,
    {
      method: "PUT",
    }
  );
}

// DELETE /api/admin/categories/{name}
export function deleteCategory(name) {
  if (!name) throw new Error("name is required");
  return jsonFetch(`/admin/categories/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

// POST /api/admin/categories/{name}/assign/{movieId}
export function assignMovieToCategory(name, movieId) {
  if (!name) throw new Error("name is required");
  if (!movieId) throw new Error("movieId is required");
  return jsonFetch(
    `/admin/categories/${encodeURIComponent(name)}/assign/${encodeURIComponent(
      movieId
    )}`,
    { method: "POST" }
  );
}

// DELETE /api/admin/categories/{name}/assign/{movieId}
export function unassignMovieFromCategory(name, movieId) {
  if (!name) throw new Error("name is required");
  if (!movieId) throw new Error("movieId is required");
  return jsonFetch(
    `/admin/categories/${encodeURIComponent(name)}/assign/${encodeURIComponent(
      movieId
    )}`,
    { method: "DELETE" }
  );
}
// ===== STATISTICS =====

// GET /api/admin/dashboard
export function getAdminStatistics() {
  return jsonFetch("/admin/dashboard", { method: "GET" });
}

// GET /api/admin/statistics/movies
export function getMovieStatistics() {
  return jsonFetch("/admin/statistics/movies", { method: "GET" });
}

// GET /api/admin/statistics/users
export function getUserStatistics() {
  return jsonFetch("/admin/statistics/users", { method: "GET" });
}

// ===== REPORTS =====

// GET /api/admin/reports
export function getReports(page = 0, size = 10, sort = "createdAt,desc") {
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
    sort: String(sort),
  });
  return jsonFetch(`/admin/reports?${params.toString()}`, { method: "GET" });
}

// PUT /api/admin/reports/{id}/resolve
export function resolveReport(reportId) {
  if (!reportId) throw new Error("reportId is required");
  return jsonFetch(`/admin/reports/${encodeURIComponent(reportId)}/resolve`, {
    method: "PUT",
  });
}

// ===== ADDITIONAL ADMIN APIs =====

// GET /api/admin/dashboard (same as getAdminStatistics)
export function getDashboardAnalytics() {
  return jsonFetch("/admin/dashboard", { method: "GET" });
}

// GET /api/admin/analytics/trending
export function getTrendingContent(period = "week") {
  const params = new URLSearchParams({ period: String(period) });
  return jsonFetch(`/admin/analytics/trending?${params.toString()}`, {
    method: "GET",
  });
}

// ===== DASHBOARD & STATISTICS (additional) =====

// GET /api/admin/stats
export function getAdminStats() {
  return jsonFetch("/admin/stats", { method: "GET" });
}

// GET /api/admin/stats/monthly/{year}
export function getMonthlyStats(year) {
  const path = `/admin/stats/monthly/${encodeURIComponent(year)}`;
  return jsonFetch(path, { method: "GET" });
}

export default {
  // User management
  getUsers,
  getUserById,
  updateUser,
  deleteUser,
  updateUserStatus,
  updateUserRoles,
  bulkDisableUsers,

  // Movie management
  getMovies,
  getMovieById,
  createMovie,
  updateMovie,
  updateMovieWithFormData,
  deleteMovie,
  toggleMovieAvailability,
  uploadMoviePoster,
  uploadMovieTrailer,
  uploadMovieSubtitle,
  getPublicMovieById,

  // Country management
  getCountries,
  getCountryById,
  createCountry,
  updateCountry,
  deleteCountry,
  activateCountry,
  deactivateCountry,

  // Actor management
  getActors,
  getActorById,
  createActor,
  updateActor,
  updateActorWithFormData,
  deleteActor,

  // Director management
  getDirectors,
  getAllDirectors,
  getDirectorById,
  createDirector,
  updateDirector,
  updateDirectorWithFormData,
  deleteDirector,

  // Category management
  getCategories,
  getCategoryByName,
  createCategory,
  updateCategory,
  deleteCategory,
  assignMovieToCategory,
  unassignMovieFromCategory,

  // Statistics
  getAdminStatistics,
  getMovieStatistics,
  getUserStatistics,

  // Reports
  getReports,
  resolveReport,

  // Additional APIs
  getDashboardAnalytics,
  getTrendingContent,
  getAdminStats,
  getMonthlyStats,
};
