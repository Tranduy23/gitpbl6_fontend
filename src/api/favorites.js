// Favorites API client

const API_BASE_URL = import.meta?.env?.VITE_API_BASE_URL || "/api";

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

// POST /api/user/favorites/{id}
export function addFavorite(movieId, token) {
  if (movieId == null) throw new Error("movieId is required");
  const id = Number(movieId);
  return jsonFetch(`/user/favorites/${encodeURIComponent(id)}`, {
    method: "POST",
    headers: { ...getAuthHeader(token), Accept: "*/*" },
  });
}

// GET /api/user/favorites?page=&size=
export function getFavorites({ page = 0, size = 24 } = {}, token) {
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
  });
  return jsonFetch(`/user/favorites?${params.toString()}`, {
    method: "GET",
    headers: { ...getAuthHeader(token), Accept: "*/*" },
  });
}

// DELETE /api/user/favorites/{movieId}
export function removeFavorite(movieIdOrFavorite, token) {
  if (movieIdOrFavorite == null) throw new Error("movie id is required");
  const movieId = (() => {
    if (typeof movieIdOrFavorite === "object" && movieIdOrFavorite !== null) {
      if (movieIdOrFavorite.movieId != null) return movieIdOrFavorite.movieId;
      if (movieIdOrFavorite.id != null) return movieIdOrFavorite.id;
    }
    return movieIdOrFavorite;
  })();
  if (movieId == null) throw new Error("movie id is required");
  const id = Number(movieId);
  return jsonFetch(`/user/favorites/${encodeURIComponent(id)}`, {
    method: "DELETE",
    headers: { ...getAuthHeader(token), Accept: "*/*" },
  });
}

export default {
  addFavorite,
  getFavorites,
  removeFavorite,
};
