const API_BASE_URL = import.meta.env.VITE_API_URL || "";

const authHeaders = (token) => ({
  Authorization: `Bearer ${token}`,
});

export async function listCollections(token) {
  const res = await fetch(`${API_BASE_URL}/api/watchlist/collections`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to list collections: ${res.status}`);
  return res.json();
}

export async function getCollection(id, token) {
  const res = await fetch(`${API_BASE_URL}/api/watchlist/collections/${id}`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to get collection: ${res.status}`);
  return res.json();
}

export async function getCollectionMovies(id, token) {
  const res = await fetch(
    `${API_BASE_URL}/api/watchlist/collections/${id}/movies`,
    { headers: authHeaders(token) }
  );
  if (!res.ok) throw new Error(`Failed to get movies: ${res.status}`);
  return res.json();
}

export async function createCollection({ name, description }, token) {
  const body = new URLSearchParams();
  if (name) body.append("name", name);
  if (description) body.append("description", description);
  const res = await fetch(`${API_BASE_URL}/api/watchlist/collections`, {
    method: "POST",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: body.toString(),
  });
  if (!res.ok) throw new Error(`Failed to create collection: ${res.status}`);
  return res.json();
}

export async function updateCollection(id, { name, description }, token) {
  const body = new URLSearchParams();
  if (name != null) body.append("name", name);
  if (description != null) body.append("description", description);
  const res = await fetch(`${API_BASE_URL}/api/watchlist/collections/${id}`, {
    method: "PUT",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: body.toString(),
  });
  if (!res.ok) throw new Error(`Failed to update collection: ${res.status}`);
  return res.json();
}

export async function deleteCollection(id, token) {
  const res = await fetch(`${API_BASE_URL}/api/watchlist/collections/${id}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to delete collection: ${res.status}`);
  return true;
}

export async function addMovieToWatchlist(
  movieId,
  { notes, priority, collectionId },
  token
) {
  const body = new URLSearchParams();
  if (notes != null) body.append("notes", String(notes));
  if (priority != null) body.append("priority", String(priority));
  if (collectionId != null) body.append("collectionId", String(collectionId));
  const res = await fetch(`${API_BASE_URL}/api/watchlist/movie/${movieId}`, {
    method: "POST",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: body.toString(),
  });
  if (!res.ok) throw new Error(`Failed to add movie: ${res.status}`);
  return res.json();
}

export async function listAllWatchlistMovies(token) {
  const res = await fetch(`${API_BASE_URL}/api/watchlist`, {
    headers: authHeaders(token),
  });
  if (!res.ok)
    throw new Error(`Failed to list watchlist movies: ${res.status}`);
  return res.json();
}

export async function isMovieInWatchlist(movieId, token) {
  const res = await fetch(`${API_BASE_URL}/api/watchlist/check/${movieId}`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to check watchlist: ${res.status}`);
  return res.json();
}

export async function removeMovieFromWatchlist(movieId, token) {
  const res = await fetch(`${API_BASE_URL}/api/watchlist/movie/${movieId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to remove movie: ${res.status}`);
  return true;
}
