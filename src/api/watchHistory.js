const API_BASE_URL = import.meta.env.VITE_API_URL || "";

const authHeaders = (token) => ({
  Authorization: `Bearer ${token}`,
});

// GET /api/user/watch-history/recent?limit=
export async function getRecentWatchHistory(limit = 10, token) {
  const params = new URLSearchParams({ limit: String(limit) });
  const res = await fetch(`${API_BASE_URL}/api/user/watch-history/recent?${params.toString()}`, {
    method: "GET",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to get recent watch history: ${res.status}`);
  return res.json();
}

// GET /api/user/watch-history/incomplete
export async function getIncompleteWatchHistory(token) {
  const res = await fetch(`${API_BASE_URL}/api/user/watch-history/incomplete`, {
    method: "GET",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to get incomplete watch history: ${res.status}`);
  return res.json();
}

// GET /api/user/watch-history?page=&size=
export async function getWatchHistory(page = 0, size = 20, token) {
  const params = new URLSearchParams({
    page: String(page),
    size: String(size),
  });
  const res = await fetch(`${API_BASE_URL}/api/user/watch-history?${params.toString()}`, {
    method: "GET",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to get watch history: ${res.status}`);
  return res.json();
}

// DELETE /api/user/watch-history/{id}
export async function deleteWatchHistoryItem(id, token) {
  const res = await fetch(`${API_BASE_URL}/api/user/watch-history/${id}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to delete watch history item: ${res.status}`);
  return res.json();
}

// DELETE /api/user/watch-history
export async function clearWatchHistory(token) {
  const res = await fetch(`${API_BASE_URL}/api/user/watch-history`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to clear watch history: ${res.status}`);
  return res.json();
}

