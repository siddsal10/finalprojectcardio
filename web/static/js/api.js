// Thin fetch wrapper: bearer token from storage, JSON in/out, readable errors.
let token = null;
try { token = localStorage.getItem("cs-token"); } catch (e) {}

export function setToken(t) {
  token = t;
  try { t ? localStorage.setItem("cs-token", t) : localStorage.removeItem("cs-token"); } catch (e) {}
}
export const hasToken = () => !!token;

export async function api(path, { method = "GET", body, form } = {}) {
  const headers = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  let payload;
  if (form) payload = form;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const res = await fetch(path, { method, headers, body: payload });
  let data = null;
  try { data = await res.json(); } catch (e) {}
  if (!res.ok) {
    const err = new Error((data && data.detail) || `${res.status} ${res.statusText}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

export const get = (p) => api(p);
export const post = (p, body) => api(p, { method: "POST", body });
export const patch = (p, body) => api(p, { method: "PATCH", body });

// Authenticated file download -> object URL (for <img> / links to protected attachments).
export async function blobUrl(path) {
  const res = await fetch(path, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (!res.ok) throw new Error(`${res.status}`);
  return URL.createObjectURL(await res.blob());
}
