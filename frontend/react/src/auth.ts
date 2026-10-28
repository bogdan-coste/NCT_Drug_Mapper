export type Session = {
  token: string;
  username: string;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const STORAGE_KEY = "nct.session";

export function loadSession(): Session | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return null;
    }

    const parsed = JSON.parse(raw) as Partial<Session>;
    if (typeof parsed.token !== "string" || typeof parsed.username !== "string") {
      return null;
    }

    return { token: parsed.token, username: parsed.username };
  } catch {
    return null;
  }
}

export function saveSession(session: Session): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  localStorage.removeItem(STORAGE_KEY);
}

/** Bearer token for API calls, or null when signed out. */
export function getToken(): string | null {
  return loadSession()?.token ?? null;
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
  } catch {
    // Not a JSON body — fall back to the status line.
  }
  return `${response.status} ${response.statusText}`;
}

/** POST /login expects form-encoded fields, not JSON. */
export async function login(
  username: string,
  password: string,
): Promise<Session> {
  const response = await fetch(`${API_BASE}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username, password }),
  });

  if (!response.ok) {
    if (response.status === 401) {
      throw new Error("Incorrect username or password.");
    }
    throw new Error(await readError(response));
  }

  const payload = (await response.json()) as { access_token: string };
  const session: Session = { token: payload.access_token, username };
  saveSession(session);
  return session;
}

export async function signup(
  username: string,
  password: string,
): Promise<void> {
  const response = await fetch(`${API_BASE}/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    if (response.status === 409) {
      throw new Error("That username is already taken.");
    }
    throw new Error(await readError(response));
  }
}

/** Revoke a session server-side. Safe to fire-and-forget on sign-out. */
export async function logout(token: string): Promise<void> {
  await fetch(`${API_BASE}/logout`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
}

/** Validate a stored token; throws when it is missing or expired. */
export async function fetchMe(token: string): Promise<string> {
  const response = await fetch(`${API_BASE}/users/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (response.status === 401) {
    throw new Error("Session expired.");
  }
  if (!response.ok) {
    throw new Error(await readError(response));
  }

  const payload = (await response.json()) as { username: string };
  return payload.username;
}
