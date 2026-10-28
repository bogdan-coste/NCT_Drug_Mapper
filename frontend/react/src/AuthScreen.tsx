import { useState, type FormEvent } from "react";
import "./AuthScreen.css";
import { login, signup, type Session } from "./auth";

type Mode = "login" | "signup";

export default function AuthScreen({
  onAuthenticated,
}: {
  onAuthenticated: (session: Session) => void;
}) {
  const [mode, setMode] = useState<Mode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const isSignup = mode === "signup";

  function switchMode(next: Mode) {
    setMode(next);
    setError(null);
    setConfirm("");
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) {
      return;
    }

    const name = username.trim();
    if (!name || !password) {
      setError("Enter a username and password.");
      return;
    }
    if (isSignup && password !== confirm) {
      setError("Passwords do not match.");
      return;
    }

    setError(null);
    setBusy(true);

    try {
      if (isSignup) {
        await signup(name, password);
      }
      onAuthenticated(await login(name, password));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="aurora" aria-hidden="true" />
      <div className="grid-overlay" aria-hidden="true" />

      <div className="auth-shell">
        <div className="auth-brand">
          <div className="brand-mark">
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 2l1.9 5.1L19 9l-5.1 1.9L12 16l-1.9-5.1L5 9l5.1-1.9L12 2zM18 15l.9 2.4 2.4.9-2.4.9-.9 2.4-.9-2.4-2.4-.9 2.4-.9L18 15z" />
            </svg>
          </div>
          <div>
            <div className="brand-title">Therapeutic Mapping</div>
            <div className="brand-sub">AI Platform</div>
          </div>
        </div>

        <form className="card auth-card" onSubmit={handleSubmit}>
          <h1 className="auth-title">
            {isSignup ? "Create your account" : "Welcome back"}
          </h1>
          <p className="auth-sub">
            {isSignup
              ? "Sign up to run ontology mapping analyses."
              : "Sign in to continue to the workspace."}
          </p>

          <div
            className="auth-tabs"
            role="tablist"
            aria-label="Authentication mode"
          >
            <button
              type="button"
              role="tab"
              aria-selected={!isSignup}
              className={`auth-tab ${!isSignup ? "active" : ""}`}
              onClick={() => switchMode("login")}
            >
              Sign in
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={isSignup}
              className={`auth-tab ${isSignup ? "active" : ""}`}
              onClick={() => switchMode("signup")}
            >
              Create account
            </button>
          </div>

          <div className="auth-fields">
            <label className="field">
              <span className="field-label">Username</span>
              <input
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                autoComplete="username"
                autoFocus
                spellCheck={false}
              />
            </label>

            <label className="field">
              <span className="field-label">Password</span>
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete={isSignup ? "new-password" : "current-password"}
              />
            </label>

            {isSignup && (
              <label className="field">
                <span className="field-label">Confirm password</span>
                <input
                  type="password"
                  value={confirm}
                  onChange={(event) => setConfirm(event.target.value)}
                  autoComplete="new-password"
                />
              </label>
            )}
          </div>

          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}

          <button type="submit" className="run-btn auth-submit" disabled={busy}>
            {busy ? "Please wait…" : isSignup ? "Create account" : "Sign in"}
          </button>

          <div className="auth-switch">
            {isSignup ? "Already have an account?" : "New here?"}{" "}
            <button
              type="button"
              className="auth-link"
              onClick={() => switchMode(isSignup ? "login" : "signup")}
            >
              {isSignup ? "Sign in" : "Create one"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
