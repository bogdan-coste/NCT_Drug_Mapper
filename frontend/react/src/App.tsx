import { useEffect, useRef, useState } from "react";
import "./App.css";
import AuthScreen from "./AuthScreen";
import { pollRun, startRun, type RunState, type StageUpdate } from "./api";
import { clearSession, fetchMe, loadSession, logout, type Session } from "./auth";

type StageDef = { id: number; title: string; blurb: string };

const STAGES: StageDef[] = [
  {
    id: 1,
    title: "Fetch & resolve identity",
    blurb:
      "Trial records are pulled from ClinicalTrials.gov and the author term is matched against the intervention list.",
  },
  {
    id: 2,
    title: "Ground the definition",
    blurb:
      "NCIt and OLS are raced; the first usable ontology definition wins, otherwise the LLM writes one.",
  },
  {
    id: 3,
    title: "Suggest categories",
    blurb:
      "The definition is turned into a specific, broad and root therapeutic label.",
  },
  {
    id: 4,
    title: "Retrieve & map",
    blurb:
      "Labels become SapBERT queries over Qdrant, fused with RRF, then mapped to one ontology term.",
  },
];

/** Read a term + NCT IDs from the first data row of a CSV/TXT drop. */
function parseRows(text: string): { term: string; nctIds: string[] } | null {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (!lines.length) {
    return null;
  }

  const startIndex = /term|intervention|nct/i.test(lines[0]) ? 1 : 0;
  const row = lines[startIndex];
  if (!row) {
    return null;
  }

  const cells = row
    .split(/[,;\t|]/)
    .map((cell) => cell.trim())
    .filter(Boolean);

  const isNct = (value: string) => /^NCT\d+/i.test(value);
  const nctIds = cells.filter(isNct).map((value) => value.toUpperCase());
  const term = cells.find((value) => !isNct(value)) ?? "";

  return term ? { term, nctIds } : null;
}

const fmtSeconds = (ms: number) => `${(ms / 1000).toFixed(1)}s`;

function Icon({
  name,
}: {
  name: "spark" | "upload" | "info" | "file" | "bolt" | "check";
}) {
  const paths: Record<string, string> = {
    spark:
      "M12 2l1.9 5.1L19 9l-5.1 1.9L12 16l-1.9-5.1L5 9l5.1-1.9L12 2zM18 15l.9 2.4 2.4.9-2.4.9-.9 2.4-.9-2.4-2.4-.9 2.4-.9L18 15z",
    upload: "M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 17v2a1 1 0 001 1h14a1 1 0 001-1v-2",
    info: "M12 8h.01M11 12h1v5h1",
    file: "M7 3h7l4 4v13a1 1 0 01-1 1H7a1 1 0 01-1-1V4a1 1 0 011-1zm7 0v4h4",
    bolt: "M13 2L4.5 13.5H11l-1 8.5 8.5-11.5H12l1-8.5z",
    check: "M5 12.5l4.5 4.5L19 7",
  };

  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={name === "spark" || name === "bolt" ? 1.6 : 1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {name === "info" ? (
        <>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 8h.01M11 12h1v5h1" />
        </>
      ) : (
        <path d={paths[name]} />
      )}
    </svg>
  );
}

function StageDetail({
  stage,
  update,
}: {
  stage: number;
  update: StageUpdate;
}) {
  if (stage === 1) {
    const matches = update.identity_resolution ?? [];
    const resolved = update.resolved_search_terms ?? [];
    const warnings = update.context_warnings ?? [];

    return (
      <>
        {(update.nct_ids?.length ?? 0) > 0 && (
          <div className="rows">
            <div className="row">
              <span className="k">Trial records</span>
              <span className="v">{update.nct_ids!.join(", ")}</span>
            </div>
          </div>
        )}

        {matches.length > 0 && (
          <div className="matches">
            {matches.map((match, index) => (
              <div className="match" key={`${match.search_term}-${index}`}>
                <div className="match-head">
                  <span className="match-term">{match.search_term}</span>
                  <span className="match-arrow">→</span>
                  <span className="match-name">{match.matched_trial_name}</span>
                  {match.relationship && (
                    <span className="badge">
                      {match.relationship.replace(/_/g, " ")}
                    </span>
                  )}
                  {match.role && (
                    <span className="badge badge-role">
                      {match.role.replace(/_/g, " ")}
                    </span>
                  )}
                </div>
                {match.evidence && (
                  <p className="match-evidence">{match.evidence}</p>
                )}
              </div>
            ))}
          </div>
        )}

        {resolved.length > 0 && (
          <div className="chips">
            <span className="chips-label">Resolved search terms</span>
            {resolved.map((term) => (
              <span className="mini-chip" key={term}>
                {term}
              </span>
            ))}
          </div>
        )}

        {warnings.length > 0 && (
          <ul className="warnings">
            {warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        )}
      </>
    );
  }

  if (stage === 2) {
    const components = update.ncit_component_concepts ?? [];

    return (
      <>
        {update.definition && <p className="detail-text">{update.definition}</p>}

        <div className="chips">
          {update.definition_source && (
            <span className="mini-chip mini-chip-accent">
              {update.definition_source}
            </span>
          )}
          {update.ncit_code && (
            <span className="mini-chip">{update.ncit_code}</span>
          )}
          {update.ncit_preferred_name && (
            <span className="mini-chip">{update.ncit_preferred_name}</span>
          )}
          {update.ncit_search_term && (
            <span className="mini-chip">
              matched “{update.ncit_search_term}”
            </span>
          )}
        </div>

        {components.length > 0 && (
          <div className="components">
            {components.map((component) => (
              <div className="component" key={component.code}>
                <div className="component-head">
                  <span className="match-name">{component.preferred_name}</span>
                  <span className="badge">{component.code}</span>
                </div>
                <p className="match-evidence">{component.definition}</p>
              </div>
            ))}
          </div>
        )}

        {update.llm_definition &&
          update.llm_definition !== update.definition && (
            <details className="llm-def">
              <summary>LLM-written definition (superseded)</summary>
              <p className="detail-text">{update.llm_definition}</p>
            </details>
          )}
      </>
    );
  }

  if (stage === 3) {
    const labels = update.category_labels ?? {};

    return (
      <div className="label-rows">
        {Object.entries(labels).map(([key, value]) => (
          <div className="label-row" key={key}>
            <span className="k">{key.replace(/_/g, " ")}</span>
            <span className="label-chip">{String(value)}</span>
          </div>
        ))}
      </div>
    );
  }

  const mapping = update.ontology_mapping ?? {};
  const candidates = update.candidates ?? [];

  return (
    <>
      {Object.keys(mapping).length > 0 && (
        <div className="rows">
          {Object.entries(mapping).map(([key, value]) => (
            <div className="row" key={key}>
              <span className="k">{key.replace(/_/g, " ")}</span>
              <span className="v">{String(value)}</span>
            </div>
          ))}
        </div>
      )}

      {candidates.length > 0 && (
        <>
          <div className="chips-label">
            Candidates · {candidates.length} retrieved
          </div>
          <ul className="candidates">
            {candidates.slice(0, 8).map((candidate, index) => (
              <li className="candidate" key={`${candidate.name}-${index}`}>
                <span className="candidate-rank">{index + 1}</span>
                <span className="candidate-name">{candidate.name}</span>
                <span className="candidate-axis">
                  {candidate.app_drug_category ?? ""}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </>
  );
}

function ResultCard({ run }: { run: RunState }) {
  const update = run.update;
  const stage = update._stage ?? 0;
  const done = Boolean(update._done);
  const failed = Boolean(update._error);
  const elapsed = Date.now() - run.startedAt;

  /** Duration of one stage: its completion time minus the previous stage's. */
  const durationFor = (id: number): number | undefined => {
    const end = run.stageTimes[id];
    if (end === undefined) {
      return undefined;
    }
    const previous = id > 1 ? (run.stageTimes[id - 1] ?? 0) : 0;
    return end - previous;
  };

  const chipClass = failed ? "chip chip-bad" : done ? "chip chip-ok" : "chip";
  const chipText = failed
    ? "FAILED"
    : done
      ? "COMPLETE"
      : stage > 0
        ? `STAGE ${stage}/4`
        : "STARTING";

  return (
    <div className="card result-card">
      <div className="card-head">
        <h2 className="card-title">Therapeutic category</h2>
        <div className="head-meta">
          <span className="elapsed">{fmtSeconds(elapsed)}</span>
          <span className={chipClass}>{chipText}</span>
        </div>
      </div>

      {failed && <div className="error">{update._error}</div>}

      <ol className="timeline">
        {STAGES.map((definition) => {
          const completed = done ? definition.id <= 4 : definition.id <= stage;
          const active = !done && !failed && definition.id === stage + 1;
          const failedHere = failed && definition.id === stage + 1;
          const duration = durationFor(definition.id);

          const state = completed
            ? "done"
            : failedHere
              ? "failed"
              : active
                ? "active"
                : "pending";

          return (
            <li className={`tl-item ${state}`} key={definition.id}>
              <div className="tl-rail">
                <span className="tl-dot">
                  {completed ? <Icon name="check" /> : definition.id}
                </span>
                {definition.id < 4 && <span className="tl-line" />}
              </div>

              <div className="tl-body">
                <div className="tl-head">
                  <span className="tl-title">{definition.title}</span>
                  {duration !== undefined && (
                    <span
                      className="tl-time"
                      title={`${fmtSeconds(
                        run.stageTimes[definition.id],
                      )} after submission`}
                    >
                      {fmtSeconds(duration)}
                    </span>
                  )}
                </div>
                <p className="tl-blurb">{definition.blurb}</p>

                {completed && (
                  <div className="tl-detail">
                    <StageDetail stage={definition.id} update={update} />
                  </div>
                )}

                {active && <div className="tl-running">Running…</div>}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export default function App() {
  const [term, setTerm] = useState("Tofacitinib");
  const [nctInput, setNctInput] = useState("NCT04412252");
  const [runs, setRuns] = useState<RunState[]>([]);
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">(
    "idle",
  );
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const [session, setSession] = useState<Session | null>(() => loadSession());
  const [checking, setChecking] = useState(() => loadSession() !== null);

  // A token from localStorage may have expired; confirm it before showing the app.
  useEffect(() => {
    if (!session || !checking) {
      return;
    }

    let cancelled = false;

    fetchMe(session.token)
      .then(() => {
        if (!cancelled) {
          setChecking(false);
        }
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        clearSession();
        setSession(null);
        setChecking(false);
      });

    return () => {
      cancelled = true;
    };
  }, [session, checking]);

  function signOut() {
    if (session) {
      // Revoke server-side, but never block the UI on it: local state is
      // cleared whether or not the request reaches the API.
      void logout(session.token).catch(() => {
        /* session already expired or API unreachable */
      });
    }

    clearSession();
    setSession(null);
    setChecking(false);
    setRuns([]);
    setStatus("idle");
    setError(null);
  }

  const nctIds = nctInput
    .split(/[\s,;]+/)
    .map((value) => value.trim())
    .filter(Boolean);
  const canRun =
    term.trim().length > 0 && nctIds.length > 0 && status !== "running";

  async function handleFile(file: File) {
    try {
      const parsed = parseRows(await file.text());
      if (!parsed) {
        setError("Could not find a term and NCT ID in that file.");
        return;
      }
      setTerm(parsed.term);
      setNctInput(parsed.nctIds.join(", "));
      setFileName(file.name);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function run() {
    if (!canRun) {
      return;
    }

    setStatus("running");
    setError(null);
    setRuns([]);

    try {
      const runId = await startRun(term.trim(), nctIds);

      setRuns([{ runId, update: {}, startedAt: Date.now(), stageTimes: {} }]);

      await pollRun(runId, (update) => {
        setRuns((prev) =>
          prev.map((runState) => {
            if (runState.runId !== runId) {
              return runState;
            }

            const stageTimes = { ...runState.stageTimes };
            if (update._stage && stageTimes[update._stage] === undefined) {
              stageTimes[update._stage] = Date.now() - runState.startedAt;
            }

            return { ...runState, update, stageTimes };
          }),
        );
      });

      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  }

  if (!session) {
    return (
      <AuthScreen
        onAuthenticated={(next) => {
          setSession(next);
          setChecking(false);
        }}
      />
    );
  }

  if (checking) {
    return (
      <div className="auth-page">
        <div className="aurora" aria-hidden="true" />
        <div className="grid-overlay" aria-hidden="true" />
        <div className="auth-check">Checking your session…</div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="aurora" aria-hidden="true" />
      <div className="grid-overlay" aria-hidden="true" />

      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Icon name="spark" />
          </div>
          <div>
            <div className="brand-title">Therapeutic Mapping</div>
            <div className="brand-sub">AI Platform</div>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="status-pill">
            <span className="dot" />
            AI Platform <span className="sep">·</span> v1.2.0
          </div>

          <span className="user-chip">{session.username}</span>

          <button type="button" className="logout-btn" onClick={signOut}>
            Sign out
          </button>
        </div>
      </header>

      <main className="main">
        <section className="hero">
          <div className="hero-badge">
            <Icon name="spark" />
            Evidence-grounded ontology resolution
          </div>

          <h1>
            Map a trial intervention
            <br />
            to its <span className="grad">therapeutic category</span>
          </h1>

          <p className="hero-sub">
            Enter an intervention term with its ClinicalTrials.gov records. The
            pipeline resolves the identity, grounds a definition in NCIt/OLS,
            retrieves SapBERT candidates and returns the ontology mapping.
          </p>
        </section>

        <section className="layout">
          <div className="card">
            <div className="card-head">
              <h2 className="card-title">Intervention data</h2>
              <span className="chip">STEP 1</span>
            </div>

            <div className="inputs">
              <label className="field">
                <span className="field-label">Intervention term</span>
                <input
                  value={term}
                  onChange={(event) => setTerm(event.target.value)}
                  placeholder="e.g. Tofacitinib"
                  spellCheck={false}
                />
              </label>

              <label className="field">
                <span className="field-label">NCT IDs</span>
                <input
                  value={nctInput}
                  onChange={(event) => setNctInput(event.target.value)}
                  placeholder="NCT04412252, NCT…"
                  spellCheck={false}
                />
              </label>
            </div>

            <div
              className={`dropzone ${dragging ? "drag" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                const file = event.dataTransfer.files?.[0];
                if (file) {
                  void handleFile(file);
                }
              }}
              onClick={() => fileInput.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  fileInput.current?.click();
                }
              }}
            >
              <div className="drop-icon">
                <Icon name="upload" />
              </div>
              <div className="drop-title">
                {fileName ? fileName : "Drop your CSV here or browse"}
              </div>
              <div className="drop-sub">
                One row: term, then NCT IDs — or type them above
              </div>
              <input
                ref={fileInput}
                type="file"
                accept=".csv,.txt"
                hidden
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) {
                    void handleFile(file);
                  }
                }}
              />
            </div>

            {error && <div className="error">{error}</div>}

            <div className="card-foot">
              <span className="hint">
                <Icon name="file" />
                Accepted formats: .csv, .txt
              </span>

              <button
                type="button"
                className="run-btn"
                onClick={() => void run()}
                disabled={!canRun}
              >
                <Icon name="bolt" />
                {status === "running" ? "Running…" : "Run analysis"}
              </button>
            </div>
          </div>

          <div className="card">
            <div className="card-head">
              <h2 className="card-title">How it works</h2>
              <span className="info-dot">
                <Icon name="info" />
              </span>
            </div>

            <ol className="steps">
              {STAGES.map((definition) => (
                <li className="step" key={definition.id}>
                  <span className="step-num">{definition.id}</span>
                  <div>
                    <p className="step-title">{definition.title}</p>
                    <p className="step-sub">{definition.blurb}</p>
                  </div>
                </li>
              ))}
            </ol>

            <div className="tags">
              <span className="tag">Evidence-grounded</span>
              <span className="tag">NCIt + OLS</span>
              <span className="tag">SapBERT retrieval</span>
            </div>
          </div>
        </section>

        {runs.length > 0 && (
          <section className="results">
            {runs.map((runState) => (
              <ResultCard key={runState.runId} run={runState} />
            ))}
          </section>
        )}

        <footer className="footer">
          Therapeutic Mapping AI <span className="sep">·</span> evidence-grounded
          ontology resolution for clinical trials
        </footer>
      </main>
    </div>
  );
}
