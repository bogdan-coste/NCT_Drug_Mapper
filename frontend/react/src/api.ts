import { getToken } from "./auth";

export type Candidate = {
  name: string;
  documentation?: string;
  object_name?: string;
  app_drug_category?: string;
};

export type IdentityMatch = {
  search_term?: string;
  matched_trial_name?: string;
  relationship?: string;
  role?: string;
  component_type?: string;
  evidence?: string;
};

export type ComponentConcept = {
  search_term: string;
  code: string;
  preferred_name: string;
  definition: string;
};

export type StageUpdate = {
  _stage?: number;
  _done?: boolean;
  _error?: string;
  term?: string;
  nct_ids?: string[];
  identity_resolution?: IdentityMatch[];
  resolved_search_terms?: string[];
  context_warnings?: string[];
  definition?: string;
  definition_source?: string;
  ncit_code?: string | null;
  ncit_preferred_name?: string | null;
  ncit_search_term?: string | null;
  ncit_component_concepts?: ComponentConcept[];
  regimen_components?: string[];
  llm_definition?: string | null;
  category_labels?: Record<string, string>;
  candidates?: Candidate[];
  ontology_mapping?: Record<string, string>;
};

export type RunState = {
  runId: string;
  update: StageUpdate;
  /** Wall-clock timestamp when the run was submitted. */
  startedAt: number;
  /** Milliseconds from submission to the moment each stage completed. */
  stageTimes: Record<number, number>;
};

const BASE = import.meta.env.VITE_API_BASE ?? "";

/** Attach the bearer token to every authenticated request. */
function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function startRun(
  term: string,
  nctIds: string[],
): Promise<string> {
  const response = await fetch(`${BASE}/start_pipeline`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ term, nct_ids: nctIds }),
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  const payload = (await response.json()) as { run_id: string };
  return payload.run_id;
}

export async function getResult(runId: string): Promise<StageUpdate> {
  const response = await fetch(`${BASE}/result/${runId}`, {
    headers: authHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Result ${runId}: ${response.status} ${response.statusText}`);
  }

  return (await response.json()) as StageUpdate;
}

export const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Poll one run until the backend reports `_done` or `_error`.
 * `onUpdate` fires on every poll so the UI can stream stages.
 */
export async function pollRun(
  runId: string,
  onUpdate: (update: StageUpdate) => void,
  intervalMs = 800,
): Promise<StageUpdate> {
  for (;;) {
    const update = await getResult(runId);
    onUpdate(update);

    if (update._done || update._error) {
      return update;
    }

    await sleep(intervalMs);
  }
}
