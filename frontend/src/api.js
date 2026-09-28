// HTTP helpers for the FastAPI pipeline endpoints.

// Kicks off the pipeline for the given mode and returns the run id.
//   mode: 'tc' (therapeutic category) | 'dc' (drug class)
export async function startPipeline(mode, term, nctIds) {
  const endpoint = mode === 'dc' ? '/start_dc_pipeline' : '/start_pipeline'
  const resp = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ term, nct_ids: nctIds }),
  })
  if (!resp.ok) throw new Error(`Server error ${resp.status}`)
  return (await resp.json()).run_id
}

// Fetches the current (possibly partial) result snapshot for a run.
export async function fetchResult(runId) {
  const resp = await fetch(`/result/${runId}`)
  if (!resp.ok) throw new Error(`Poll failed: HTTP ${resp.status}`)
  return resp.json()
}
