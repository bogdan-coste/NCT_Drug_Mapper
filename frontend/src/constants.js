// Shared pipeline constants and small display helpers.

export const TC_STEPS = [
  'Fetching NCT records',
  'Looking up NCIt / generating definition',
  'Suggesting therapeutic categories',
  'Retrieving candidates & mapping ontology term',
]

export const DC_STEPS = [
  'Fetching NCT records',
  'Generating composition-centered definition',
  'Suggesting retrieval queries',
  'Retrieving candidates & mapping drug class',
]

// Maps the server's _stage number to a 0-based progress-step index.
export const STAGE_TO_STEP = { 1: 0, 2: 1, 3: 2, 4: 3 }

export const SOURCE_META = {
  ncit: { label: 'NCIt definition', cls: 'source-ncit' },
  ncit_class_exemplar: { label: 'NCIt (class exemplar)', cls: 'source-ncit-exemplar' },
  ncit_components_plus_llm: { label: 'NCIt components + LLM', cls: 'source-ncit-llm' },
  llm: { label: 'LLM generated', cls: 'source-llm' },
}

// Buckets a free-text compatibility string into a CSS class.
export function compatClassFor(compatibility) {
  const c = compatibility ?? ''
  if (c.startsWith('directly')) return 'compat-directly'
  if (c.startsWith('broader')) return 'compat-broader'
  if (c.startsWith('related')) return 'compat-related'
  return 'compat-none'
}
