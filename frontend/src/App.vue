<script setup>
import { ref, computed, onUnmounted } from 'vue'
import SearchForm from './components/SearchForm.vue'
import TcResultPanel from './components/TcResultPanel.vue'
import DcResultPanel from './components/DcResultPanel.vue'
import { TC_STEPS, DC_STEPS, STAGE_TO_STEP, SOURCE_META, compatClassFor } from './constants.js'
import { startPipeline, fetchResult } from './api.js'

const loading = ref(false)
const error = ref(null)
const result = ref(null)
const currentStep = ref(-1)
const liveMessage = ref('')
const mode = ref('tc')

const STEPS = computed(() => (mode.value === 'tc' ? TC_STEPS : DC_STEPS))

const mapping = computed(() => result.value?.ontology_mapping ?? {})
const dcMapping = computed(() => result.value?.drug_class_mapping ?? {})
const compatClass = computed(() => compatClassFor(mapping.value.compatibility))
const dcCompatClass = computed(() => compatClassFor(dcMapping.value.compatibility))

const sourceLabel = computed(
  () => SOURCE_META[result.value?.definition_source]?.label ?? 'Unknown source',
)
const sourceClass = computed(
  () => SOURCE_META[result.value?.definition_source]?.cls ?? 'source-unknown',
)

const selectedTcCandidate = computed(
  () => result.value?.candidates?.find((c) => c.name === mapping.value.selected_term) ?? null,
)
const selectedDcCandidate = computed(
  () => result.value?.candidates?.find((c) => c.name === dcMapping.value.selected_term) ?? null,
)

let watchdogTimer = null
let pollTimer = null

function stopAll() {
  if (watchdogTimer) {
    clearTimeout(watchdogTimer)
    watchdogTimer = null
  }
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

function switchMode(m, event) {
  if (loading.value) return
  // Ripple effect on the button
  if (event) {
    const btn = event.currentTarget
    const ripple = document.createElement('span')
    ripple.className = 'ripple'
    const rect = btn.getBoundingClientRect()
    ripple.style.top = event.clientY - rect.top + 'px'
    ripple.style.left = event.clientX - rect.left + 'px'
    btn.appendChild(ripple)
    ripple.addEventListener('animationend', () => ripple.remove())
  }
  mode.value = m
  result.value = null
  error.value = null
  currentStep.value = -1
}

async function handleSubmit({ term, nctIds }) {
  if (loading.value) return

  stopAll()
  loading.value = true
  error.value = null
  result.value = null // clear old results
  currentStep.value = 0
  liveMessage.value = ''

  let runId
  try {
    runId = await startPipeline(mode.value, term, nctIds)
  } catch (e) {
    error.value = e.message ?? 'Network error.'
    loading.value = false
    return
  }

  // Poll every 2 s — partial results are merged into result.value live
  pollTimer = setInterval(async () => {
    let payload
    try {
      payload = await fetchResult(runId)
    } catch (e) {
      stopAll()
      error.value = e.message ?? String(e)
      loading.value = false
      return
    }

    if (payload._error) {
      stopAll()
      error.value = payload._error
      loading.value = false
      currentStep.value = -1
      return
    }

    // Advance the progress indicator to reflect the latest completed stage
    if (payload._stage != null) {
      const step = STAGE_TO_STEP[payload._stage] ?? currentStep.value
      if (step > currentStep.value) currentStep.value = step
    }

    // Merge partial payload into result ref so Vue re-renders incrementally
    result.value = { ...(result.value ?? {}), ...payload }

    // Pipeline finished
    if (payload._done) {
      stopAll()
      loading.value = false
      currentStep.value = -1
    }
  }, 2000)

  // Safety watchdog — 5 minutes
  watchdogTimer = setTimeout(
    () => {
      if (loading.value) {
        stopAll()
        error.value = 'Pipeline timed out after 5 minutes. Check server logs.'
        loading.value = false
        currentStep.value = -1
      }
    },
    5 * 60 * 1000,
  )
}

onUnmounted(stopAll)
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-logo">
      <h1>Ontology<br />Mapping</h1>
      <p>NCT evidence pipeline</p>
    </div>
    <div class="tab-bar">
      <span class="tab-bar-label">Pipeline</span>
      <button
        class="tab-btn"
        :class="{ active: mode === 'tc' }"
        @click="switchMode('tc', $event)"
        :disabled="loading"
      >
        <span class="tab-icon">&#9703;</span> Therapeutic Category
      </button>
      <button
        class="tab-btn"
        :class="{ active: mode === 'dc' }"
        @click="switchMode('dc', $event)"
        :disabled="loading"
      >
        <span class="tab-icon">&#10022;</span> Drug Class
      </button>
    </div>
  </aside>

  <div class="main-content">
    <div class="container">
      <SearchForm
        :loading="loading"
        :steps="STEPS"
        :current-step="currentStep"
        :live-message="liveMessage"
        @submit="handleSubmit"
      />

      <div v-if="error" class="error-box">{{ error }}</div>

      <div
        v-if="result && result.context_warnings && result.context_warnings.length"
        class="warn-box"
      >
        <strong>Warnings</strong>
        <ul>
          <li v-for="w in result.context_warnings" :key="w">{{ w }}</li>
        </ul>
      </div>

      <div class="mode-info" v-if="!result && !loading">
        <!-- Therapeutic Category panel -->
        <div v-if="mode === 'tc'" class="mode-panel tc" :key="'tc'">
          <div class="mode-panel-icon">&#9703;</div>
          <div class="mode-panel-title">Therapeutic Category Mapping</div>
          <div class="mode-panel-desc">
            Maps an intervention term to its functional therapeutic or pharmacological category
            in an ontology index. The pipeline first looks up the term in NCI Thesaurus; if no
            entry is found it generates a definition from ClinicalTrials.gov evidence, then uses
            SapBERT embeddings to retrieve and rank the closest ontology nodes.
          </div>
          <div class="mode-steps">
            <div class="mode-step">
              <div class="mode-step-num">1</div>
              <div class="mode-step-text">
                <strong>Name resolution</strong>
                <span>The LLM reads NCT records and resolves the author term to a canonical searchable name, classifying it as single product, regimen, composite, or class request.</span>
              </div>
            </div>
            <div class="mode-step">
              <div class="mode-step-num">2</div>
              <div class="mode-step-text">
                <strong>Definition lookup</strong>
                <span>NCIt is searched first. If a match is found the official definition is used directly. Otherwise the LLM generates a definition from trial evidence.</span>
              </div>
            </div>
            <div class="mode-step">
              <div class="mode-step-num">3</div>
              <div class="mode-step-text">
                <strong>Category suggestion</strong>
                <span>The LLM proposes a three-level functional hierarchy (specific → broad → root). Each level becomes a SapBERT query into the Qdrant ontology index.</span>
              </div>
            </div>
            <div class="mode-step">
              <div class="mode-step-num">4</div>
              <div class="mode-step-text">
                <strong>Candidate retrieval &amp; mapping</strong>
                <span>Up to 20 candidates are retrieved via RRF fusion. The LLM selects the best match and assigns a compatibility level: directly, broader, or related compatible.</span>
              </div>
            </div>
          </div>
        </div>

        <div v-if="mode === 'dc'" class="mode-panel dc" :key="'dc'">
          <div class="mode-panel-icon">&#10022;</div>
          <div class="mode-panel-title">Drug Class Mapping</div>
          <div class="mode-panel-desc">
            Maps an intervention term to its structural or biochemical drug class in the ontology.
            Unlike therapeutic category mapping, this pipeline focuses on molecular composition,
            mechanism, and structural identity rather than clinical indication or functional role.
          </div>
          <div class="mode-steps">
            <div class="mode-step">
              <div class="mode-step-num">1</div>
              <div class="mode-step-text">
                <strong>Name resolution</strong>
                <span>The LLM resolves the author intervention name against NCT records, identifying whether it is a single entity, regimen, or composite product.</span>
              </div>
            </div>
            <div class="mode-step">
              <div class="mode-step-num">2</div>
              <div class="mode-step-text">
                <strong>Composition-centered definition</strong>
                <span>The LLM generates a definition focused strictly on molecular structure, biochemical composition, and biological origin — not on indication or modality.</span>
              </div>
            </div>
            <div class="mode-step">
              <div class="mode-step-num">3</div>
              <div class="mode-step-text">
                <strong>Retrieval query generation</strong>
                <span>The LLM converts the definition into one primary and up to two alternative drug-class retrieval queries for SapBERT embedding search.</span>
              </div>
            </div>
            <div class="mode-step">
              <div class="mode-step-num">4</div>
              <div class="mode-step-text">
                <strong>Candidate retrieval &amp; mapping</strong>
                <span>Candidates are retrieved from the drug-class axis of the Qdrant index and ranked by RRF. The LLM selects the most compatible drug-class node.</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- TC summary card (shown when pipeline is complete) -->
      <div class="summary-card" v-if="mode === 'tc' && result && result._done">
        <div class="summary-term">{{ result.term }}</div>
        <div class="summary-arrow">&#8595; normalized &amp; resolved</div>
        <div class="summary-mapped">{{ mapping.selected_term || 'NONE_OF_THE_ABOVE' }}</div>
        <div class="summary-meta">
          <span class="summary-badge">{{ result.definition_source || 'llm' }}</span>
          <span class="compat-badge" :class="compatClass" style="font-size: 0.68rem">{{ mapping.compatibility || 'none' }}</span>
          <span class="summary-badge">{{ mapping.match_type || '—' }}</span>
        </div>
      </div>

      <!-- DC summary card -->
      <div class="summary-card" v-if="mode === 'dc' && result && result._done">
        <div class="summary-term">{{ result.term }}</div>
        <div class="summary-arrow">&#8595; classified</div>
        <div class="summary-mapped">{{ dcMapping.selected_term || 'NONE_OF_THE_ABOVE' }}</div>
        <div class="summary-meta">
          <span class="compat-badge" :class="dcCompatClass" style="font-size: 0.68rem">{{ dcMapping.compatibility || 'none' }}</span>
          <span class="summary-badge">{{ dcMapping.match_type || '—' }}</span>
        </div>
      </div>

      <!-- Therapeutic Category step cards -->
      <TcResultPanel
        v-if="mode === 'tc'"
        :result="result"
        :mapping="mapping"
        :compat-class="compatClass"
        :source-label="sourceLabel"
        :source-class="sourceClass"
        :selected-candidate="selectedTcCandidate"
      />

      <!-- Drug Class step cards -->
      <DcResultPanel
        v-if="mode === 'dc'"
        :result="result"
        :dc-mapping="dcMapping"
        :dc-compat-class="dcCompatClass"
        :selected-candidate="selectedDcCandidate"
      />
    </div>
  </div>
</template>
