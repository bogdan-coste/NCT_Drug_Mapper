<script setup>
import CandidateList from './CandidateList.vue'

defineProps({
  result: { type: Object, default: null },
  dcMapping: { type: Object, default: () => ({}) },
  dcCompatClass: { type: String, default: '' },
  selectedCandidate: { type: Object, default: null },
})
</script>

<template>
  <div class="card" v-if="result && result.identity_resolution">
    <div class="step-header">
      <span class="step-badge">Step 1</span>
      <h2>Name resolution</h2>
      <span class="source-badge source-llm">LLM</span>
    </div>
    <p class="narrative">
      The input term <strong>"{{ result.term }}"</strong> was looked up against
      {{ result.nct_ids.length }} NCT record(s)
      (<code>{{ result.nct_ids.join(", ") }}</code>).
      The LLM classified it as
      <strong>{{ result.identity_resolution.resolution_kind }}</strong>.
    </p>
    <div v-if="result.identity_resolution.matches && result.identity_resolution.matches.length">
      <p class="section-divider">LLM identity matches</p>
      <table class="identity-table">
        <thead>
          <tr>
            <th>Search term</th>
            <th>Matched trial name</th>
            <th>Role</th>
            <th>Relationship</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="m in result.identity_resolution.matches" :key="m.search_term + m.role">
            <td><strong>{{ m.search_term }}</strong></td>
            <td style="color: #6b7280; font-size: 0.8rem">{{ m.matched_trial_name }}</td>
            <td><span class="role-badge">{{ m.role }}</span></td>
            <td style="font-size: 0.8rem">{{ m.relationship }}</td>
            <td style="font-size: 0.78rem; color: #6b7280; font-style: italic">{{ m.evidence }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="card" v-if="result && result.definition">
    <div class="step-header">
      <span class="step-badge">Step 2</span>
      <h2>Composition-centered definition</h2>
      <span class="source-badge source-llm">LLM</span>
    </div>
    <p class="narrative">
      The LLM generated a definition focused on the molecular, structural,
      biochemical, compositional, or origin-based identity of the intervention
      using ClinicalTrials.gov evidence.
    </p>
    <p class="definition-text">{{ result.definition }}</p>
  </div>

  <div class="card" v-if="result && result.retrieval_queries">
    <div class="step-header">
      <span class="step-badge">Step 3</span>
      <h2>Drug-class retrieval</h2>
      <span class="source-badge source-llm">LLM + SapBERT</span>
    </div>
    <p class="narrative">
      The LLM converted the definition into retrieval queries, which were embedded
      via SapBERT and used to search the drug-class ontology axis in Qdrant.
    </p>
    <p class="section-divider">Retrieval queries</p>
    <div class="category-ladder">
      <div class="cat-level" v-if="result.retrieval_queries.primary_query">
        <span class="cat-label">Primary</span>
        <span class="cat-value">{{ result.retrieval_queries.primary_query }}</span>
      </div>
      <template v-if="result.retrieval_queries.alternative_queries && result.retrieval_queries.alternative_queries.length">
        <div class="cat-arrow">&#9650;</div>
        <div class="cat-level" v-for="(q, i) in result.retrieval_queries.alternative_queries" :key="i">
          <span class="cat-label">Alt {{ i + 1 }}</span>
          <span class="cat-value">{{ q }}</span>
        </div>
      </template>
    </div>

    <template v-if="result.candidates && result.candidates.length">
      <p class="section-divider" style="margin-top: 1.25rem">Qdrant candidates retrieved ({{ result.candidates.length }})</p>
      <CandidateList :candidates="result.candidates" :selected-term="dcMapping.selected_term" />
    </template>
    <p v-else-if="!result.drug_class_mapping" style="font-size: 0.8rem; color: #9ca3af; margin-top: 1rem">Retrieving candidates...</p>
  </div>

  <div class="card" v-if="result && result.drug_class_mapping">
    <div class="step-header">
      <span class="step-badge">Step 4</span>
      <h2>Drug Class Result</h2>
      <span class="source-badge source-llm">LLM</span>
    </div>
    <div class="mapping-result-box">
      <div class="mapping-result-label">Mapped drug class</div>
      <div class="mapping-result-term">{{ dcMapping.selected_term || "NONE_OF_THE_ABOVE" }}</div>
      <div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; margin-top: 0.5rem">
        <span class="compat-badge" :class="dcCompatClass">{{ dcMapping.compatibility || "none" }}</span>
        <span style="font-size: 0.8rem; color: #6b7280">match type: <strong>{{ dcMapping.match_type || "—" }}</strong></span>
      </div>
      <p class="mapping-reason" style="margin-top: 0.75rem">{{ dcMapping.reason || "—" }}</p>
    </div>

    <p class="section-divider" style="margin-top: 1.25rem">Mapping detail</p>
    <div class="meta-grid">
      <span class="meta-key">Selected term</span>
      <span class="meta-val">{{ dcMapping.selected_term || "NONE_OF_THE_ABOVE" }}</span>
      <span class="meta-key">Compatibility</span>
      <span class="meta-val"><span class="compat-badge" :class="dcCompatClass">{{ dcMapping.compatibility || "—" }}</span></span>
      <span class="meta-key">Match type</span>
      <span class="meta-val"><strong>{{ dcMapping.match_type || "—" }}</strong></span>
      <span class="meta-key">Reason</span>
      <span class="meta-val" style="font-style: italic">{{ dcMapping.reason || "—" }}</span>
    </div>

    <div class="node-detail" v-if="selectedCandidate && dcMapping.selected_term !== 'NONE_OF_THE_ABOVE'">
      <div class="node-detail-title">Ontology node detail</div>
      <div class="node-doc" v-if="selectedCandidate.documentation">{{ selectedCandidate.documentation }}</div>
      <div class="node-id">
        <span v-if="selectedCandidate.durable_id">ID: <code>{{ selectedCandidate.durable_id }}</code></span>
        <span v-if="selectedCandidate.object_name" style="margin-left: 0.75rem">Object name: <code>{{ selectedCandidate.object_name }}</code></span>
      </div>
      <div class="node-relations" v-if="(selectedCandidate.parents && selectedCandidate.parents.length) || (selectedCandidate.children && selectedCandidate.children.length)">
        <div class="node-rel-group" v-if="selectedCandidate.parents && selectedCandidate.parents.length">
          <div class="node-rel-label">Parents</div>
          <div v-for="(p, pi) in selectedCandidate.parents" :key="pi" class="node-rel-item">{{ p.name }}</div>
        </div>
        <div class="node-rel-group" v-if="selectedCandidate.children && selectedCandidate.children.length">
          <div class="node-rel-label">Children</div>
          <div v-for="(ch, ci) in selectedCandidate.children" :key="ci" class="node-rel-item">{{ ch.name }}</div>
        </div>
      </div>
    </div>
  </div>
</template>
