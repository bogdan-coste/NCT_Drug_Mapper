<script setup>
const props = defineProps({
  candidates: { type: Array, default: () => [] },
  selectedTerm: { type: String, default: '' },
})

function isSelected(c) {
  return Boolean(props.selectedTerm) && c.name === props.selectedTerm
}

function preview(doc) {
  if (!doc) return doc
  return doc.length > 120 ? doc.slice(0, 120) + '…' : doc
}
</script>

<template>
  <ul class="candidate-list">
    <li
      v-for="(c, i) in candidates"
      :key="i"
      :style="isSelected(c) ? 'background:#f0fdf4;border-radius:4px;padding:0.4rem 0.5rem' : 'padding:0.4rem 0.5rem'"
    >
      <div style="display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap">
        <span class="rank">{{ i + 1 }}.</span>
        <span :style="isSelected(c) ? 'font-weight:600' : ''">{{ c.name }}</span>
        <span
          v-if="isSelected(c)"
          class="exact-badge"
          style="background: #16a34a; color: #fff"
        >&#10003; selected</span>
        <span v-else-if="c.exact_match" class="exact-badge">exact match</span>
        <span class="rrf-score" v-if="c.rrf_score != null">RRF {{ c.rrf_score.toFixed(5) }}</span>
      </div>

      <!-- documentation preview -->
      <div v-if="c.documentation" class="cand-doc" :title="c.documentation">
        {{ preview(c.documentation) }}
      </div>

      <!-- retrieval provenance chips -->
      <div v-if="c.retrieval_provenance && c.retrieval_provenance.length" class="cand-meta">
        <span v-for="(p, pi) in c.retrieval_provenance" :key="pi" class="prov-chip">
          {{ p.query }} (rank {{ p.rank }})
        </span>
      </div>

      <!-- ontology context collapsible -->
      <details
        class="detail-block"
        v-if="(c.parents && c.parents.length) || (c.children && c.children.length)"
      >
        <summary>Ontology context</summary>
        <div class="detail-inner">
          <div v-if="c.parents && c.parents.length">
            <div class="node-rel-label">Parents</div>
            <div v-for="(p, pi) in c.parents" :key="pi" class="node-rel-item">{{ p.name }}</div>
          </div>
          <div v-if="c.children && c.children.length" style="margin-top: 0.5rem">
            <div class="node-rel-label">Children</div>
            <div v-for="(ch, ci) in c.children" :key="ci" class="node-rel-item">{{ ch.name }}</div>
          </div>
        </div>
      </details>
    </li>
  </ul>
</template>
