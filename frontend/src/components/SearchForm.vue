<script setup>
import { ref, computed } from 'vue'
import ProgressSteps from './ProgressSteps.vue'

const props = defineProps({
  loading: { type: Boolean, default: false },
  steps: { type: Array, default: () => [] },
  currentStep: { type: Number, default: -1 },
  liveMessage: { type: String, default: '' },
})

const emit = defineEmits(['submit'])

const term = ref('')
const nctRaw = ref('')

const canSubmit = computed(() => term.value.trim() && nctRaw.value.trim())

function submit() {
  if (!canSubmit.value || props.loading) return
  const nctIds = nctRaw.value
    .split(',')
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean)
  emit('submit', { term: term.value.trim(), nctIds })
}
</script>

<template>
  <div class="card">
    <div class="field">
      <label for="term">Intervention term</label>
      <input
        id="term"
        type="text"
        v-model="term"
        placeholder="e.g. EntityA 40 mg IV"
        :disabled="loading"
        @keyup.enter="submit"
      />
    </div>
    <div class="field">
      <label for="nct">NCT IDs</label>
      <input
        id="nct"
        type="text"
        v-model="nctRaw"
        placeholder="e.g. NCT01234567, NCT09876543"
        :disabled="loading"
        @keyup.enter="submit"
      />
      <p class="hint">Comma-separated. At least one required.</p>
    </div>
    <button class="btn" @click="submit" :disabled="loading || !canSubmit">
      <span v-if="loading" class="spinner"></span>
      <span>{{ loading ? 'Running...' : 'Run pipeline' }}</span>
    </button>

    <ProgressSteps
      v-if="loading"
      :steps="steps"
      :current-step="currentStep"
      :live-message="liveMessage"
    />
  </div>
</template>
