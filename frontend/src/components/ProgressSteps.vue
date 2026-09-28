<script setup>
const props = defineProps({
  steps: { type: Array, default: () => [] },
  currentStep: { type: Number, default: -1 },
  liveMessage: { type: String, default: '' },
})

function progClass(i) {
  if (props.currentStep > i) return 'done'
  if (props.currentStep === i) return 'active'
  return ''
}
</script>

<template>
  <div class="progress-steps">
    <div v-for="(step, i) in steps" :key="i" class="prog-step" :class="progClass(i)">
      <div class="prog-icon">
        <span v-if="currentStep > i">&#10003;</span>
        <div v-else-if="currentStep === i" class="prog-spinner"></div>
        <span v-else style="opacity: 0.35">&#9679;</span>
      </div>
      <span>{{ currentStep === i ? liveMessage || step : step }}</span>
    </div>
  </div>
</template>
