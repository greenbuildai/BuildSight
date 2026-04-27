import assert from 'node:assert/strict'
import {
  buildPeakRiskMoment,
  detectionsToHeatmapPoints,
  mergePeakRiskMoments,
  type DetectionLike,
} from './src/lib/detectionIntelligence.ts'

function worker(
  box: [number, number, number, number],
  overrides: Partial<DetectionLike> = {},
): DetectionLike {
  return {
    class: 'worker',
    confidence: 0.9,
    box,
    has_helmet: true,
    has_vest: true,
    ...overrides,
  }
}

const heatmapPoints = detectionsToHeatmapPoints([
  worker([64, 36, 128, 144]),
  worker([320, 180, 420, 320], { has_vest: false }),
], 640, 360, 1234)

assert.equal(heatmapPoints.length, 2, 'expected one heatmap point per worker/person detection')
assert.deepEqual(heatmapPoints[0], {
  x: 0.15,
  y: 0.25,
  time: 1234,
  type: 'worker',
  value: 0.28,
  riskLevel: 'LOW',
})
assert.deepEqual(heatmapPoints[1], {
  x: 0.578125,
  y: 0.6944444444444444,
  time: 1234,
  type: 'violation',
  value: 0.9,
  riskLevel: 'HIGH',
})

assert.equal(
  buildPeakRiskMoment([worker([0, 0, 40, 80])], 8.2),
  null,
  'compliant detections should not generate a risk moment',
)

const clusteredRisk = buildPeakRiskMoment([
  worker([100, 100, 180, 220], { has_helmet: false }),
  worker([150, 110, 220, 230], { has_vest: false }),
  worker([170, 120, 250, 240], { has_helmet: false, has_vest: false }),
], 12.4)

assert.ok(clusteredRisk, 'multiple nearby PPE violations should generate a risk moment')
assert.equal(clusteredRisk?.type, 'PPE_VIOLATION')
assert.ok(clusteredRisk!.score > 3, 'density should boost the raw violation count')

const merged = mergePeakRiskMoments(
  [
    { time: 5, score: 2.2, type: 'PPE_VIOLATION' },
    { time: 15, score: 4.1, type: 'PPE_VIOLATION' },
  ],
  { time: 5.8, score: 3.7, type: 'PPE_VIOLATION' },
)

assert.deepEqual(merged, [
  { time: 15, score: 4.1, type: 'PPE_VIOLATION' },
  { time: 5.8, score: 3.7, type: 'PPE_VIOLATION' },
])

console.log('detection intelligence checks passed')
