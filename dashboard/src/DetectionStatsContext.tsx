import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'
import type { AlertItem } from './components/AlertLog'
import { useDetectionStore, type WorkerPosition, type ZoneViolation } from './store/detectionStore'

/* ═══════════════════════════════════════════════════════════════════════════════
   BuildSight — Detection Stats Context
   Shared real-time detection telemetry consumed by compliance metric cards.
   DetectionPanel writes stats here; AppInner reads them to drive live metrics.
   ═══════════════════════════════════════════════════════════════════════════════ */

export interface DetectionStats {
  /** Total workers detected in the current/last inference cycle */
  totalWorkers: number
  /** Workers wearing helmets */
  helmetsDetected: number
  /** Workers wearing safety vests */
  vestsDetected: number
  /** Number of proximity / restricted-zone violations */
  proximityViolations: number
  /** Mean confidence across all detections (0-1) */
  avgConfidence: number
  /** Inference latency in ms */
  elapsedMs: number
  /** Total frames scanned since session start */
  framesScanned: number
  /** Whether detection is currently running */
  isRunning: boolean
  /** Model name string from the backend */
  modelName: string
}

const EMPTY: DetectionStats = {
  totalWorkers: 0,
  helmetsDetected: 0,
  vestsDetected: 0,
  proximityViolations: 0,
  avgConfidence: 0,
  elapsedMs: 0,
  framesScanned: 0,
  isRunning: false,
  modelName: '',
}

interface DetectionStatsCtx {
  stats: DetectionStats
  liveAlerts: AlertItem[]
  /** Called by DetectionPanel on every inference result */
  pushDetections: (
    detections: Array<{ class: string; confidence: number }>, 
    elapsedMs: number,
    validWorkers?: WorkerPosition[],
    violations?: ZoneViolation[],
    fps?: number,
    sceneCondition?: string
  ) => void
  /** Mark detection as running or stopped */
  setRunning: (running: boolean) => void
  /** Set model name */
  setModelName: (name: string) => void
  /** Reset all stats (e.g. when switching modes) */
  resetStats: () => void
}

const Ctx = createContext<DetectionStatsCtx | null>(null)

const MAX_ALERTS = 6
const ALERT_DEDUPE_MS = 8000

function formatAlertTime(date: Date) {
  const hours = String(date.getHours()).padStart(2, '0')
  const mins = String(date.getMinutes()).padStart(2, '0')
  const secs = String(date.getSeconds()).padStart(2, '0')
  return `${hours}:${mins}:${secs}`
}

function buildLiveAlerts(
  detections: Array<{ class: string; confidence: number }>,
  effectiveWorkers: number,
  helmets: number,
  vests: number,
): AlertItem[] {
  const now = new Date()
  const avgConfidence = detections.length > 0
    ? detections.reduce((sum, det) => sum + det.confidence, 0) / detections.length
    : 0
  const alerts: AlertItem[] = []

  if (effectiveWorkers > 0 && helmets < effectiveWorkers) {
    alerts.push({
      id: `DL-${now.getTime()}-H`,
      time: formatAlertTime(now),
      camera: 'DETECTION / ACTIVE WORKSPACE',
      severity: 'critical',
      title: 'Hardhat non-compliance detected',
      detail: `${effectiveWorkers - helmets} worker${effectiveWorkers - helmets !== 1 ? 's' : ''} missing helmet protection in the current scan.`,
    })
  }

  if (effectiveWorkers > 0 && vests < effectiveWorkers) {
    alerts.push({
      id: `DL-${now.getTime()}-V`,
      time: formatAlertTime(now),
      camera: 'DETECTION / ACTIVE WORKSPACE',
      severity: 'warning',
      title: 'High-vis vest non-compliance detected',
      detail: `${effectiveWorkers - vests} worker${effectiveWorkers - vests !== 1 ? 's' : ''} missing high-visibility vest coverage in the current scan.`,
    })
  }

  if (detections.length > 0 && avgConfidence < 0.55) {
    alerts.push({
      id: `DL-${now.getTime()}-C`,
      time: formatAlertTime(now),
      camera: 'DETECTION / ACTIVE WORKSPACE',
      severity: 'info',
      title: 'Detection confidence degraded',
      detail: `Average confidence dropped to ${(avgConfidence * 100).toFixed(0)}%. Manual operator review is recommended.`,
    })
  }

  return alerts
}

function appendDedupedAlerts(prev: AlertItem[], nextAlerts: AlertItem[]) {
  const now = Date.now()

  const deduped = nextAlerts.filter((candidate) => {
    const match = prev.find((existing) => {
      const existingStamp = Number(existing.id.split('-')[1] ?? 0)
      return existing.title === candidate.title
        && existing.detail === candidate.detail
        && existing.camera === candidate.camera
        && now - existingStamp < ALERT_DEDUPE_MS
    })

    return !match
  })

  return [...deduped, ...prev].slice(0, MAX_ALERTS)
}

export function useDetectionStats() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useDetectionStats must be used inside DetectionStatsProvider')
  return ctx
}

export function DetectionStatsProvider({ children }: { children: ReactNode }) {
  const [stats, setStats] = useState<DetectionStats>({ ...EMPTY })
  const [liveAlerts, setLiveAlerts] = useState<AlertItem[]>([])

  // Bridge to global Zustand store so GeoAI live-mode check picks up
  // video-upload inference state without needing a backend WS message.
  const storeSetWorkers    = useDetectionStore(s => s.setWorkerPositions)
  const storeSetViolations = useDetectionStore(s => s.setViolations)
  const storeSetCount      = useDetectionStore(s => s.setWorkerCount)
  const storeSetFPS        = useDetectionStore(s => s.setFPS)
  const storeSetLatency    = useDetectionStore(s => s.setLatencyMs)
  const storeSetScene      = useDetectionStore(s => s.setSceneCondition)
  const storeSetRunning    = useDetectionStore(s => s.setRunning)

  const pushDetections = useCallback(
    (
      detections: Array<{ class: string; confidence: number }>, 
      elapsedMs: number,
      validWorkers?: WorkerPosition[],
      violations?: ZoneViolation[],
      fps?: number,
      sceneCondition?: string
    ) => {
      let workers = 0, helmets = 0, vests = 0

      for (const d of detections) {
        const cls = d.class.toLowerCase()
        if (cls === 'worker' || cls === 'person') workers++
        if (cls === 'helmet' || cls === 'hardhat') helmets++
        if (cls === 'safety_vest' || cls === 'safety-vest' || cls === 'vest') vests++
      }

      const effectiveWorkers = workers > 0 ? workers : Math.max(helmets, vests)
      setLiveAlerts((prevAlerts) => appendDedupedAlerts(
        prevAlerts,
        buildLiveAlerts(detections, effectiveWorkers, helmets, vests),
      ))

      setStats(prev => {
        // Count class occurrences
        let confSum = 0

        for (const d of detections) {
          confSum += d.confidence
        }

        return {
          ...prev,
          totalWorkers: effectiveWorkers,
          helmetsDetected: helmets,
          vestsDetected: vests,
          proximityViolations: violations?.length ?? prev.proximityViolations,
          avgConfidence: detections.length > 0 ? confSum / detections.length : prev.avgConfidence,
          elapsedMs,
          framesScanned: prev.framesScanned + 1,
          isRunning: true,
        }
      })
      // Bridge to global store for GeoAI dashboard
      if (validWorkers) storeSetWorkers(validWorkers)
      if (violations)   storeSetViolations(violations)
      storeSetCount(effectiveWorkers)
      if (fps !== undefined) storeSetFPS(fps)
      storeSetLatency(elapsedMs)
      if (sceneCondition) storeSetScene(sceneCondition)
    },
    [storeSetWorkers, storeSetViolations, storeSetCount, storeSetFPS, storeSetLatency, storeSetScene, storeSetRunning],
  )

  const setRunning = useCallback((running: boolean) => {
    setStats(prev => ({ ...prev, isRunning: running }))
    // Also update the global Zustand store so GeoAI detects the live state
    storeSetRunning(running)
  }, [storeSetRunning])

  const setModelName = useCallback((name: string) => {
    setStats(prev => ({ ...prev, modelName: name }))
  }, [])

  const resetStats = useCallback(() => {
    setStats({ ...EMPTY })
    setLiveAlerts([])
  }, [])

  return (
    <Ctx.Provider value={{ stats, liveAlerts, pushDetections, setRunning, setModelName, resetStats }}>
      {children}
    </Ctx.Provider>
  )
}
