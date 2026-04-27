import { createContext, useContext, useState, useCallback, useRef, useEffect, type ReactNode } from 'react'
import type { AlertItem } from './components/AlertLog'
import { useDetectionStore, type WorkerPosition, type ZoneViolation } from './store/detectionStore'
import { useSettings } from './SettingsContext'
import { buildPeakRiskMoment, mergePeakRiskMoments } from './lib/detectionIntelligence'

/* ═══════════════════════════════════════════════════════════════════════════════
   BuildSight — Detection Pipeline Provider
   The engine for persistent background inference. Moves the loop from
   DetectionPanel (view) to the global root (service).
   ═══════════════════════════════════════════════════════════════════════════════ */

const API = 'http://localhost:8000/api'
const MAX_ALERTS = 6
const ALERT_DEDUPE_MS = 8000

export interface DetectionStats {
  totalWorkers: number
  helmetsDetected: number
  vestsDetected: number
  proximityViolations: number
  avgConfidence: number
  elapsedMs: number
  framesScanned: number
  isRunning: boolean
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

interface DetectionPipelineCtx {
  stats: DetectionStats
  liveAlerts: AlertItem[]
  
  // Pipeline Controls
  startDetection: (file: File) => void
  pauseDetection: () => void
  resumeDetection: () => void
  stopDetection: () => void
  resetDetection: () => void
  seekTo: (time: number) => void
  
  // Bridge for manual/WS updates
  pushDetections: (
    detections: Array<{ class: string; confidence: number }>,
    elapsedMs: number,
    validWorkers?: WorkerPosition[],
    violations?: ZoneViolation[],
    fps?: number,
    sceneCondition?: string
  ) => void
  setModelName:  (name: string) => void
  setRunning:    (running: boolean) => void
}

const Ctx = createContext<DetectionPipelineCtx | null>(null)

export function useDetectionPipeline() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useDetectionPipeline must be used inside DetectionPipelineProvider')
  return ctx
}

// Keep backward compatibility for existing code that uses useDetectionStats
export const useDetectionStats = useDetectionPipeline

export function DetectionPipelineProvider({ children }: { children: ReactNode }) {
  const [stats, setStats] = useState<DetectionStats>({ ...EMPTY })
  const [liveAlerts, setLiveAlerts] = useState<AlertItem[]>([])
  const { settings } = useSettings()

  // Hidden DOM Refs for Background Processing
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  // Engine Refs
  const loopIdRef = useRef<number>(0)
  const isProcessingRef = useRef<boolean>(false)
  const lastInferenceTimeRef = useRef<number>(0)
  const currentSessionIdRef = useRef<string | null>(null)

  // Zustand Store Bridge
  const store = useDetectionStore()

  // ── Alert Helpers ───────────────────────────────────────────────────────────
  const formatAlertTime = (date: Date) => {
    const hours = String(date.getHours()).padStart(2, '0')
    const mins = String(date.getMinutes()).padStart(2, '0')
    const secs = String(date.getSeconds()).padStart(2, '0')
    return `${hours}:${mins}:${secs}`
  }

  const buildLiveAlerts = (
    detections: Array<{ class: string; confidence: number }>,
    effectiveWorkers: number,
    helmets: number,
    vests: number,
  ): AlertItem[] => {
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
        detail: `Average confidence dropped to ${(avgConfidence * 100).toFixed(0)}%. Manual review recommended.`,
      })
    }

    return alerts
  }

  const appendDedupedAlerts = useCallback((prev: AlertItem[], nextAlerts: AlertItem[]) => {
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
  }, [])

  // ── Engine Core ─────────────────────────────────────────────────────────────
  
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

      // Prefer backend's valid_workers count — it's the authoritative spatially-mapped count
      const effectiveWorkers = (validWorkers && validWorkers.length > 0)
        ? validWorkers.length
        : workers > 0 ? workers : Math.max(helmets, vests)

      setLiveAlerts(prev => appendDedupedAlerts(prev, buildLiveAlerts(detections, effectiveWorkers, helmets, vests)))

      setStats(prev => ({
        ...prev,
        totalWorkers: effectiveWorkers,
        helmetsDetected: helmets,
        vestsDetected: vests,
        proximityViolations: violations?.length ?? prev.proximityViolations,
        avgConfidence: detections.length > 0
          ? detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length
          : prev.avgConfidence,
        elapsedMs,
        framesScanned: prev.framesScanned + 1,
        isRunning: true,
      }))

      // Sync to Zustand
      const s = useDetectionStore.getState()
      if (validWorkers) s.setWorkerPositions(validWorkers)
      if (violations)   s.setViolations(violations)
      s.setWorkerCount(effectiveWorkers)
      if (fps !== undefined) s.setFPS(fps)
      s.setLatencyMs(elapsedMs)
      if (sceneCondition) s.setSceneCondition(sceneCondition)
      s.setDetections(detections)
      if (validWorkers) s.setPpeWorkers(validWorkers)
    },
    [appendDedupedAlerts],
  )

  const runInference = async (sessionId: string) => {
    if (!videoRef.current || !canvasRef.current || isProcessingRef.current) return
    if (videoRef.current.paused || videoRef.current.ended) return
    if (sessionId !== currentSessionIdRef.current) return

    isProcessingRef.current = true
    const t0 = performance.now()
    
    try {
      const video = videoRef.current
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // Extract high-quality frame
      const MAX_DIM = 640
      const scale = Math.min(1, MAX_DIM / Math.max(video.videoWidth, video.videoHeight, 1))
      canvas.width = Math.round(video.videoWidth * scale)
      canvas.height = Math.round(video.videoHeight * scale)
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      const b64 = canvas.toDataURL('image/jpeg', 0.8)
      
      const form = new FormData()
      form.append('image_b64', b64)
      form.append('model', settings.selectedModel)
      form.append('condition', store.sceneCondition)
      form.append('auto_condition', '1')
      if (store.roiPoly && store.roiPoly.length >= 3) {
        form.append('zone_poly', JSON.stringify(store.roiPoly))
      }

      const res = await fetch(`${API}/detect/frame`, { method: 'POST', body: form })
      
      // Secondary check: verify session hasn't changed during the async wait
      if (sessionId !== currentSessionIdRef.current) return

      if (res.ok) {
        const data = await res.json()
        const elapsed = Math.round(performance.now() - t0)
        pushDetections(data.detections, elapsed, data.valid_workers, data.violations, undefined, data.condition)
        
        const nextMoment = buildPeakRiskMoment(data.detections, video.currentTime)
        if (nextMoment) {
          store.setPeakRiskMoments(
            mergePeakRiskMoments(store.peakRiskMoments, nextMoment, 6, 1.75),
          )
        }

        // Calculate Processing Rate (Inference FPS)
        const rate = Math.round(1000 / Math.max(16, elapsed))
        store.setProcessingRate(rate)

        // Sync Playback Progress
        const progress = Math.round((video.currentTime / video.duration) * 100)
        store.setPlayback(video.currentTime, video.duration, progress)
      }
    } catch (err) {
      console.error('Inference Loop Error:', err)
    } finally {
      isProcessingRef.current = false
    }
  }

  const loop = () => {
    if (!store.isRunning || store.isPaused || !currentSessionIdRef.current) return
    
    const now = performance.now()
    // Target ~12 inference FPS to not saturate the backend
    if (now - lastInferenceTimeRef.current >= 80) {
      lastInferenceTimeRef.current = now
      runInference(currentSessionIdRef.current!)
    }
    loopIdRef.current = requestAnimationFrame(loop)
  }

  // ── Unified Controller API ──────────────────────────────────────────────────
  
  const startDetection = useCallback((file: File) => {
    const video = videoRef.current
    if (!video) return

    if (currentSessionIdRef.current) resetDetection()

    const url = URL.createObjectURL(file)
    const sid = `v-${Date.now()}`
    currentSessionIdRef.current = sid
    video.src = url

    video.play().then(() => {
      useDetectionStore.getState().setVideoSession(file)
      useDetectionStore.getState().setRunning(true)

      // Upload to server-side BG service so GeoAI gets live worker positions
      const form = new FormData()
      form.append('video', file)
      fetch(`${API}/video/upload-background`, { method: 'POST', body: form })
        .then(r => r.ok ? r.json() : Promise.reject(r.status))
        .then(() => useDetectionStore.getState().requestSnapshot())
        .catch(err => console.warn('[BG] Server-side upload failed:', err))
    }).catch(e => console.error('[Detection] Video play failed:', e))
  }, [])

  const pauseDetection = useCallback(() => {
    videoRef.current?.pause()
    useDetectionStore.getState().setPaused(true)
    fetch(`${API}/detection/pause`, { method: 'POST' }).catch(() => {})
  }, [])

  const resumeDetection = useCallback(() => {
    videoRef.current?.play().then(() => {
      useDetectionStore.getState().setPaused(false)
      fetch(`${API}/detection/resume`, { method: 'POST' }).catch(() => {})
    }).catch(() => {})
  }, [])

  const seekTo = useCallback((time: number) => {
    if (videoRef.current) {
      const dur = videoRef.current.duration || useDetectionStore.getState().videoDuration || 1
      videoRef.current.currentTime = time
      useDetectionStore.getState().setPlayback(time, dur, Math.round((time / dur) * 100))
    }
  }, [])

  const stopDetection = useCallback(() => {
    videoRef.current?.pause()
    useDetectionStore.getState().setRunning(false)
    useDetectionStore.getState().setPaused(false)
    if (loopIdRef.current) cancelAnimationFrame(loopIdRef.current)
    fetch(`${API}/detection/stop`, { method: 'POST' }).catch(() => {})
  }, [])

  const resetDetection = useCallback(() => {
    stopDetection()
    currentSessionIdRef.current = null
    if (videoRef.current) {
      const src = videoRef.current.src
      if (src) URL.revokeObjectURL(src)
      videoRef.current.src = ''
      videoRef.current.load()
    }
    useDetectionStore.getState().resetSession()
    setStats({ ...EMPTY })
    setLiveAlerts([])
  }, [stopDetection])

  const setModelName = useCallback((name: string) => {
    setStats(prev => ({ ...prev, modelName: name }))
  }, [])

  // Start the background loop when state changes
  useEffect(() => {
    if (store.isRunning && !store.isPaused && currentSessionIdRef.current) {
      loopIdRef.current = requestAnimationFrame(loop)
    } else {
      if (loopIdRef.current) cancelAnimationFrame(loopIdRef.current)
    }
    return () => { if (loopIdRef.current) cancelAnimationFrame(loopIdRef.current) }
  }, [store.isRunning, store.isPaused])

  // Monitor video end
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const onEnded = () => {
      console.log('Background video ended naturally.')
      store.setPaused(false)
      store.setRunning(false)
      store.setPlayback(video.duration, video.duration, 100)
    }
    video.addEventListener('ended', onEnded)
    return () => video.removeEventListener('ended', onEnded)
  }, [store])

  return (
    <Ctx.Provider value={{
      stats,
      liveAlerts,
      startDetection,
      pauseDetection,
      resumeDetection,
      stopDetection,
      resetDetection,
      seekTo,
      pushDetections,
      setModelName,
      setRunning: store.setRunning,
    }}>
      {children}
      {/* ── HIDDEN WORKER ELEMENTS ── */}
      <div style={{ position: 'fixed', top: -1000, left: -1000, visibility: 'hidden', pointerEvents: 'none' }}>
        <video 
          ref={videoRef} 
          muted 
          playsInline 
          preload="auto"
          onLoadedMetadata={() => {
            if (videoRef.current) store.setPlayback(0, videoRef.current.duration, 0)
          }}
        />
        <canvas ref={canvasRef} />
      </div>
    </Ctx.Provider>
  )
}

// Backward-compat alias so App.tsx doesn't need updating
export const DetectionStatsProvider = DetectionPipelineProvider
