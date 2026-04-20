import { 
  createContext, 
  useContext, 
  useState, 
  useCallback, 
  useMemo, 
  useRef, 
  useEffect, 
  type ReactNode 
} from 'react'
import type { AlertItem } from './components/AlertLog'
import { useDetectionStore, type WorkerPosition, type ZoneViolation } from './store/detectionStore'
import { useSettings } from './SettingsContext'

/* ═══════════════════════════════════════════════════════════════════════════════
   BuildSight — Persistent Detection Pipeline Context
   --------------------------------------------------
   This is the "Global Engine" of the BuildSight detection system. 
   It owns a persistent background inference loop using a hidden <video> 
   and <canvas> element. This ensures that detections continue even when 
   the user navigates between Dashboard, GeoAI, and Surveillance tabs.
   ═══════════════════════════════════════════════════════════════════════════════ */

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
  modelName: 'BS-ENSEMBLE-WBF',
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
  setModelName: (name: string) => void
  setRunning: (running: boolean) => void
}

const Ctx = createContext<DetectionPipelineCtx | null>(null)

export function useDetectionPipeline() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useDetectionPipeline must be used inside DetectionStatsProvider')
  return ctx
}

// Keep backward compatibility for existing code that uses useDetectionStats
export const useDetectionStats = useDetectionPipeline

export function DetectionStatsProvider({ children }: { children: ReactNode }) {
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

  // Zustand Store Bridge - Use stable state access
  const storeSetRunning = useDetectionStore(s => s.setRunning)
  const storeResetSession = useDetectionStore(s => s.resetSession)
  const storeSetVideoSession = useDetectionStore(s => s.setVideoSession)
  const storeSetPaused = useDetectionStore(s => s.setPaused)
  const storeSetPlayback = useDetectionStore(s => s.setPlayback)
  const storeSetWorkerPositions = useDetectionStore(s => s.setWorkerPositions)
  const storeSetViolations = useDetectionStore(s => s.setViolations)
  const storeSetWorkerCount = useDetectionStore(s => s.setWorkerCount)
  const storeSetFPS = useDetectionStore(s => s.setFPS)
  const storeSetProcessingRate = useDetectionStore(s => s.setProcessingRate)
  const storeSetLatencyMs = useDetectionStore(s => s.setLatencyMs)
  const storeSetSceneCondition = useDetectionStore(s => s.setSceneCondition)
  const storePeakRiskMoments = useDetectionStore(s => s.peakRiskMoments)
  const storeSetPeakRiskMoments = useDetectionStore(s => s.setPeakRiskMoments)
  const storeSetDetections = useDetectionStore(s => s.setDetections)
  const storeSetPpeWorkers = useDetectionStore(s => s.setPpeWorkers)

  // ── Alert Helpers ───────────────────────────────────────────────────────────
  const formatAlertTime = (date: Date) => {
    const hours = String(date.getHours()).padStart(2, '0')
    const mins = String(date.getMinutes()).padStart(2, '0')
    const secs = String(date.getSeconds()).padStart(2, '0')
    return `${hours}:${mins}:${secs}`
  }

  const buildLiveAlerts = useCallback((
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
        camera: 'GLOBAL / BACKGROUND PIPELINE',
        severity: 'critical',
        title: 'Hardhat non-compliance detected',
        detail: `${effectiveWorkers - helmets} worker${effectiveWorkers - helmets !== 1 ? 's' : ''} missing protection.`,
      })
    }

    if (effectiveWorkers > 0 && vests < effectiveWorkers) {
      alerts.push({
        id: `DL-${now.getTime()}-V`,
        time: formatAlertTime(now),
        camera: 'GLOBAL / BACKGROUND PIPELINE',
        severity: 'warning',
        title: 'High-vis vest non-compliance',
        detail: `${effectiveWorkers - vests} worker${effectiveWorkers - vests !== 1 ? 's' : ''} missing vest coverage.`,
      })
    }

    return alerts
  }, [])

  const appendDedupedAlerts = useCallback((prev: AlertItem[], nextAlerts: AlertItem[]) => {
    const now = Date.now()
    const DEDUPE_MS = 8000
    const MAX_ALERTS = 8

    const deduped = nextAlerts.filter((candidate) => {
      const match = prev.find((existing) => {
        const existingStamp = Number(existing.id.split('-')[1] ?? 0)
        return existing.title === candidate.title && now - existingStamp < DEDUPE_MS
      })
      return !match
    })

    return [...deduped, ...prev].slice(0, MAX_ALERTS)
  }, [])

  // ── Pipeline Controls ───────────────────────────────────────────────────────
  
  const startDetection = useCallback((file: File) => {
    console.log('--- GLOBAL PIPELINE START ---', file.name)
    currentSessionIdRef.current = `sess-${Date.now()}`
    
    // Reset local and store state
    setStats({ ...EMPTY, isRunning: true })
    setLiveAlerts([])
    storeResetSession()
    storeSetVideoSession(file)
    
    if (videoRef.current) {
      const url = URL.createObjectURL(file)
      videoRef.current.src = url
      videoRef.current.play()
    }
  }, [storeResetSession, storeSetVideoSession])

  const pauseDetection = useCallback(() => {
    videoRef.current?.pause()
    setStats(prev => ({ ...prev, isPaused: false })) 
    storeSetPaused(true)
  }, [storeSetPaused])

  const resumeDetection = useCallback(() => {
    videoRef.current?.play()
    storeSetPaused(false)
  }, [storeSetPaused])

  const stopDetection = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.src = ''
    }
    setStats({ ...EMPTY })
    storeResetSession()
    currentSessionIdRef.current = null
  }, [storeResetSession])

  const resetDetection = useCallback(() => {
    stopDetection()
    setLiveAlerts([])
  }, [stopDetection])

  const seekTo = useCallback((time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time
    }
  }, [])

  const setModelName = useCallback((name: string) => {
    setStats(prev => ({ ...prev, modelName: name }))
  }, [])

  const setRunning = useCallback((running: boolean) => {
    setStats(prev => ({ ...prev, isRunning: running }))
    storeSetRunning(running)
  }, [storeSetRunning])

  // Process incoming detection data (bridge for non-video detections)
  const pushDetections = useCallback((
    detections: Array<{ class: string; confidence: number }>,
    elapsedMs: number,
    validWorkers?: WorkerPosition[],
    violations?: ZoneViolation[],
    fps?: number,
    sceneCondition?: string
  ) => {
    let workers = 0, helmets = 0, vests = 0, confSum = 0

    for (const d of detections) {
      const cls = d.class.toLowerCase()
      if (cls === 'worker' || cls === 'person') workers++
      if (cls === 'helmet' || cls === 'hardhat') helmets++
      if (cls === 'safety_vest' || cls === 'vest') vests++
      confSum += d.confidence
    }

    const effectiveWorkers = workers > 0 ? workers : Math.max(helmets, vests)
    const avgConfidence = detections.length > 0 ? confSum / detections.length : 0

    // Update Local Context
    setStats(prev => ({
      ...prev,
      totalWorkers: effectiveWorkers,
      helmetsDetected: helmets,
      vestsDetected: vests,
      proximityViolations: violations?.length ?? prev.proximityViolations,
      avgConfidence,
      elapsedMs,
      framesScanned: prev.framesScanned + 1,
      isRunning: true
    }))

    // Generate Alerts
    const alerts = buildLiveAlerts(detections, effectiveWorkers, helmets, vests)
    if (alerts.length > 0) {
      setLiveAlerts(prev => appendDedupedAlerts(prev, alerts))
    }

    // Update Global Zustand Store (The true persistence layer)
    storeSetDetections(detections)
    if (validWorkers) {
      storeSetWorkerPositions(validWorkers)
      storeSetPpeWorkers(validWorkers)
    }
    if (violations) storeSetViolations(violations)
    storeSetWorkerCount(effectiveWorkers)
    if (fps) {
        storeSetFPS(fps)
        storeSetProcessingRate(fps)
    }
    storeSetLatencyMs(elapsedMs)
    if (sceneCondition) storeSetSceneCondition(sceneCondition)
  }, [buildLiveAlerts, appendDedupedAlerts, storeSetDetections, storeSetWorkerPositions, storeSetPpeWorkers, storeSetViolations, storeSetWorkerCount, storeSetFPS, storeSetProcessingRate, storeSetLatencyMs, storeSetSceneCondition])

  // ── INFERENCE LOOP ENGINE ──────────────────────────────────────────────────
  
  const processFrame = useCallback(async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.paused || video.ended || isProcessingRef.current) return

    // Throttle inference based on target FPS (default 12 FPS)
    const now = performance.now()
    const targetInterval = 1000 / 12
    if (now - lastInferenceTimeRef.current < targetInterval) return

    isProcessingRef.current = true
    lastInferenceTimeRef.current = now

    try {
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      
      // Ensure canvas matches video resolution
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }

      ctx.drawImage(video, 0, 0)

      // /api/detect/frame expects image_b64 (base64 string), NOT a raw file blob
      const image_b64 = canvas.toDataURL('image/jpeg', 0.8)

      // Update Playback Stats in Store
      storeSetPlayback(video.currentTime, video.duration, (video.currentTime / video.duration) * 100)

      const formData = new FormData()
      formData.append('image_b64', image_b64)
      formData.append('condition', 'S1_normal')
      formData.append('auto_condition', '1')
      formData.append('confidence', settings.confidenceThreshold.toString())

      const res = await fetch('http://localhost:8000/api/detect/frame', {
        method: 'POST',
        body: formData,
      })

      if (res.ok) {
        const data = await res.json()
        pushDetections(
          data.detections || [],
          data.elapsed_ms || 0,
          data.valid_workers || [],
          [],
          data.elapsed_ms ? Math.round(1000 / Math.max(data.elapsed_ms, 1)) : 12,
          data.condition || 'S1_normal'
        )

        // PEAK RISK: derive from valid_workers ppe_compliant flag (authoritative server list)
        const nonCompliant = (data.valid_workers || []).filter((w: any) => !w.ppe_compliant)
        if (nonCompliant.length > 0) {
          const currentMoments = storePeakRiskMoments
          const alreadyTracked = currentMoments.some(m => Math.abs(m.time - video.currentTime) < 2)
          if (!alreadyTracked) {
            const newMoment = {
              time: video.currentTime,
              score: nonCompliant.length * 20,
              type: 'PPE_VIOLATION' as const
            }
            storeSetPeakRiskMoments([...currentMoments, newMoment].slice(-10))
          }
        }
      }
    } catch (err) {
      console.warn('Background Inference Error:', err)
    } finally {
      isProcessingRef.current = false
    }
  }, [pushDetections, settings.confidenceThreshold, storeSetPlayback, storePeakRiskMoments, storeSetPeakRiskMoments])

  useEffect(() => {
    const loop = () => {
      processFrame()
      loopIdRef.current = requestAnimationFrame(loop)
    }
    loopIdRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(loopIdRef.current)
  }, [processFrame])

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
  }, [storeSetPaused, storeSetRunning, storeSetPlayback])

  const value = useMemo(() => ({ 
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
    setRunning
  }), [
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
    setRunning
  ])

  return (
    <Ctx.Provider value={value}>
      {children}
      {/* ── HIDDEN WORKER ELEMENTS ── */}
      <div style={{ position: 'fixed', top: -1000, left: -1000, visibility: 'hidden', pointerEvents: 'none' }}>
        <video 
          ref={videoRef} 
          muted 
          playsInline 
          preload="auto"
          onLoadedMetadata={() => {
            if (videoRef.current) storeSetPlayback(0, videoRef.current.duration, 0)
          }}
        />
        <canvas ref={canvasRef} />
      </div>
    </Ctx.Provider>
  )
}
