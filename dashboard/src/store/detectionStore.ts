/**
 * BuildSight Global Detection State Store
 * =========================================
 * Persists background detection state across ALL tab navigation.
 * Uses Zustand for lightweight global state.
 * Updated via a single native WebSocket connection to the FastAPI
 * backend (/ws/detection) that is opened once at app startup and
 * never closed on tab change.
 */

import { create } from 'zustand'

export interface WorkerPosition {
  worker_id:      string
  confidence:     number
  lat:            number
  lng:            number
  utm_e:          number
  utm_n:          number
  has_helmet:     boolean | null
  has_vest:       boolean | null
  ppe_compliant:  boolean
  violation_type: string[]
  pixel_x:        number
  pixel_y:        number
  zone_id:        string | null
  zone_name:      string | null
  source?:        string
}

export interface ZoneViolation {
  type:       string
  severity:   'CRITICAL' | 'WARNING' | 'INFO'
  worker_id?: string
  zone_id:    string
  zone_name:  string
  details:    string
  lat:        number
  lng:        number
  timestamp:  number
}

interface DetectionState {
  // Connection
  isConnected:     boolean
  _socket:         WebSocket | null

  // Video / service state
  isRunning:       boolean
  isPaused:        boolean
  currentFrame:    number
  totalFrames:     number
  progressPercent: number

  // Detection metrics
  workerCount:     number
  sceneCondition:  string
  fps:             number
  latencyMs:       number
  frameCount:      number

  // Spatial state — drives GeoAI map live markers
  workerPositions: WorkerPosition[]
  zoneOccupancy:   Record<string, string[]>
  violations:      ZoneViolation[]

  // Actions
  connect:         () => void
  disconnect:      () => void
  requestSnapshot: () => void
  /** Directly set isRunning (called by video-upload inference to trigger GeoAI live mode) */
  setRunning:         (running: boolean) => void
  setWorkerPositions: (workers: WorkerPosition[]) => void
  setViolations:      (violations: ZoneViolation[]) => void
  setWorkerCount:     (count: number) => void
  setFPS:             (fps: number) => void
  setLatencyMs:       (ms: number) => void
  setSceneCondition:  (cond: string) => void
}

const WS_URL = 'ws://localhost:8000/ws/detection'

export const useDetectionStore = create<DetectionState>((set, get) => ({
  isConnected:     false,
  _socket:         null,
  isRunning:       false,
  isPaused:        false,
  currentFrame:    0,
  totalFrames:     0,
  progressPercent: 0,
  workerCount:     0,
  sceneCondition:  'S1_normal',
  fps:             0,
  latencyMs:       0,
  frameCount:      0,
  workerPositions: [],
  zoneOccupancy:   {},
  violations:      [],

  connect: () => {
    const existing = get()._socket
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return
    }

    const socket = new WebSocket(WS_URL)

    socket.addEventListener('open', () => {
      set({ isConnected: true })
      socket.send('request_state_snapshot')
    })

    socket.addEventListener('close', () => {
      set({ isConnected: false, _socket: null })
      // Auto-reconnect after 2 s
      setTimeout(() => {
        if (!get()._socket) get().connect()
      }, 2000)
    })

    socket.addEventListener('error', () => {
      // close event follows automatically — reconnect handled there
    })

    socket.addEventListener('message', (event) => {
      let data: any
      try { data = JSON.parse(event.data as string) } catch { return }

      const type: string = data.type ?? ''

      if (type === 'detection_update') {
        const total = data.total_frames ?? 0
        const frame = data.frame_number ?? 0
        set({
          workerCount:     data.worker_count      ?? 0,
          sceneCondition:  data.scene_condition   ?? 'S1_normal',
          fps:             data.fps               ?? 0,
          latencyMs:       data.latency_ms        ?? 0,
          currentFrame:    frame,
          totalFrames:     total,
          progressPercent: total > 0 ? Math.round((frame / total) * 100) : 0,
          workerPositions: data.worker_positions  ?? [],
          zoneOccupancy:   data.zone_occupancy    ?? {},
          violations:      data.violations        ?? [],
          frameCount:      get().frameCount + 1,
        })
      } else if (type === 'detection_state_snapshot') {
        const total = data.total_frames ?? 0
        const frame = data.current_frame ?? 0
        set({
          isRunning:       data.is_running        ?? false,
          isPaused:        data.is_paused         ?? false,
          currentFrame:    frame,
          totalFrames:     total,
          progressPercent: total > 0 ? Math.round((frame / total) * 100) : 0,
          workerCount:     data.worker_count      ?? 0,
          sceneCondition:  data.scene_condition   ?? 'S1_normal',
          fps:             data.fps               ?? 0,
          latencyMs:       data.latency_ms        ?? 0,
          workerPositions: data.worker_positions  ?? [],
          zoneOccupancy:   data.zone_occupancy    ?? {},
          violations:      data.violations        ?? [],
        })
      } else if (type === 'detection_started') {
        set({ isRunning: true, isPaused: false, totalFrames: data.total_frames ?? 0 })
      } else if (type === 'detection_paused') {
        set({ isPaused: true })
      } else if (type === 'detection_resumed') {
        set({ isPaused: false })
      } else if (type === 'detection_stopped') {
        set({
          isRunning:       false,
          workerPositions: [],
          zoneOccupancy:   {},
          violations:      [],
          progressPercent: 0,
        })
      }
    })

    set({ _socket: socket })
  },

  disconnect: () => {
    get()._socket?.close()
    set({ _socket: null, isConnected: false })
  },

  requestSnapshot: () => {
    const socket = get()._socket
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send('request_state_snapshot')
    }
  },

  setRunning: (running: boolean) => {
    set({ isRunning: running })
  },

  setWorkerPositions: (workers: WorkerPosition[]) => {
    set({ workerPositions: workers })
  },

  setViolations: (violations: ZoneViolation[]) => {
    set({ violations })
  },

  setWorkerCount: (count: number) => {
    set({ workerCount: count })
  },

  setFPS: (fps: number) => {
    set({ fps })
  },

  setLatencyMs: (ms: number) => {
    set({ latencyMs: ms })
  },

  setSceneCondition: (cond: string) => {
    set({ sceneCondition: cond })
  },
}))
