/* ─────────────────────────────────────────────────────────────────────────────
 * useGeoAIWebSocket — WebSocket connection to the heatmap engine
 * v2: Intelligence Edition — includes events, trails, KPI, and health data
 * Falls back to demo mode if the engine is unreachable.
 * ──────────────────────────────────────────────────────────────────────────── */

import { useState, useEffect, useRef, useCallback } from 'react'
import type {
  HeatmapUpdatePayload, WorkerPosition, RiskCell, GeoAIAlert,
  SpatialEvent, WorkerTrail, KPISummary, BackendHealth,
  AlertStateType,
} from '../types/geoai'

const WS_URL = 'ws://localhost:8765'
const DEMO_FALLBACK_MS = 4000
const DEMO_TICK_MS = 2000

/* ── Site coordinates (from SITE_CONFIG in heatmap engine) ───────────── */
const SW_LAT = 10.81658333
const SW_LON = 78.66873333

type ConnectionState = 'connecting' | 'live' | 'demo' | 'disconnected'

interface GeoAIState {
  data: HeatmapUpdatePayload | null
  connectionState: ConnectionState
  isLive: boolean
  isDemoMode: boolean
  cycle: number
  acknowledgeEvent: (eventId: string) => void
  resolveEvent: (eventId: string) => void
}

/* ── Demo data generators (ported from buildsight_heatmap_viz.html) ──── */

const DEMO_WORKERS: WorkerPosition[] = [
  { lat: 10.81665, lon: 78.66880, ppe_ok: false, has_helmet: false, has_vest: true,  zone: 'high_risk_scaffolding', risk: 'HIGH',     height_m: 5.2, worker_id: 1, status: 'AT_RISK',  dwell_time_s: 45, dwell_severity: 'LOW',  camera_id: 'CAM-01' },
  { lat: 10.81660, lon: 78.66878, ppe_ok: true,  has_helmet: true,  has_vest: true,  zone: 'moderate_risk_interior', risk: 'MODERATE', height_m: 1.8, worker_id: 2, status: 'SAFE',     dwell_time_s: 0,  dwell_severity: 'NONE', camera_id: 'CAM-01' },
  { lat: 10.81663, lon: 78.66882, ppe_ok: true,  has_helmet: true,  has_vest: true,  zone: 'moderate_risk_interior', risk: 'MODERATE', height_m: 2.1, worker_id: 3, status: 'SAFE',     dwell_time_s: 12, dwell_severity: 'NONE', camera_id: 'CAM-02' },
  { lat: 10.81667, lon: 78.66875, ppe_ok: false, has_helmet: true,  has_vest: false, zone: 'high_risk_staircase',    risk: 'CRITICAL', height_m: 5.8, worker_id: 4, status: 'CRITICAL', dwell_time_s: 85, dwell_severity: 'MEDIUM', camera_id: 'CAM-01' },
]

const round6 = (n: number) => Math.round(n * 1e6) / 1e6
const round4 = (n: number) => Math.round(n * 1e4) / 1e4

function buildDemoCells(): RiskCell[] {
  const cells: RiskCell[] = []
  for (let ci = 0; ci < 9; ci++) {
    for (let ri = 0; ri < 4; ri++) {
      const wx = ci * 2 + 1
      const wy = ri * 2 + 1
      const lat = SW_LAT + wy / 110574
      const lon = SW_LON + wx / (111319 * Math.cos(SW_LAT * Math.PI / 180))
      const edge = Math.min(wx, 18.9 - wx, wy, 9.75 - wy)
      const risk = Math.max(0, 0.7 - edge * 0.08 + Math.random() * 0.15)
      cells.push({
        lat: round6(lat),
        lon: round6(lon),
        risk_score: round4(risk),
        risk_level: risk > 0.7 ? 'CRITICAL' : risk > 0.5 ? 'HIGH' : risk > 0.3 ? 'MODERATE' : 'LOW',
        worker_count: Math.random() > 0.7 ? 1 : 0,
        ppe_violations: 0,
      })
    }
  }
  return cells
}

const DEMO_ALERTS: GeoAIAlert[] = [
  {
    alert_level: 'CRITICAL',
    alert_type: 'FALL_RISK',
    alert_message: 'FALL RISK: 1 worker detected at roof/parapet level (≥4.88m)',
    bocw_reference: 'BOCW §40 — Fall Protection Required',
    worker_count: 1,
    ppe_violations: 1,
    risk_score: 0.95,
    timestamp: new Date().toISOString(),
  },
  {
    alert_level: 'WARNING',
    alert_type: 'PPE_VIOLATION',
    alert_message: 'PPE VIOLATION: 2/4 workers without required PPE',
    bocw_reference: 'BOCW §40 + NBC 2016 §4.8',
    worker_count: 4,
    ppe_violations: 2,
    risk_score: 0.50,
    timestamp: new Date(Date.now() - 45000).toISOString(),
  },
]

/* ── Demo intelligence data ────────────────────────────────────────────── */

let demoEventIdCounter = 0
function makeDemoEventId() {
  demoEventIdCounter++
  return `demo-${demoEventIdCounter.toString(16).padStart(4, '0')}`
}

function buildDemoEvents(_cycle: number): SpatialEvent[] {
  const now = new Date()
  const events: SpatialEvent[] = [
    {
      event_id: makeDemoEventId(),
      event_type: 'GEOFENCE_BREACH',
      priority: 'CRITICAL',
      timestamp: new Date(now.getTime() - 10000).toISOString(),
      worker_id: 4,
      zone: 'high_risk_staircase',
      camera_id: 'CAM-01',
      message: 'GEOFENCE: RESTRICTED_ENTRY — Worker W4 at high risk staircase',
      details: { breach_type: 'RESTRICTED_ENTRY', violations: 2 },
      state: 'ACTIVE',
      sla_remaining_s: 280,
    },
    {
      event_id: makeDemoEventId(),
      event_type: 'DWELL_BREACH',
      priority: 'WARNING',
      timestamp: new Date(now.getTime() - 30000).toISOString(),
      worker_id: 4,
      zone: 'high_risk_staircase',
      camera_id: 'CAM-01',
      message: 'Worker W4 stationary for 85s in high risk staircase',
      details: { dwell_time_s: 85, severity: 'MEDIUM' },
      state: 'ACTIVE',
      sla_remaining_s: 250,
    },
    {
      event_id: makeDemoEventId(),
      event_type: 'PPE_VIOLATION',
      priority: 'WARNING',
      timestamp: new Date(now.getTime() - 60000).toISOString(),
      worker_id: 1,
      zone: 'high_risk_scaffolding',
      camera_id: 'CAM-01',
      message: 'PPE missing: helmet — Worker W1',
      details: { missing: ['helmet'] },
      state: 'ACTIVE',
    },
    {
      event_id: makeDemoEventId(),
      event_type: 'ZONE_ENTRY',
      priority: 'INFO',
      timestamp: new Date(now.getTime() - 120000).toISOString(),
      worker_id: 2,
      zone: 'moderate_risk_interior',
      camera_id: 'CAM-01',
      message: 'W2 entered moderate risk interior',
      details: {},
      state: 'ACTIVE',
    },
    {
      event_id: makeDemoEventId(),
      event_type: 'STATUS_CHANGE',
      priority: 'CRITICAL',
      timestamp: new Date(now.getTime() - 180000).toISOString(),
      worker_id: 4,
      zone: 'high_risk_staircase',
      camera_id: 'CAM-01',
      message: 'W4 status: AT_RISK → CRITICAL',
      details: { from: 'AT_RISK', to: 'CRITICAL' },
      state: 'ACTIVE',
      sla_remaining_s: 120,
    },
  ]
  return events
}

function buildDemoTrails(): WorkerTrail[] {
  const now = Date.now() / 1000
  return DEMO_WORKERS.map(w => ({
    worker_id: w.worker_id ?? 0,
    camera_id: w.camera_id ?? 'CAM-01',
    current_zone: w.zone,
    status: w.status ?? 'SAFE',
    positions: Array.from({ length: 20 }, (_, i) => ({
      lat: w.lat + (Math.random() - 0.5) * 0.00005 * (20 - i),
      lon: w.lon + (Math.random() - 0.5) * 0.00005 * (20 - i),
      world_x: 5 + Math.random() * 10,
      world_y: 2 + Math.random() * 6,
      timestamp: now - (20 - i) * 30,
      status: w.status ?? 'SAFE',
      dwell_severity: w.dwell_severity ?? 'NONE',
    })),
  }))
}

function buildDemoKPI(): KPISummary {
  return {
    active_workers: DEMO_WORKERS.length,
    critical_alerts: 2,
    escalated_alerts: 0,
    total_events: 11,
    ppe_compliance: 50.0,
    avg_site_risk: 0.42,
    system_degraded: false,
    mapper_status: 'DEMO_MODE',
  }
}

function buildDemoHealth(): BackendHealth[] {
  return [
    { service: 'PostGIS',       status: 'DEGRADED', buffer_workers: 12, buffer_events: 5 },
    { service: 'SpatialMapper', status: 'LINEAR_FALLBACK' },
    { service: 'WebSocket',     status: 'HEALTHY' },
    { service: 'Pipeline',      status: 'HEALTHY' },
    { service: 'GEE',           status: 'UNAVAILABLE' },
  ]
}

function generateDemoTick(cycle: number): HeatmapUpdatePayload {
  // Jitter positions slightly for realism
  const workers = DEMO_WORKERS.map(w => ({
    ...w,
    lat: w.lat + (Math.random() - 0.5) * 0.00003,
    lon: w.lon + (Math.random() - 0.5) * 0.00003,
    dwell_time_s: (w.dwell_time_s ?? 0) + cycle * 2,
  }))

  const cells = buildDemoCells().map(c => ({
    ...c,
    risk_score: Math.min(1, Math.max(0, c.risk_score + (Math.random() - 0.5) * 0.05)),
  }))

  const compliant = workers.filter(w => w.ppe_ok).length

  return {
    type: 'heatmap_update',
    cycle,
    workers,
    cells,
    alerts: DEMO_ALERTS.map(a => ({ ...a, timestamp: new Date().toISOString() })),
    site_stats: {
      total_workers: workers.length,
      ppe_compliant: compliant,
      max_risk_score: Math.max(...cells.map(c => c.risk_score)),
      critical_zones: cells.filter(c => c.risk_level === 'CRITICAL').length,
    },
    events: buildDemoEvents(cycle),
    trails: buildDemoTrails(),
    kpi: buildDemoKPI(),
    backend_health: buildDemoHealth(),
  }
}

/* ── Hook ──────────────────────────────────────────────────────────────── */

export function useGeoAIWebSocket(): GeoAIState {
  const [data, setData] = useState<HeatmapUpdatePayload | null>(null)
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')
  const cycleRef = useRef(0)
  const wsRef = useRef<WebSocket | null>(null)
  const demoIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const startDemoMode = useCallback(() => {
    if (demoIntervalRef.current) return // already running
    setConnectionState('demo')
    demoIntervalRef.current = setInterval(() => {
      cycleRef.current += 1
      setData(generateDemoTick(cycleRef.current))
    }, DEMO_TICK_MS)
    // Emit first tick immediately
    cycleRef.current += 1
    setData(generateDemoTick(cycleRef.current))
  }, [])

  const stopDemoMode = useCallback(() => {
    if (demoIntervalRef.current) {
      clearInterval(demoIntervalRef.current)
      demoIntervalRef.current = null
    }
  }, [])

  /* ── Event management callbacks ─────────────────────────────────────── */
  const acknowledgeEvent = useCallback((eventId: string) => {
    setData(prev => {
      if (!prev?.events) return prev
      const updatedEvents = prev.events.map(e =>
        e.event_id === eventId && e.state === 'ACTIVE'
          ? { ...e, state: 'ACKNOWLEDGED' as AlertStateType, acknowledged_by: 'operator', acknowledged_at: new Date().toISOString() }
          : e
      )
      return { ...prev, events: updatedEvents }
    })
  }, [])

  const resolveEvent = useCallback((eventId: string) => {
    setData(prev => {
      if (!prev?.events) return prev
      const updatedEvents = prev.events.map(e =>
        e.event_id === eventId && (e.state === 'ACTIVE' || e.state === 'ACKNOWLEDGED' || e.state === 'ESCALATED')
          ? { ...e, state: 'RESOLVED' as AlertStateType, resolved_by: 'operator', resolved_at: new Date().toISOString() }
          : e
      )
      return { ...prev, events: updatedEvents }
    })
  }, [])

  useEffect(() => {
    let fallbackTimeout: ReturnType<typeof setTimeout>

    const connectWS = () => {
      try {
        const ws = new WebSocket(WS_URL)
        wsRef.current = ws

        ws.onopen = () => {
          setConnectionState('live')
          stopDemoMode()
        }

        ws.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data) as HeatmapUpdatePayload
            if (payload.type === 'heatmap_update') {
              cycleRef.current = payload.cycle
              setData(payload)
            }
          } catch {
            // Ignore malformed messages
          }
        }

        ws.onclose = () => {
          setConnectionState('disconnected')
          wsRef.current = null
          // Fall back to demo if connection lost
          startDemoMode()
        }

        ws.onerror = () => {
          ws.close()
        }
      } catch {
        // WebSocket constructor threw (invalid URL, etc.)
        startDemoMode()
      }
    }

    connectWS()

    // If not connected within timeout, start demo mode
    fallbackTimeout = setTimeout(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        startDemoMode()
      }
    }, DEMO_FALLBACK_MS)

    return () => {
      clearTimeout(fallbackTimeout)
      stopDemoMode()
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [startDemoMode, stopDemoMode])

  return {
    data,
    connectionState,
    isLive: connectionState === 'live',
    isDemoMode: connectionState === 'demo',
    cycle: cycleRef.current,
    acknowledgeEvent,
    resolveEvent,
  }
}
