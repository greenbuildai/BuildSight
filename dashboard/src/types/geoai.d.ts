/* ─────────────────────────────────────────────────────────────────────────────
 * BuildSight GeoAI — TypeScript Declarations (v2 — Intelligence Edition)
 * ──────────────────────────────────────────────────────────────────────────── */

/* ── Leaflet & heatmap.js globals (loaded via CDN in index.html) ────────── */
import type * as LeafletNS from 'leaflet'

declare global {
  interface Window {
    L: typeof LeafletNS
    h337: {
      create: (config: {
        container: HTMLElement
        radius?: number
        maxOpacity?: number
        minOpacity?: number
        blur?: number
        gradient?: Record<string, string>
        [key: string]: unknown
      }) => HeatmapInstance
    }
    HeatmapOverlay: new (config: {
      radius?: number
      maxOpacity?: number
      minOpacity?: number
      blur?: number
      gradient?: Record<string, string>
      latField?: string
      lngField?: string
      valueField?: string
      scaleRadius?: boolean
      useLocalExtrema?: boolean
      [key: string]: unknown
    }) => LeafletNS.Layer & {
      setData: (data: { min: number; max: number; data: HeatmapDataPoint[] }) => void
    }
  }

  const L: typeof LeafletNS
}

interface HeatmapInstance {
  setData: (data: { min: number; max: number; data: unknown[] }) => void
  addData: (data: unknown) => void
  repaint: () => void
}

/* ── WebSocket Payload Types ───────────────────────────────────────────── */

export interface HeatmapDataPoint {
  lat: number
  lon: number
  value: number
}

export interface WorkerPosition {
  lat: number
  lon: number
  ppe_ok: boolean
  has_helmet: boolean
  has_vest: boolean
  zone: string
  risk: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
  height_m: number
  /* ── Intelligence v2 fields ──────────────────────────────────────── */
  worker_id?: number
  status?: WorkerStatusType
  dwell_time_s?: number
  dwell_severity?: DwellSeverity
  camera_id?: string
}

export interface RiskCell {
  lat: number
  lon: number
  risk_score: number
  risk_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
  worker_count: number
  ppe_violations: number
}

export interface GeoAIAlert {
  alert_level: 'ADVISORY' | 'WARNING' | 'CRITICAL'
  alert_type: string
  alert_message: string
  bocw_reference: string
  worker_count: number
  ppe_violations: number
  risk_score: number
  timestamp: string
}

export interface SiteStats {
  total_workers: number
  ppe_compliant: number
  max_risk_score: number
  critical_zones: number
}

export interface HeatmapUpdatePayload {
  type: 'heatmap_update'
  cycle: number
  workers: WorkerPosition[]
  cells: RiskCell[]
  alerts: GeoAIAlert[]
  site_stats: SiteStats
  /* ── Intelligence v2 additions ───────────────────────────────────── */
  events?: SpatialEvent[]
  trails?: WorkerTrail[]
  kpi?: KPISummary
  backend_health?: BackendHealth[]
}

/* ── Intelligence v2 Types ─────────────────────────────────────────────── */

export type EventPriority = 'INFO' | 'WARNING' | 'CRITICAL'
export type WorkerStatusType = 'SAFE' | 'AT_RISK' | 'CRITICAL'
export type DwellSeverity = 'NONE' | 'LOW' | 'MEDIUM' | 'HIGH'
export type AlertStateType = 'ACTIVE' | 'ACKNOWLEDGED' | 'ESCALATED' | 'RESOLVED' | 'EXPIRED'
export type ShiftType = 'MORNING' | 'AFTERNOON' | 'NIGHT'

export type GeofenceBreachType =
  | 'RESTRICTED_ENTRY'
  | 'REVERSE_DIRECTION'
  | 'PROLONGED_PRESENCE'
  | 'REPEAT_VIOLATION'

export type EventTypeEnum =
  | 'ZONE_ENTRY'
  | 'ZONE_EXIT'
  | 'PPE_VIOLATION'
  | 'DWELL_BREACH'
  | 'GEOFENCE_BREACH'
  | 'STATUS_CHANGE'
  | 'ALERT_ESCALATED'

export interface SpatialEvent {
  event_id: string
  event_type: EventTypeEnum
  priority: EventPriority
  timestamp: string
  worker_id?: number
  zone: string
  camera_id: string
  message: string
  details: Record<string, unknown>
  state: AlertStateType
  acknowledged_by?: string
  acknowledged_at?: string
  resolved_by?: string
  resolved_at?: string
  sla_remaining_s?: number
}

export interface WorkerTrail {
  worker_id: number
  camera_id: string
  positions: TrailPoint[]
  current_zone: string
  status: WorkerStatusType
}

export interface TrailPoint {
  lat: number
  lon: number
  world_x: number
  world_y: number
  timestamp: number
  status: WorkerStatusType
  dwell_severity: DwellSeverity
}

export interface KPISummary {
  active_workers: number
  critical_alerts: number
  escalated_alerts: number
  total_events: number
  ppe_compliance: number
  avg_site_risk: number
  system_degraded: boolean
  mapper_status: string
}

export interface BackendHealth {
  service: string
  status: 'HEALTHY' | 'DEGRADED' | 'UNAVAILABLE' | string
  buffer_workers?: number
  buffer_events?: number
}

/* ── GeoJSON Zone Features ─────────────────────────────────────────────── */

export interface ZoneProperties {
  zone: string
  risk?: string
  alert?: boolean
  alert_level?: string
  description?: string
  z_base_m?: number
  z_top_m?: number
  height_m?: number
  type?: string
  camera_id?: string
  camera_height_m?: number
  fov_horizontal_deg?: number
  [key: string]: unknown
}

export interface ZoneFeature {
  type: 'Feature'
  properties: ZoneProperties
  geometry: {
    type: 'Polygon' | 'MultiPolygon' | 'Point'
    coordinates: number[][][] | number[][] | number[]
  }
}

export interface ZoneCollection {
  type: 'FeatureCollection'
  features: ZoneFeature[]
}

/* ── GeoAI View Modes ──────────────────────────────────────────────────── */

export type GeoAIMode = 'HEATMAP' | '3D ZONES' | 'TRACKING'
export type GeoAIVisualMode = 'tactical' | 'satellite' | 'plan2d' | 'view3d'

export {}
