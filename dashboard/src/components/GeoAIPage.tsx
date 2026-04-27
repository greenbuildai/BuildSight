import { useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GeoAIMap } from './GeoAIMap'
import { GeoAI3DView } from './GeoAI3DView'
import { GeoAIPlan2D } from './GeoAIPlan2D'
import { GeoAIHUD } from './GeoAIHUD'
import { DynamicZoneEditor } from './DynamicZoneEditor'
import { useGeoAIWebSocket } from '../hooks/useGeoAIWebSocket'
import type { GeoAIMode, GeoAIVisualMode, DynamicZone } from '../types/geoai'
import { useSettings } from '../SettingsContext'
import { useDetectionStore } from '../store/detectionStore'
import './GeoAIPage.css'

function isOpticalVlmSource(source: string | undefined): boolean {
  return source === 'florence2' || source === 'moondream2' || source === 'vlm_chained_with_turner_ai'
}

const MODE_META: Record<GeoAIMode, { label: string; detail: string; badge: string }> = {
  HEATMAP: {
    label: 'Live risk density',
    detail: 'KDE-driven risk scoring across worker clusters, PPE loss, and zone pressure.',
    badge: 'Spatial risk surface',
  },
  '3D ZONES': {
    label: 'Vertical exposure model',
    detail: 'Height-aware zone analysis for slab edges, parapets, and scaffold interfaces.',
    badge: '3D compliance shell',
  },
  TRACKING: {
    label: 'Movement intelligence',
    detail: 'Worker path monitoring with status changes, dwell time, and rapid breach detection.',
    badge: 'Trajectory telemetry',
  },
}

const VISUAL_MODES: { id: GeoAIVisualMode; label: string; icon: string }[] = [
  { id: 'tactical', label: 'Tactical', icon: '📍' },
  { id: 'satellite', label: 'Satellite', icon: '🛰️' },
  { id: 'plan2d', label: '2D Plan', icon: '📐' },
  { id: 'view3d', label: '3D', icon: '🧊' }
]

const LS_MAP_MODE = 'buildsight_geoai_mode'
const LS_VISUAL_MODE = 'buildsight_geoai_visual_mode'
const LS_SHOW_ZONES = 'buildsight_geoai_zones'
const LS_SHOW_LABELS = 'buildsight_geoai_labels'
const LS_SHOW_FOV = 'buildsight_geoai_fov'
const LS_HEATMAP_OPACITY = 'buildsight_geoai_hm_opacity'

function loadLS<T>(key: string, fallback: T): T {
  try {
    const value = localStorage.getItem(key)
    return value !== null ? JSON.parse(value) : fallback
  } catch {
    return fallback
  }
}

function AnimatedCounter({ value }: { value: number | string }) {
  return (
    <motion.div
      key={value}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.2 }}
    >
      {value}
    </motion.div>
  )
}

export function GeoAIPage() {
  const { settings } = useSettings()
  const { data, narration, connectionState, cycle, acknowledgeEvent, resolveEvent } = useGeoAIWebSocket()

  // Background detection state — persists across tab navigation
  const detectionWorkers    = useDetectionStore(s => s.workerPositions)
  const detectionViolations = useDetectionStore(s => s.violations)
  const bgWorkerCount       = useDetectionStore(s => s.workerCount)
  const bgIsRunning         = useDetectionStore(s => s.isRunning)
  const bgCompliance        = useDetectionStore(s => s.workerPositions.length > 0
    ? Math.round((s.workerPositions.filter(w => w.ppe_compliant).length / s.workerPositions.length) * 100)
    : 100)
  const requestSnapshot     = useDetectionStore(s => s.requestSnapshot)

  // True when inference is running (either via background WebSocket or manual upload).
  // Used to override the GeoAI spatial-engine demo mode display.
  const detectionIsLive = bgIsRunning

  // When GeoAI tab becomes active, request a fresh snapshot so markers
  // populate immediately even if detection started on another tab.
  useEffect(() => { requestSnapshot() }, [])

  const [dynamicZones, setDynamicZones] = useState<DynamicZone[]>([])
  const [isZoneFormOpen, setIsZoneFormOpen] = useState(false)

  // ── VLM Scene Narration ─────────────────────────────────────────────────────
  const [vlmEntry, setVlmEntry] = useState<{ description: string; source: string; timestamp: number; question: string; vlm_available: boolean } | null>(null)
  const [vlmLoading, setVlmLoading] = useState(false)
  const [vlmQuestion, setVlmQuestion] = useState('')

  const fetchVlm = (question?: string) => {
    setVlmLoading(true)
    const req = question
      ? fetch('http://localhost:8000/api/geoai/vlm/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, force_refresh: true }),
        })
      : fetch('http://localhost:8000/api/geoai/vlm/latest')
    req
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setVlmEntry(d) })
      .catch(() => {})
      .finally(() => setVlmLoading(false))
  }

  useEffect(() => {
    fetchVlm()
    const id = setInterval(() => fetchVlm(), 30_000)
    return () => clearInterval(id)
  }, [])

  // Load zones from backend on mount so map always shows persisted zones
  useEffect(() => {
    const load = () =>
      fetch('http://localhost:8000/api/geoai/dynamic-zones')
        .then(r => r.ok ? r.json() : [])
        .then((zones: DynamicZone[]) => setDynamicZones(zones))
        .catch(() => {})
    load()
    const id = setInterval(load, 30_000)
    return () => clearInterval(id)
  }, [])

  const [mode] = useState<GeoAIMode>(() => loadLS(LS_MAP_MODE, 'HEATMAP'))
  const [visualMode, setVisualMode] = useState<GeoAIVisualMode>(() => {
    const val = loadLS(LS_VISUAL_MODE, 'tactical')
    return ['tactical', 'satellite', 'plan2d', 'view3d'].includes(val) ? val : 'tactical'
  })

  const [showZones, setShowZones] = useState(() => loadLS(LS_SHOW_ZONES, true))
  const [showLabels, setShowLabels] = useState(() => loadLS(LS_SHOW_LABELS, true))
  const [showCameraFOV] = useState(() => loadLS(LS_SHOW_FOV, true))
  const [heatmapOpacity, setHeatmapOpacity] = useState(() => loadLS(LS_HEATMAP_OPACITY, 0.65))

  const [isTransitioning, setIsTransitioning] = useState(false)

  useEffect(() => { localStorage.setItem(LS_MAP_MODE, JSON.stringify(mode)) }, [mode])
  useEffect(() => { localStorage.setItem(LS_VISUAL_MODE, JSON.stringify(visualMode)) }, [visualMode])
  useEffect(() => { localStorage.setItem(LS_SHOW_ZONES, JSON.stringify(showZones)) }, [showZones])
  useEffect(() => { localStorage.setItem(LS_SHOW_LABELS, JSON.stringify(showLabels)) }, [showLabels])
  useEffect(() => { localStorage.setItem(LS_SHOW_FOV, JSON.stringify(showCameraFOV)) }, [showCameraFOV])
  useEffect(() => { localStorage.setItem(LS_HEATMAP_OPACITY, JSON.stringify(heatmapOpacity)) }, [heatmapOpacity])


  const showHeatmap = mode === 'HEATMAP'
  const showWorkers = mode === 'TRACKING' || mode === 'HEATMAP'

  // When detection is live, override the spatial engine's demo posture so the
  // operator sees a coherent "live" status rather than "demo / degraded".
  const isDegraded = detectionIsLive ? false : (data?.kpi?.system_degraded || connectionState === 'demo')

  // Resolved connection label: prefer detection live status over spatial engine state
  const resolvedConnectionState = detectionIsLive ? 'live' : connectionState

  // Prefer live detection values when bg service is running
  const activeWorkers  = bgWorkerCount > 0
    ? bgWorkerCount
    : (data?.kpi?.active_workers ?? data?.site_stats?.total_workers ?? 0)
  const criticalAlerts = detectionViolations.filter(v => v.severity === 'CRITICAL').length
    || data?.kpi?.critical_alerts
    || 0
  const compliance = bgWorkerCount > 0
    ? bgCompliance
    : (data?.kpi?.ppe_compliance ?? 100)
  const avgRisk = data?.kpi?.avg_site_risk ?? 0
  const eventCount = data?.events?.length ?? 0
  const siteStatusLabel = isDegraded ? 'Degraded' : 'Operational'


  const activeStatusLabel = useMemo(() => {
    if (visualMode === 'tactical') return 'Tactical Overview Active'
    if (visualMode === 'satellite') return 'Satellite Hybrid Active'
    if (visualMode === 'plan2d') return '2D Plan Active'
    return '3D View Active'
  }, [visualMode])

  const statusData = useMemo(() => {
    const workers = data?.workers ?? []
    const events  = data?.events  ?? []

    // Highest-risk worker on site right now
    const topWorker = workers.reduce<typeof workers[number] | null>((best, w) => {
      if (!best) return w
      const riskRank = (r: string) => r === 'CRITICAL' ? 3 : r === 'HIGH' ? 2 : r === 'MODERATE' ? 1 : 0
      return riskRank(w.risk) > riskRank(best.risk) ? w : best
    }, null)

    // Most recent critical/warning event
    const topEvent = events.find(e => e.priority === 'CRITICAL') ??
                     events.find(e => e.priority === 'WARNING') ??
                     events[0] ?? null

    const postgisHealth = data?.backend_health?.find(h => h.service === 'PostGIS')
    const mapperStatus = postgisHealth
      ? (postgisHealth.status === 'HEALTHY' ? 'PostGIS Connected' : 'PostGIS Degraded')
      : 'Demo Fallback'

    return {
      modeMeta: MODE_META[mode],
      topWorker,
      topEvent,
      mapperStatus,
    }
  }, [data, mode])

  const handleVisualModeChange = (nextMode: GeoAIVisualMode) => {
    if (nextMode === visualMode) return
    setIsTransitioning(true)
    setTimeout(() => {
      setVisualMode(nextMode)
      setTimeout(() => setIsTransitioning(false), 300)
    }, 200)
  }

  return (
    <div className="geoai-page geoai-page--active" data-theme={settings.theme}>
      <div className="hud-scanline" />
      <div className="scanline-sweep" />
      <div className="geoai-page__glow geoai-page__glow--left" />
      <div className="geoai-page__glow geoai-page__glow--right" />

      <section className="geoai-hero">
        <div className="geoai-hero__intro">
          <p className="geoai-eyebrow">Spatial Intelligence Command Center</p>
          <h1>GeoAI Mission Control</h1>
          <p className="geoai-hero__lede">
            Live worker telemetry, risk zoning, and height-aware site context for construction safety decisions.
          </p>

          <div className="geoai-hero__pills">
            <span className={`geoai-status-chip geoai-status-chip--${resolvedConnectionState}`}>
              {resolvedConnectionState === 'live'
                ? (detectionIsLive ? 'Detection Live' : 'Engine Live')
                : resolvedConnectionState}
            </span>
            <span className={`geoai-status-chip ${isDegraded ? 'geoai-status-chip--degraded' : 'geoai-status-chip--healthy'}`}>
              {isDegraded ? 'Fallback Posture' : 'Full Fidelity'}
            </span>
            <span className="geoai-status-chip">
              {detectionIsLive
                ? `${bgWorkerCount} worker${bgWorkerCount !== 1 ? 's' : ''} detected`
                : cycle ? `Cycle ${cycle} synced` : 'Waiting for stream'}
            </span>
          </div>
        </div>

        <div className="geoai-hero__sidebar">
          <div className={`geoai-kpi-item geoai-kpi-item--status ${isDegraded ? 'geoai-kpi-item--degraded' : ''}`}>
            <span className="geoai-kpi-val geoai-kpi-val--status">
              <span className="geoai-status-pulse" />
              {siteStatusLabel}
            </span>
            <span className="geoai-kpi-label">Site Status</span>
          </div>
          <div className="geoai-kpi-item">
            <span className="geoai-kpi-val"><AnimatedCounter value={activeWorkers} /></span>
            <span className="geoai-kpi-label">Active Workers</span>
          </div>
          <div className="geoai-kpi-item geoai-kpi-item--critical">
            <span className="geoai-kpi-val"><AnimatedCounter value={criticalAlerts} /></span>
            <span className="geoai-kpi-label">Critical Risks</span>
          </div>
          <div className="geoai-kpi-item">
            <span className="geoai-kpi-val" style={{ color: compliance >= 80 ? 'var(--color-safe)' : 'var(--color-warning)' }}>
              <AnimatedCounter value={`${compliance.toFixed(0)}%`} />
            </span>
            <span className="geoai-kpi-label">PPE Compliance</span>
          </div>
          <div className="geoai-kpi-item">
            <span className="geoai-kpi-val" style={{ color: avgRisk > 0.5 ? 'var(--color-danger)' : 'var(--color-safe)' }}>
              <AnimatedCounter value={avgRisk.toFixed(2)} />
            </span>
            <span className="geoai-kpi-label">Indexed Risk</span>
          </div>
          <div className="geoai-kpi-item">
            <span className="geoai-kpi-val"><AnimatedCounter value={eventCount} /></span>
            <span className="geoai-kpi-label">Events</span>
          </div>
        </div>
      </section>


      {/* RENDER STAGE + HUD */}
      <motion.div
        className="geoai-layout"
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
      >
        <div className="geoai-layout__map">
          <div className={`geoai-map-transition-layer ${isTransitioning ? 'geoai-map-transition-layer--active' : ''}`} />

          {/* Floating Switcher Controls */}
          <div className="geoai-floating-controls">
            <div className="geoai-floating-switcher">
              {VISUAL_MODES.map((v, index) => (
                <div key={v.id} style={{ display: 'flex', alignItems: 'center' }}>
                  <button
                    className={`geoai-floating-btn ${visualMode === v.id ? 'geoai-floating-btn--active' : ''}`}
                    onClick={() => handleVisualModeChange(v.id)}
                  >
                    <span className="geoai-floating-icon">{v.icon}</span>
                    <span className="geoai-floating-lbl">{v.label}</span>
                  </button>
                  {index < VISUAL_MODES.length - 1 && <div className="geoai-floating-divider" />}
                </div>
              ))}
            </div>
            <div className="geoai-mode-status">
              <span className="geoai-status-pulse" style={{ width: 6, height: 6 }} />
              {activeStatusLabel}
            </div>
          </div>

          <div className="geoai-map-shell">
            <div className="geoai-map-shell__toolbar">
              <div className="geoai-map-shell__title">
                <span className="geoai-map-shell__eyebrow">Primary scene</span>
                <strong>
                  {visualMode === 'tactical' ? 'Tactical Overview'
                    : visualMode === 'satellite' ? 'Satellite Overview'
                      : visualMode === 'plan2d' ? '2D CAD Plan'
                        : '3D Volumetric Scene'}
                </strong>
              </div>

              <div className="geoai-map-shell__readouts">
                <button 
                  className={`btn-create-zone-top ${isZoneFormOpen ? 'active' : ''}`}
                  onClick={() => setIsZoneFormOpen(!isZoneFormOpen)}
                >
                  <span className="icon">{isZoneFormOpen ? '✕' : '+'}</span>
                  <span>{isZoneFormOpen ? 'Cancel Zone' : 'Create Zone'}</span>
                </button>
                <div className="v-divider" />
                <span>{MODE_META[mode].label}</span>
                <span>{showLabels ? 'Labels ON' : 'Labels OFF'}</span>
                {(visualMode === 'tactical' || visualMode === 'satellite') && <span>Heat {Math.round(heatmapOpacity * 100)}%</span>}
              </div>
            </div>

            <div className="geoai-map-shell__stage">
              <AnimatePresence mode="wait">
                {visualMode === 'view3d' ? (
                  <motion.div key="view3d" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ height: '100%', width: '100%' }}>
                    <GeoAI3DView data={data} showLabels={showLabels} showCameraFOV={showCameraFOV} onReturn2D={() => handleVisualModeChange('tactical')} />
                  </motion.div>
                ) : visualMode === 'plan2d' ? (
                  <motion.div key="plan2d" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ height: '100%', width: '100%' }}>
                    <GeoAIPlan2D showLabels={showLabels} />
                  </motion.div>
                ) : (
                  <motion.div key="map2d" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ height: '100%', width: '100%' }}>
                    <GeoAIMap
                      data={data}
                      showZones={showZones}
                      showLabels={showLabels}
                      showCameraFOV={showCameraFOV}
                      showHeatmap={showHeatmap}
                      showWorkers={showWorkers}
                      heatmapOpacity={heatmapOpacity}
                      viewMode={visualMode}
                      dynamicZones={dynamicZones}
                      narration={narration}
                      theme={settings.theme as 'light' | 'dark'}
                      detectionWorkers={detectionWorkers}
                      detectionViolations={detectionViolations}
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="geoai-map-legend">
              <div className="geoai-map-legend__item">
                <span className="geoai-map-legend__swatch geoai-map-legend__swatch--critical" />
                Critical exposure
              </div>
              <div className="geoai-map-legend__item">
                <span className="geoai-map-legend__swatch geoai-map-legend__swatch--warning" />
                Elevated risk
              </div>
              <div className="geoai-map-legend__item">
                <span className="geoai-map-legend__swatch geoai-map-legend__swatch--safe" />
                Compliant activity
              </div>
            </div>
          </div>
        </div>

        <div className="geoai-layout__hud">
          <div className="geoai-hud-shell">
            <div className="geoai-command-card" style={{ padding: '1rem', borderBottom: '1px solid var(--color-border)', background: 'rgba(10,12,16,0.6)' }}>
              <p className="geoai-command-card__eyebrow" style={{ marginBottom: '0.6rem' }}>Map overlays</p>
              <div className="geoai-control-grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) auto auto', gap: '0.8rem', alignItems: 'center' }}>
                <label className="geoai-slider-card" style={{ minHeight: 'auto', padding: '0.5rem', background: 'transparent', border: 'none' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span className="geoai-slider-card__label">Heat opacity</span>
                    <strong>{Math.round(heatmapOpacity * 100)}%</strong>
                  </div>
                  <input
                    type="range"
                    min={0.2} max={0.95} step={0.05}
                    value={heatmapOpacity}
                    onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                    disabled={visualMode === 'view3d'}
                  />
                </label>
                <div style={{ display: 'flex', gap: '1rem' }}>
                  <label className="geoai-toggle-card" style={{ minHeight: 'auto', padding: 0, border: 'none', background: 'transparent' }}>
                    <input type="checkbox" checked={showZones} onChange={e => setShowZones(e.target.checked)} />
                    <span>Zones</span>
                  </label>
                  <label className="geoai-toggle-card" style={{ minHeight: 'auto', padding: 0, border: 'none', background: 'transparent' }}>
                    <input type="checkbox" checked={showLabels} onChange={e => setShowLabels(e.target.checked)} />
                    <span>Labels</span>
                  </label>
                </div>
              </div>
            </div>

            {/* ── SAM Zone Intelligence — promoted to top for easy access ── */}
            <div className="geoai-command-card" style={{ padding: '1rem', borderBottom: '1px solid var(--color-border)', background: 'rgba(10,12,16,0.8)' }}>
              <DynamicZoneEditor
                onZonesChange={setDynamicZones}
                isFormOpen={isZoneFormOpen}
                onFormToggle={setIsZoneFormOpen}
              />
            </div>

            <GeoAIHUD
              data={data}
              connectionState={resolvedConnectionState}
              cycle={cycle}
              activeMode={visualMode}
              onAcknowledge={acknowledgeEvent}
              onResolve={resolveEvent}
              statusData={statusData}
            />
          </div>
        </div>
      </motion.div>
    </div>
  )
}
