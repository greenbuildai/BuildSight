import {
  useRef, useState, useEffect, useCallback, useMemo,
} from 'react'
import { useDetectionStats } from '../DetectionStatsContext'
import { useSettings } from '../SettingsContext'
import './DetectionPanel.css'

const API = 'http://localhost:8000/api'

const CONDITIONS = ['S1_normal', 'S2_dusty', 'S3_low_light', 'S4_crowded'] as const
type Condition = typeof CONDITIONS[number]

interface Detection {
  class: string
  confidence: number
  box: [number, number, number, number]
  /** Only present on worker/person detections — set by backend association */
  has_helmet?: boolean
  has_vest?: boolean
}

interface DetectionHeatmapPoint {
  x: number
  y: number
  value: number
  type: 'worker' | 'violation'
  risk_level?: RiskLevel
}

type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'

interface DetectionRiskZone {
  box: [number, number, number, number]
  center: [number, number]
  risk_score: number
  risk_level: RiskLevel
  label: string
  worker_count: number
  ppe_violations: number
}

interface DetectionHeatmapPayload {
  frame_width?: number
  frame_height?: number
  points: DetectionHeatmapPoint[]
  zones: DetectionRiskZone[]
  summary: {
    worker_count: number
    risk_zone_count: number
    critical_zone_count: number
    max_risk_score: number
  }
}

interface DetectResult {
  detections: Detection[]
  class_counts: Record<string, number>
  total: number
  image_b64: string
  mode: string
  condition: Condition
  elapsed_ms: number
  heatmap?: DetectionHeatmapPayload
}

const CLASS_COLORS: Record<string, string> = {
  helmet: '#00ff55', // Green
  safety_vest: '#ffdd00', // Yellow
  worker: '#0088ff', // Blue
  person: '#0088ff', // Blue (same as worker)
  'safety-vest': '#ffdd00', // Yellow alias
}

/** 
 * Helper to resolve CSS variables like 'var(--color-accent)' to hex/rgb 
 * since Canvas fillStyle doesn't support them natively.
 */
function resolveCssVar(varName: string): string {
  if (!varName.startsWith('var(')) return varName
  const name = varName.match(/\((--[^)]+)\)/)?.[1]
  if (!name) return '#ff9900' // Fail-safe fallback
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#ff9900'
}

// ── Lightweight IoU tracker (module-level, no React state) ─────────────────────

let _nextTrackId = 0

interface Track {
  id: number
  cls: string
  confidence: number
  /** Bounding box in the sent-frame's pixel space (e.g. 640×360) */
  frameBox: [number, number, number, number]
  /** Smoothed display coords on the overlay canvas (lerped each RAF tick) */
  sx1: number; sy1: number; sx2: number; sy2: number
  initialized: boolean
  /** How many consecutive inference cycles had no match */
  missed: number
  /** PPE association flags — only set on worker/person tracks */
  has_helmet?: boolean
  has_vest?: boolean
}

interface HeatmapPoint {
  x: number
  y: number
  time: number
  type: 'worker' | 'violation'
  value: number
  riskLevel?: RiskLevel
}

interface PeakRiskMoment {
  time: number
  score: number
  type: string
}

function PeakRiskMomentsPanel({
  moments,
  onSelect,
  emptyLabel,
  className = '',
}: {
  moments: PeakRiskMoment[]
  onSelect?: (moment: PeakRiskMoment) => void
  emptyLabel: string
  className?: string
}) {
  return (
    <div className={`det-risk-sidebar ${className}`.trim()}>
      <div className="det-risk-sidebar__header">
        <span className="det-risk-sidebar__icon">⚠️</span>
        PEAK RISK MOMENTS
      </div>
      <div className="det-risk-sidebar__list">
        {moments.length === 0 ? (
          <div className="det-risk-sidebar__empty">{emptyLabel}</div>
        ) : (
          moments.map((moment, idx) => {
            const content = (
              <>
                <div className="det-risk-item__time">
                  {onSelect ? _fmtTime(moment.time) : `LIVE CAPTURE #${idx + 1}`}
                </div>
                <div className="det-risk-item__meta">
                  <span className="det-risk-item__label">{moment.type}</span>
                  <span className="det-risk-item__score">
                    {onSelect ? `Risk Level: ${moment.score}` : `RISK: ${moment.score}`}
                  </span>
                </div>
              </>
            )

            if (!onSelect) {
              return (
                <div key={idx} className="det-risk-item det-risk-item--static">
                  {content}
                </div>
              )
            }

            return (
              <button
                key={idx}
                className="det-risk-item"
                onClick={() => onSelect(moment)}
              >
                {content}
              </button>
            )
          })
        )}
      </div>
    </div>
  )
}

function _boxIou(a: number[], b: number[]): number {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1])
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3])
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  if (inter === 0) return 0
  const ua = (a[2] - a[0]) * (a[3] - a[1])
  const ub = (b[2] - b[0]) * (b[3] - b[1])
  return inter / Math.max(ua + ub - inter, 1e-6)
}

function mergeTracks(tracks: Track[], newDets: Detection[]): Track[] {
  const matched = new Set<number>()

  for (const t of tracks) {
    let bestIou = 0.25, bestIdx = -1
    for (let i = 0; i < newDets.length; i++) {
      if (newDets[i].class !== t.cls) continue
      const iou = _boxIou(t.frameBox as number[], newDets[i].box as number[])
      if (iou > bestIou) { bestIou = iou; bestIdx = i }
    }
    if (bestIdx >= 0) {
      const nd = newDets[bestIdx]
      t.frameBox = nd.box
      t.confidence = nd.confidence
      t.missed = 0
      // Carry forward PPE association flags from the latest detection
      t.has_helmet = nd.has_helmet
      t.has_vest = nd.has_vest
      matched.add(bestIdx)
    } else {
      t.missed++
    }
  }

  const alive = tracks.filter(t => t.missed <= 2)
  newDets.forEach((d, i) => {
    if (matched.has(i)) return
    alive.push({
      id: _nextTrackId++, cls: d.class, confidence: d.confidence,
      frameBox: d.box,
      sx1: 0, sy1: 0, sx2: 0, sy2: 0,
      initialized: false, missed: 0,
      has_helmet: d.has_helmet,
      has_vest: d.has_vest,
    })
  })
  return alive
}

/** Compute letterbox offsets for object-fit:contain video rendering */
function _letterbox(vw: number, vh: number, dw: number, dh: number) {
  if (!vw || !vh) return { rw: dw, rh: dh, ox: 0, oy: 0 }
  const va = vw / vh, da = dw / dh
  return va > da
    ? { rw: dw, rh: dw / va, ox: 0, oy: (dh - dw / va) / 2 }
    : { rw: dh * va, rh: dh, ox: (dw - dh * va) / 2, oy: 0 }
}

function _fmtTime(s: number): string {
  const m = Math.floor(s / 60)
  return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`
}

function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  points: HeatmapPoint[],
  now: number,
  rw: number,
  rh: number,
  ox: number,
  oy: number,
  ttlMs = 1500,
  radius = 24,
  maxOpacity = 0.06
): HeatmapPoint[] {
  const activePoints = points.filter(point => now - point.time < ttlMs)
  if (activePoints.length === 0) return activePoints

  ctx.save()
  ctx.globalCompositeOperation = 'source-over'

  activePoints.forEach(point => {
    const hx = point.x * rw + ox
    const hy = point.y * rh + oy
    const age = (now - point.time) / ttlMs
    const intensity = Math.max(0.15, Math.min(1, point.value || 0.25))
    const opacity = Math.max(0, (1 - age) * maxOpacity * (0.5 + intensity * 0.5))
    const scaledRadius = radius * (0.6 + intensity * 0.4)
    const gradient = ctx.createRadialGradient(hx, hy, 0, hx, hy, scaledRadius)
    const color = point.riskLevel === 'CRITICAL'
      ? '255, 42, 42'
      : point.riskLevel === 'HIGH'
        ? '255, 140, 0'
        : point.riskLevel === 'MODERATE'
          ? '255, 214, 0'
          : point.type === 'violation'
            ? '255, 96, 60'
            : '0, 255, 136'

    gradient.addColorStop(0, `rgba(${color}, ${opacity})`)
    gradient.addColorStop(1, `rgba(${color}, 0)`)

    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(hx, hy, scaledRadius, 0, Math.PI * 2)
    ctx.fill()
  })

  ctx.restore()
  return activePoints
}

function drawRiskZones(
  ctx: CanvasRenderingContext2D,
  zones: DetectionRiskZone[],
  fw: number,
  fh: number,
  rw: number,
  rh: number,
  ox: number,
  oy: number,
) {
  if (!zones.length || !fw || !fh) return

  ctx.save()
  zones.forEach(zone => {
    if (zone.risk_score < 0.4) return
    const [x1, y1, x2, y2] = zone.box
    const sx1 = (x1 / fw) * rw + ox
    const sy1 = (y1 / fh) * rh + oy
    const sx2 = (x2 / fw) * rw + ox
    const sy2 = (y2 / fh) * rh + oy
    const color = zone.risk_level === 'CRITICAL'
      ? '255, 42, 42'
      : zone.risk_level === 'HIGH'
        ? '255, 140, 0'
        : '255, 214, 0'

    // Stroke-only outline — no filled rectangle to avoid haze
    ctx.strokeStyle = `rgba(${color}, ${0.35 + zone.risk_score * 0.3})`
    ctx.lineWidth = 1.5
    ctx.setLineDash([5, 4])
    ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1)
    ctx.setLineDash([])

    const label = `${zone.risk_level} ${(zone.risk_score * 100).toFixed(0)}%`
    ctx.font = 'bold 10px monospace'
    const tw = ctx.measureText(label).width
    ctx.fillStyle = `rgba(${color}, 0.85)`
    ctx.fillRect(sx1, Math.max(0, sy1 - 16), tw + 8, 16)
    ctx.fillStyle = '#000'
    ctx.fillText(label, sx1 + 4, Math.max(10, sy1 - 4))
  })
  ctx.restore()
}

function pointsFromHeatmap(payload: DetectionHeatmapPayload | undefined, now: number): HeatmapPoint[] {
  if (!payload?.points?.length) return []
  return payload.points.map(point => ({
    x: point.x,
    y: point.y,
    time: now,
    type: point.type,
    value: point.value,
    riskLevel: point.risk_level,
  }))
}

// ── Shared condition picker ─────────────────────────────────────────────────────
const CONDITION_COLORS: Record<string, string> = {
  S1_normal:    '#00cc66',
  S2_dusty:     '#ffaa00',
  S3_low_light: '#8866ff',
  S4_crowded:   '#ff4444',
  S1_NORMAL:    '#00cc66',
  S2_DUSTY:     '#ffaa00',
  S3_LOW_LIGHT: '#8866ff',
  S4_CROWDED:   '#ff4444',
}

function SceneAutoIndicator({
  condition, autoMode, onToggleAuto, onManualChange,
}: {
  condition: string
  autoMode: boolean
  onToggleAuto: () => void
  onManualChange: (c: Condition) => void
}) {
  const color = CONDITION_COLORS[condition] ?? '#aaaaaa'
  const label = condition.replace(/_/g, ' ').toUpperCase()
  return (
    <div className="det-scene-indicator">
      <div className="det-scene-indicator__header">
        <p className="section-label" style={{ marginBottom: 0 }}>SCENE DETECTION</p>
        <button
          className={`det-scene-indicator__toggle${autoMode ? ' det-scene-indicator__toggle--on' : ''}`}
          onClick={onToggleAuto}
          title={autoMode ? 'Switch to manual condition selection' : 'Enable automatic scene classification'}
        >
          {autoMode ? 'AUTO' : 'MANUAL'}
        </button>
      </div>
      {autoMode ? (
        <div className="det-scene-indicator__badge" style={{ borderColor: color, color }}>
          <span className="det-scene-indicator__dot" style={{ background: color }} />
          {label}
        </div>
      ) : (
        <ConditionPicker value={condition as Condition} onChange={onManualChange} />
      )}
    </div>
  )
}

function ConditionPicker({
  value, onChange,
}: { value: Condition; onChange: (c: Condition) => void }) {
  return (
    <div className="det-condition">
      <p className="section-label" style={{ marginBottom: '0.75rem' }}>ENVIRONMENT CONDITIONS</p>
      <div className="det-condition__chips">
        {CONDITIONS.map(c => (
          <button
            key={c}
            className={`site-chip ${value === c ? 'site-chip--active' : ''}`}
            onClick={() => onChange(c)}
          >
            {c.replace(/_/g, ' ')}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Detection sidebar (class counts + list) ────────────────────────────────────
function DetectionSidebar({
  detections, elapsed, mode, extra, renderControls,
}: {
  detections: Detection[]
  elapsed?: number
  mode?: string
  extra?: React.ReactNode
  renderControls?: React.ReactNode
}) {
  const counts: Record<string, number> = {}
  detections.forEach(d => { counts[d.class] = (counts[d.class] ?? 0) + 1 })

  return (
    <div className="det-results__panel">
      {renderControls && (
        <div className="det-sidebar-controls">
          {renderControls}
        </div>
      )}

      {/* ── Summary Metrics Card ── */}
      <div className="det-summary-card">
        <div className="det-summary-item">
          <span className="det-summary-label">TOTAL DETECTIONS</span>
          <span className="det-summary-value">{detections.length}</span>
        </div>
        <div className="det-summary-sep" />
        <div className="det-summary-item">
          <span className="det-summary-label">LATENCY</span>
          <span className="det-summary-value">{elapsed !== undefined ? `${elapsed}ms` : '--'}</span>
        </div>
        {mode && (
          <>
            <div className="det-summary-sep" />
            <div className="det-summary-item">
              <span className="det-summary-label">MODEL</span>
              <span className="det-summary-value">{mode.toUpperCase()}</span>
            </div>
          </>
        )}
      </div>

      {/* ── Class Breakdown (Pills) ── */}
      <div className="det-class-pills">
        {Object.entries(counts).map(([cls, count]) => (
          <div key={cls} className="det-class-pill">
            <span className="det-class-swatch" style={{ background: CLASS_COLORS[cls] ?? '#aaa' }} />
            <span className="det-class-name">{cls}</span>
            <span className="det-class-count">{count}</span>
          </div>
        ))}
        {detections.length === 0 && (
          <p className="det-hint" style={{ opacity: 0.5, fontSize: 'var(--fs-xs)' }}>
            Waiting for inference...
          </p>
        )}
      </div>

      {/* ── Individual Detection List (Scrollable) ── */}
      <div className="det-det-header">
        <p className="section-label">LIVE DETECTIONS</p>
      </div>
      <div className="det-det-list">
        {detections.map((d, i) => {
          const isWorker = d.class === 'worker' || d.class === 'person';
          const hasPpeIssue = isWorker && (!d.has_helmet || !d.has_vest);

          return (
            <div key={i} className={`det-det-row ${hasPpeIssue ? 'det-det-row--alert' : ''}`}>
              <span className="det-class-swatch" style={{ background: CLASS_COLORS[d.class] ?? '#aaa' }} />
              <div className="det-det-info">
                <span className="det-det-class">{d.class}</span>
                {isWorker && (
                  <div className="det-ppe-badges">
                    <span
                      className={`det-ppe-badge ${d.has_helmet ? 'det-ppe-badge--ok' : 'det-ppe-badge--missing'}`}
                      title={d.has_helmet ? "Helmet: Pass" : "Helmet: MISSING"}
                    >
                      {d.has_helmet ? 'H' : 'H'}
                    </span>
                    <span
                      className={`det-ppe-badge ${d.has_vest ? 'det-ppe-badge--ok' : 'det-ppe-badge--missing'}`}
                      title={d.has_vest ? "Vest: Pass" : "Vest: MISSING"}
                    >
                      {d.has_vest ? 'V' : 'V'}
                    </span>
                  </div>
                )}
              </div>
              <span className="det-det-conf">{(d.confidence * 100).toFixed(0)}%</span>
            </div>
          );
        })}
        {detections.length === 0 && (
          <div className="det-empty-state">
            <span className="det-hint">No active objects detected in the site zone.</span>
          </div>
        )}
      </div>

      {extra && <div style={{ marginTop: 'auto', paddingTop: '1.25rem' }}>{extra}</div>}
    </div>
  )
}

// ── Image Upload Mode (unchanged from original) ────────────────────────────────
function DetectionImageOverlay({ result }: { result: DetectResult }) {
  const zones = result.heatmap?.zones ?? []
  const points = result.heatmap?.points ?? []
  const riskZones = zones.filter(zone => zone.risk_score >= 0.4)

  return (
    <svg className="det-results__heatmap" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        {points.map((point, index) => {
          const color = point.risk_level === 'CRITICAL'
            ? 'rgba(255,42,42,'
            : point.risk_level === 'HIGH'
              ? 'rgba(255,140,0,'
              : point.risk_level === 'MODERATE'
                ? 'rgba(255,214,0,'
                : point.type === 'violation'
                  ? 'rgba(255,96,60,'
                  : 'rgba(0,255,136,'
          return (
            <radialGradient id={`risk-grad-${index}`} key={`grad-${index}`}>
              <stop offset="0%" stopColor={`${color}${Math.min(0.08 + point.value * 0.08, 0.18)})`} />
              <stop offset="100%" stopColor={`${color}0)`} />
            </radialGradient>
          )
        })}
      </defs>

      {points.map((point, index) => (
        <ellipse
          key={`point-${index}`}
          cx={point.x}
          cy={point.y}
          rx={0.03 + point.value * 0.03}
          ry={0.05 + point.value * 0.05}
          fill={`url(#risk-grad-${index})`}
        />
      ))}

      {riskZones.map((zone, index) => {
        const [x1, y1, x2, y2] = zone.box
        const frameW = Math.max(result.heatmap?.frame_width ?? x2, 1)
        const frameH = Math.max(result.heatmap?.frame_height ?? y2, 1)
        const x = x1 / frameW
        const y = y1 / frameH
        const width = (x2 - x1) / frameW
        const height = (y2 - y1) / frameH
        const color = zone.risk_level === 'CRITICAL'
          ? '#ff2a2a'
          : zone.risk_level === 'HIGH'
            ? '#ff8c00'
            : '#ffd600'
        return (
          <g key={`zone-${index}`}>
            <rect
              x={x}
              y={y}
              width={width}
              height={height}
              fill="none"
              stroke={color}
              strokeOpacity={0.5 + zone.risk_score * 0.3}
              strokeWidth={0.002}
              strokeDasharray="0.012 0.008"
            />
            <text x={x + 0.01} y={Math.max(0.035, y - 0.01)} fontSize={0.026} className="det-results__risk-label">
              {zone.risk_level} {Math.round(zone.risk_score * 100)}%
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function ImageUploadMode() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<DetectResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [condition, setCondition] = useState<Condition>('S1_normal')
  const { settings } = useSettings()
  const { pushDetections, setRunning } = useDetectionStats()
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    setFile(f); setResult(null); setError(null)
    setPreview(URL.createObjectURL(f))
  }

  const handleDetect = async () => {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('condition', condition)
      form.append('confidence', settings.confidenceThreshold.toString())
      form.append('model', settings.selectedModel)
      const res = await fetch(`${API}/detect/image`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const data = await res.json()
      setResult(data)
      // Push to shared stats context
      pushDetections(data.detections, data.elapsed_ms)
      setRunning(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Detection failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="det-upload">
      {!result && !file && (
        <div
          className="det-dropzone"
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleFile(f) }}
          onDragOver={e => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
        >
          <div className="det-dropzone__empty">
            <span>DROP IMAGE HERE</span>
            <span className="det-hint">or click to browse</span>
          </div>
          <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }}
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
        </div>
      )}

      {file && !result && (
        <div className="det-results">
          <div className="det-results__image-wrap">
            <img src={preview!} className="det-results__image" alt="preview" />
          </div>
          <DetectionSidebar
            detections={[]}
            renderControls={
              <>
                <ConditionPicker value={condition} onChange={setCondition} />
                <button
                  className={`det-run-btn ${loading ? 'det-run-btn--loading' : ''}`}
                  onClick={handleDetect}
                  disabled={loading}
                  style={{ width: '100%', marginTop: '1rem' }}
                >
                  {loading ? 'DETECTING...' : 'RUN DETECTION'}
                </button>
                <button className="det-reset-btn" onClick={() => { setFile(null); setPreview(null) }}>
                  CANCEL
                </button>
              </>
            }
          />
        </div>
      )}

      {error && <p className="det-error">{error}</p>}

      {result && (
        <div className="det-results">
          <div className="det-results__image-wrap">
            <img src={`data:image/jpeg;base64,${result.image_b64}`}
              className="det-results__image" alt="detection result" />
            {result.heatmap && <DetectionImageOverlay result={result} />}
          </div>
          <DetectionSidebar
            detections={result.detections}
            elapsed={result.elapsed_ms}
            mode={result.mode}
            renderControls={<ConditionPicker value={condition} onChange={setCondition} />}
            extra={
              <button className="det-reset-btn"
                onClick={() => { setResult(null); setFile(null); setPreview(null) }}>
                DETECT ANOTHER
              </button>
            }
          />
        </div>
      )}
    </div>
  )
}

// ── Video Upload Mode ──────────────────────────────────────────────────────────
//
// Architecture:
//   • No native <video controls> — custom play/pause + seek bar so the canvas
//     overlay covers exactly the video element (no controls-height offset bug).
//   • Self-scheduling async inference loop (no setInterval pile-up).
//     Fires once, awaits response, then schedules next call with
//     max(0, MIN_GAP_MS - inferenceTime) delay → at most ~2-3 inferences/sec.
//   • Frames resized to max 640 px before sending — matches YOLO's native
//     resolution, reduces JPEG size & backend decode time, does NOT change
//     ensemble or WBF logic on the server.
//   • IoU tracker (mergeTracks): matches new detections to existing tracks by
//     class + IoU ≥ 0.25; carries missed tracks for 2 cycles (faded opacity)
//     before dropping them. Eliminates box flicker and false-negative blinks.
//   • Letterbox-aware coordinate transform: accounts for object-fit:contain
//     bars so boxes land exactly on detected objects regardless of video
//     aspect ratio or display size.
//   • Position lerp (α = 0.25/frame at 60 fps): smoothly slides each track's
//     drawn box toward its latest detected position — no sudden jumps.
//
function VideoUploadMode() {
  const [file, setFile] = useState<File | null>(null)
  const [condition, setCondition] = useState<Condition>('S1_normal')
  const [autoMode, setAutoMode] = useState(true)
  const [detectedCondition, setDetectedCondition] = useState<string>('S1_normal')
  const [liveFps, setLiveFps] = useState(0)
  const [detections, setDetections] = useState<Detection[]>([])
  const [elapsed, setElapsed] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  /** Consecutive backend failures — shown in status bar so user knows why 0 detections */
  const [inferError, setInferError] = useState<string | null>(null)
  const [peakRiskMoments, setPeakRiskMoments] = useState<PeakRiskMoment[]>([])
  const { settings } = useSettings()
  const { pushDetections, setRunning } = useDetectionStats()

  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const captureRef = useRef<HTMLCanvasElement>(null)  // offscreen, not in DOM
  const inputRef = useRef<HTMLInputElement>(null)

  // Refs for state that inference loop / RAF must read without stale closures
  const rafRef = useRef<number | null>(null)
  const isRunningRef = useRef(false)
  const configRef = useRef({
    condition,
    autoMode: true,
    model: settings.selectedModel,
  })
  const tracksRef = useRef<Track[]>([])
  const pendingRef = useRef<Detection[] | null>(null)  // set by inference, consumed by RAF
  const pendingHeatmapRef = useRef<DetectionHeatmapPayload | null>(null)
  const riskZonesRef = useRef<DetectionRiskZone[]>([])
  const frameWRef = useRef(1)   // dimensions of last sent frame (for coord transform)
  const frameHRef = useRef(1)
  // ROI polygon in letterbox-normalised content coords [0-1].
  // null = no zone filter. Set via the Draw Zone tool.
  const roiRef = useRef<[number, number][] | null>(null)
  const [roiPoly, setRoiPoly] = useState<[number, number][] | null>(null)
  const [roiDrawMode, setRoiDrawMode] = useState(false)
  const [roiDraft, setRoiDraft] = useState<[number, number][]>([])
  const [showHeatmap, setShowHeatmap] = useState(false)
  const heatmapHistoryRef = useRef<HeatmapPoint[]>([])
  const fpsHistoryRef = useRef<number[]>([])
  const isFirstFrameRef = useRef(false)
  // Throttle low-priority DOM label updates (fps, condition, det count) to at
  // most once per second so they don't trigger extra React re-renders on every
  // inference cycle when the pipeline is running at 10–15 fps.
  const lastLabelUpdateRef = useRef<number>(0)
  const pendingFpsRef = useRef<number>(0)
  const pendingCondRef = useRef<string>('S1_normal')
  const pendingDetCountRef = useRef<number>(0)

  useEffect(() => {
    configRef.current = {
      condition,
      autoMode,
      model: settings.selectedModel,
    }
  }, [condition, autoMode, settings])

  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file])
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl) }, [videoUrl])

  // ── ROI helpers ──────────────────────────────────────────────────────────────
  // Point-in-polygon via ray-casting (works in any coord space)
  const pointInRoi = useCallback((px: number, py: number, poly: [number, number][]): boolean => {
    if (poly.length < 3) return true  // no valid polygon = pass everything
    let inside = false
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j]
      if (((yi > py) !== (yj > py)) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi)
        inside = !inside
    }
    return inside
  }, [])

  // ── RAF draw loop ────────────────────────────────────────────────────────────
  // Runs at ~60 fps. Consumes pendingRef → tracker → lerp → canvas draw.
  // Also draws the ROI polygon boundary and filters detections outside the zone.
  const drawOverlay = useCallback(() => {
    const canvas = overlayRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const dw = video.clientWidth
    const dh = video.clientHeight
    if (canvas.width !== dw) canvas.width = dw
    if (canvas.height !== dh) canvas.height = dh

    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, dw, dh)

    // Letterbox: actual video content rect within element
    const { rw, rh, ox, oy } = _letterbox(video.videoWidth, video.videoHeight, dw, dh)
    const fw = frameWRef.current, fh = frameHRef.current

    // ── Draw ROI polygon (if set) ─────────────────────────────────────────
    const roi = roiRef.current
    if (roi && roi.length >= 2) {
      ctx.save()
      ctx.strokeStyle = '#ffcc00'
      ctx.lineWidth = 2
      ctx.setLineDash([6, 4])
      ctx.beginPath()
      const [px0, py0] = roi[0]
      ctx.moveTo(px0 * rw + ox, py0 * rh + oy)
      for (let i = 1; i < roi.length; i++) {
        const [pxi, pyi] = roi[i]
        ctx.lineTo(pxi * rw + ox, pyi * rh + oy)
      }
      ctx.closePath()
      ctx.stroke()
      // Semi-transparent fill to show the active zone
      ctx.fillStyle = 'rgba(255, 204, 0, 0.06)'
      ctx.fill()
      ctx.setLineDash([])
      // Label
      ctx.font = 'bold 10px monospace'
      ctx.fillStyle = '#ffcc00'
      ctx.fillText('SITE ZONE', roi[0][0] * rw + ox + 4, roi[0][1] * rh + oy - 4)
      ctx.restore()
    }

    // ── Draw Heatmap (if enabled) ──────────────────────────────────────────
    if (showHeatmap && heatmapHistoryRef.current.length > 0) {
      heatmapHistoryRef.current = drawHeatmap(
        ctx,
        heatmapHistoryRef.current,
        performance.now(),
        rw,
        rh,
        ox,
        oy
      )
    }

    // Consume pending detections → filter by ROI → update tracker
    if (showHeatmap) {
      drawRiskZones(ctx, riskZonesRef.current, fw, fh, rw, rh, ox, oy)
    }

    if (pendingRef.current !== null) {
      let dets = pendingRef.current
      if (roi && roi.length >= 3) {
        dets = dets.filter(d => {
          const nx = (d.box[0] + d.box[2]) / 2 / fw
          const ny = (d.box[1] + d.box[3]) / 2 / fh
          return pointInRoi(nx, ny, roi)
        })
      }
      tracksRef.current = mergeTracks(tracksRef.current, dets)

      const backendHeatmap = pendingHeatmapRef.current
      riskZonesRef.current = backendHeatmap?.zones ?? []

      // Add backend risk points to heatmap history. Fall back to local worker
      // centroids for older backend responses.
      const now = performance.now()
      const backendPoints = pointsFromHeatmap(backendHeatmap ?? undefined, now)
      if (backendPoints.length > 0) {
        heatmapHistoryRef.current.push(...backendPoints)
      } else {
        dets.forEach(d => {
          const isWorker = d.class === 'worker' || d.class === 'person'
          const hasViolation = isWorker && (!d.has_helmet || !d.has_vest)
          heatmapHistoryRef.current.push({
            x: (d.box[0] + d.box[2]) / 2 / fw,
            y: (d.box[1] + d.box[3]) / 2 / fh,
            time: now,
            type: hasViolation ? 'violation' : 'worker',
            value: hasViolation ? 0.78 : 0.28,
            riskLevel: hasViolation ? 'HIGH' : 'LOW',
          })
        })
      }

      pendingRef.current = null
      pendingHeatmapRef.current = null
    }

    // α=0.40 per RAF tick: boxes reach the new position faster than α=0.25,
    // reducing perceived lag on fast-moving workers.
    const LERP = 0.40

    for (const t of tracksRef.current) {
      const [x1, y1, x2, y2] = t.frameBox
      const tx1 = (x1 / fw) * rw + ox, ty1 = (y1 / fh) * rh + oy
      const tx2 = (x2 / fw) * rw + ox, ty2 = (y2 / fh) * rh + oy

      if (!t.initialized) {
        t.sx1 = tx1; t.sy1 = ty1; t.sx2 = tx2; t.sy2 = ty2
        t.initialized = true
      } else {
        t.sx1 += (tx1 - t.sx1) * LERP
        t.sy1 += (ty1 - t.sy1) * LERP
        t.sx2 += (tx2 - t.sx2) * LERP
        t.sy2 += (ty2 - t.sy2) * LERP
      }

      ctx.globalAlpha = t.missed > 0 ? 0.45 : 1.0
      const col = resolveCssVar(CLASS_COLORS[t.cls] ?? '#aaaaaa')

      ctx.strokeStyle = col
      ctx.lineWidth = 2.5
      ctx.strokeRect(t.sx1, t.sy1, t.sx2 - t.sx1, t.sy2 - t.sy1)

      const label = `${t.cls} ${(t.confidence * 100).toFixed(0)}%`
      ctx.font = 'bold 11px monospace'
      const tw = ctx.measureText(label).width
      ctx.fillStyle = col
      ctx.fillRect(t.sx1, t.sy1 - 17, tw + 8, 17)
      ctx.fillStyle = '#000000'
      ctx.fillText(label, t.sx1 + 4, t.sy1 - 3)

      ctx.globalAlpha = 1
    }

    rafRef.current = requestAnimationFrame(drawOverlay)
  }, [pointInRoi])

  // ── ROI polygon click handler ─────────────────────────────────────────────────
  const handleOverlayClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!roiDrawMode) return
    const canvas = overlayRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    const rect = canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top
    // Convert canvas click → normalised content-area coords
    const { rw, rh, ox, oy } = _letterbox(video.videoWidth, video.videoHeight, canvas.width, canvas.height)
    const nx = (cx - ox) / rw
    const ny = (cy - oy) / rh
    setRoiDraft(prev => [...prev, [nx, ny] as [number, number]])
  }, [roiDrawMode])

  const handleOverlayDblClick = useCallback(() => {
    if (!roiDrawMode || roiDraft.length < 3) return
    roiRef.current = roiDraft
    setRoiPoly(roiDraft)
    setRoiDraft([])
    setRoiDrawMode(false)
  }, [roiDrawMode, roiDraft])

  const clearRoi = useCallback(() => {
    roiRef.current = null
    setRoiPoly(null)
    setRoiDraft([])
    setRoiDrawMode(false)
  }, [])

  // Draw draft polygon points while user is placing them
  useEffect(() => {
    if (!roiDrawMode || roiDraft.length === 0) return
    const canvas = overlayRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')!
    const { rw, rh, ox, oy } = _letterbox(video.videoWidth, video.videoHeight, canvas.width, canvas.height)
    ctx.save()
    ctx.strokeStyle = '#ffcc00'
    ctx.fillStyle = 'rgba(255,204,0,0.8)'
    ctx.lineWidth = 2
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    const [dx0, dy0] = roiDraft[0]
    ctx.moveTo(dx0 * rw + ox, dy0 * rh + oy)
    roiDraft.forEach(([dx, dy]) => ctx.lineTo(dx * rw + ox, dy * rh + oy))
    ctx.stroke()
    roiDraft.forEach(([dx, dy]) => {
      ctx.beginPath()
      ctx.arc(dx * rw + ox, dy * rh + oy, 4, 0, Math.PI * 2)
      ctx.fill()
    })
    ctx.setLineDash([])
    ctx.restore()
  }, [roiDraft, roiDrawMode])

  // ── Inference loop (self-scheduling, never piles up) ─────────────────────────
  // MIN_GAP_MS lowered from 350→80: inference races at GPU speed (~6-10 fps)
  // while the RAF render loop stays fully independent at 60 fps.
  const startInferenceLoop = useCallback(() => {
    const MIN_GAP_MS = 80  // was 350 — GPU FP16 inference ~100-150ms so total ≈ inference time

    const loop = async () => {
      if (!isRunningRef.current) return

      const video = videoRef.current
      const capture = captureRef.current
      if (!video || !capture || video.readyState < 2 || video.paused || video.ended) {
        if (isRunningRef.current) setTimeout(loop, 80)
        return
      }

      const MAX_DIM = 640
      const vw = video.videoWidth, vh = video.videoHeight
      const scale = Math.min(1, MAX_DIM / Math.max(vw, vh, 1))
      const fw = Math.round(vw * scale), fh = Math.round(vh * scale)
      capture.width = fw; capture.height = fh
      capture.getContext('2d')!.drawImage(video, 0, 0, fw, fh)
      frameWRef.current = fw; frameHRef.current = fh

      const b64 = capture.toDataURL('image/jpeg', 0.72)  // slightly higher quality than before
      const t0 = performance.now()
      const cfg = configRef.current

      // Consume the first-frame reset flag before the async fetch
      const sendReset = isFirstFrameRef.current
      if (sendReset) isFirstFrameRef.current = false

      try {
        const form = new FormData()
        form.append('image_b64', b64)
        form.append('condition', cfg.autoMode ? 'auto' : cfg.condition)
        form.append('reset_tracker', sendReset ? '1' : '0')
        form.append('model', cfg.model)
        if (roiRef.current && roiRef.current.length >= 3) {
          form.append('zone_poly', JSON.stringify(roiRef.current))
        }

        const res = await fetch(`${API}/detect/frame`, { method: 'POST', body: form })
        if (res.ok && isRunningRef.current) {
          const data = await res.json()
          const frameElapsed = Math.round(performance.now() - t0)

          // Always update canvas-driving refs immediately (no React re-render)
          pendingRef.current = data.detections
          pendingHeatmapRef.current = data.heatmap ?? null
          setInferError(null)
          pushDetections(data.detections, frameElapsed)

          // 10-frame rolling FPS average — stored in pending ref, not state yet
          if (frameElapsed > 0) {
            const hist = fpsHistoryRef.current
            hist.push(Math.round(1000 / frameElapsed))
            if (hist.length > 10) hist.shift()
            pendingFpsRef.current = Math.round(hist.reduce((a, b) => a + b, 0) / hist.length)
          }

          // Accumulate pending label values without triggering state updates
          if (data.scene_condition) pendingCondRef.current = data.scene_condition
          pendingDetCountRef.current = data.detections.length

          // Throttle React state updates to ≤ 1 Hz — the status bar, sidebar
          // count, FPS counter, and scene label do not need to update every
          // frame; capping at once/second eliminates ~90% of re-renders while
          // the canvas RAF loop continues drawing at full inference speed.
          const nowMs = performance.now()
          if (nowMs - lastLabelUpdateRef.current >= 1000) {
            lastLabelUpdateRef.current = nowMs
            setLiveFps(pendingFpsRef.current)
            setDetectedCondition(data.scene_condition ?? pendingCondRef.current)
            setElapsed(frameElapsed)
            setDetections([...data.detections])
          }

          // Peak risk moments: evaluated every frame using raw data, not state
          const violations = data.detections.filter((d: any) => (d.class === 'worker' || d.class === 'person') && (!d.has_helmet || !d.has_vest))
          if (violations.length >= 2) {
            setPeakRiskMoments(prev => {
              if (prev.some(m => Math.abs(m.time - video.currentTime) < 2)) return prev
              return [...prev, { time: video.currentTime, score: violations.length, type: 'PPE_VIOLATION' }].sort((a, b) => b.score - a.score).slice(0, 5)
            })
          }
        } else if (!res.ok) {
          setInferError(`Backend error ${res.status}`)
        }
      } catch (e) {
        setInferError(e instanceof Error ? e.message : 'Backend unreachable — is server.py running?')
      }

      const gap = Math.max(0, MIN_GAP_MS - (performance.now() - t0))
      if (isRunningRef.current) setTimeout(loop, gap)
    }

    loop()
  }, [])

  // ── Stop RAF + inference ─────────────────────────────────────────────────────
  const stopAll = useCallback((clearCanvas = false) => {
    isRunningRef.current = false
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    tracksRef.current = []
    pendingRef.current = null
    pendingHeatmapRef.current = null
    riskZonesRef.current = []
    heatmapHistoryRef.current = []
    if (clearCanvas) {
      const c = overlayRef.current
      c?.getContext('2d')?.clearRect(0, 0, c.width, c.height)
    }
  }, [])

  // ── Video event handlers ─────────────────────────────────────────────────────
  const onPlay = useCallback(() => {
    setIsPlaying(true)
    isRunningRef.current = true
    rafRef.current = requestAnimationFrame(drawOverlay)
    startInferenceLoop()
    setRunning(true)
  }, [drawOverlay, startInferenceLoop, setRunning])

  const onPause = useCallback(() => {
    setIsPlaying(false)
    isRunningRef.current = false
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    // Keep last overlay visible when paused
    setRunning(false)
  }, [setRunning])

  const onEnded = useCallback(() => {
    setIsPlaying(false)
    stopAll(true)
    setDetections([])
    setRunning(false)
  }, [stopAll, setRunning])

  const onTimeUpdate = useCallback(() => {
    const v = videoRef.current; if (v) setCurrentTime(v.currentTime)
  }, [])

  const onLoadedMetadata = useCallback(() => {
    const v = videoRef.current; if (v) setDuration(v.duration)
  }, [])

  useEffect(() => () => stopAll(), [stopAll])

  // ── Custom seek bar ──────────────────────────────────────────────────────────
  const togglePlay = useCallback(() => {
    const v = videoRef.current; if (!v) return
    v.paused ? v.play() : v.pause()
  }, [])

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const v = videoRef.current; if (!v) return
    v.currentTime = parseFloat(e.target.value)
    setCurrentTime(v.currentTime)
  }, [])

  // ── File handling ────────────────────────────────────────────────────────────
  const handleFile = (f: File) => {
    stopAll(true)
    setDetections([]); setIsPlaying(false); setError(null); setInferError(null)
    setCurrentTime(0); setDuration(0)
    setLiveFps(0)
    fpsHistoryRef.current = []
    isFirstFrameRef.current = true   // tells the loop to send reset_tracker=1 once
    lastLabelUpdateRef.current = 0   // allow immediate first-frame label update
    pendingFpsRef.current = 0
    pendingCondRef.current = 'S1_normal'
    pendingDetCountRef.current = 0
    setFile(f)
  }

  // ── Download annotated video (unchanged — full server-side pipeline) ──────────
  const handleDownload = async () => {
    if (!file) return
    setDownloading(true); setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('condition', condition)
      form.append('confidence', settings.confidenceThreshold.toString())
      form.append('model', settings.selectedModel)
      form.append('sample_every', '3')
      const res = await fetch(`${API}/detect/video`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = file.name.replace(/\.[^.]+$/, '') + '_buildsight_detected.mp4'
      a.click(); URL.revokeObjectURL(url)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Download failed')
    } finally { setDownloading(false) }
  }

  // ── Dropzone (no file selected) ──────────────────────────────────────────────
  if (!file) {
    return (
      <div className="det-upload">
        <div
          className="det-dropzone"
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleFile(f) }}
          onDragOver={e => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
        >
          <div className="det-dropzone__empty">
            <span>DROP VIDEO HERE</span>
            <span className="det-hint">or click to browse · MP4, MOV, AVI, MKV</span>
          </div>
          <input ref={inputRef} type="file" accept="video/*" style={{ display: 'none' }}
            onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
        </div>
      </div>
    )
  }

  return (
    <div className="det-video">
      {error && <p className="det-error">{error}</p>}

      {/* ── Status bar ── */}
      <div className="det-video__status">
        <span className={`det-video__indicator ${isPlaying && !inferError ? 'det-video__indicator--live' : ''}`} />
        {inferError
          ? <span style={{ color: 'var(--color-danger, #ff4444)', fontWeight: 700 }}>
            ⚠ {inferError}
          </span>
          : <span>{isPlaying ? 'LIVE DETECTION' : 'PAUSED — CLICK VIDEO TO START'}</span>
        }
        {isPlaying && !inferError && elapsed > 0 && (
          <>
            <span className="det-video__status-sep" />
            <span>{elapsed}ms</span>
            {liveFps > 0 && (
              <>
                <span className="det-video__status-sep" />
                <span>{liveFps} FPS</span>
              </>
            )}
            <span className="det-video__status-sep" />
            <span>{detections.length} det</span>
            {autoMode && detectedCondition && (
              <>
                <span className="det-video__status-sep" />
                <span style={{ color: CONDITION_COLORS[detectedCondition] ?? '#aaa', fontWeight: 600 }}>
                  {detectedCondition.replace(/_/g, ' ').toUpperCase()}
                </span>
              </>
            )}
          </>
        )}
        {roiPoly && (
          <>
            <span className="det-video__status-sep" />
            <span style={{ color: '#ffcc00' }}>ZONE ACTIVE</span>
          </>
        )}
        {roiDrawMode && (
          <>
            <span className="det-video__status-sep" />
            <span style={{ color: '#ffcc00' }}>Click to add points · Double-click to close</span>
          </>
        )}
      </div>

      {/* ── Main layout ── */}
      <div className="det-video__layout">
        <div className="det-video__col">
          {/* player-wrap: click = play/pause UNLESS in draw mode */}
          <div
            className="det-video__player-wrap"
            onClick={roiDrawMode ? undefined : togglePlay}
            style={{ cursor: roiDrawMode ? 'crosshair' : 'pointer' }}
          >
            {videoUrl && (
              <video
                ref={videoRef}
                src={videoUrl}
                className="det-video__player"
                onPlay={onPlay}
                onPause={onPause}
                onEnded={onEnded}
                onTimeUpdate={onTimeUpdate}
                onLoadedMetadata={onLoadedMetadata}
              />
            )}
            {/* Overlay canvas — handles both detection drawing and ROI clicks */}
            <canvas
              ref={overlayRef}
              className="det-video__overlay"
              onClick={handleOverlayClick}
              onDoubleClick={handleOverlayDblClick}
              style={{ cursor: roiDrawMode ? 'crosshair' : 'pointer' }}
            />
            {!isPlaying && !roiDrawMode && (
              <div className="det-video__play-hint" aria-hidden="true">▶</div>
            )}
            {roiDrawMode && (
              <div className="det-video__play-hint" style={{ fontSize: '1rem', padding: '0.4rem 0.8rem', background: 'rgba(255,204,0,0.18)', color: '#ffcc00' }} aria-hidden="true">
                Click to define zone · Dbl-click to finish
              </div>
            )}
          </div>

          {/* Custom seek bar */}
          <div className="det-video__seekbar">
            <button
              className="det-video__play-btn"
              onClick={e => { e.stopPropagation(); togglePlay() }}
              aria-label={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? '⏸' : '▶'}
            </button>
            <input
              type="range" className="det-video__seek"
              min={0} max={duration || 1} step={0.05} value={currentTime}
              onChange={handleSeek}
              onClick={e => e.stopPropagation()}
            />
            <span className="det-video__time">
              {_fmtTime(currentTime)} / {_fmtTime(duration)}
            </span>
          </div>

          <PeakRiskMomentsPanel
            className="det-risk-sidebar--under-video"
            moments={peakRiskMoments}
            emptyLabel="No high-risk moments detected yet."
            onSelect={(moment) => {
              if (videoRef.current) {
                videoRef.current.currentTime = moment.time
              }
            }}
          />
        </div>

        <DetectionSidebar
          detections={detections}
          elapsed={isPlaying ? elapsed : undefined}
          mode={isPlaying ? 'live-ensemble' : undefined}
          renderControls={
            <>
              <SceneAutoIndicator
                condition={autoMode ? detectedCondition : condition}
                autoMode={autoMode}
                onToggleAuto={() => setAutoMode(a => !a)}
                onManualChange={setCondition}
              />
              <div className="det-sidebar-actions">
                <button
                  className={`det-run-btn ${roiDrawMode ? 'det-run-btn--stop' : ''}`}
                  onClick={() => setRoiDrawMode(!roiDrawMode)}
                >
                  {roiDrawMode ? 'CANCEL DRAW' : 'DRAW ZONE'}
                </button>
                {roiPoly && (
                  <button className="det-run-btn det-run-btn--stop" onClick={clearRoi}>
                    CLEAR ZONE
                  </button>
                )}
                <button className="det-run-btn" onClick={handleDownload} disabled={downloading}>
                  {downloading ? 'PROCESSING...' : 'DOWNLOAD ANNOTATED'}
                </button>
                <button
                  className={`det-run-btn ${showHeatmap ? 'det-run-btn--active' : ''}`}
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  title="Toggle spatial density heatmap overlay"
                >
                  {showHeatmap ? 'HIDE HEATMAP' : 'SHOW HEATMAP'}
                </button>
                <button className="det-reset-btn" onClick={() => { stopAll(true); setFile(null); setDetections([]) }}>
                  CHANGE VIDEO
                </button>
              </div>
            </>
          }
        />

      </div>

      {/* Hidden offscreen capture canvas */}
      <canvas ref={captureRef} style={{ display: 'none' }} />
    </div>
  )
}

// ── Live Camera Mode ───────────────────────────────────────────────────────────
// Refactored (Jovi handoff 2026-04-06) to use the same canvas-overlay + IoU
// tracker architecture as VideoUploadMode instead of the old base64 round-trip:
//
//   OLD path: capture → toDataURL → POST → server draws OpenCV boxes → encode
//             → send back image_b64 → decode → display  (≈3-4× extra latency)
//
//   NEW path: capture → toDataURL → POST → receive JSON dets → tracker →
//             lerp → draw on transparent canvas overlay  (no image round-trip)
//
export function LiveMode() {
  const { pushDetections, setRunning } = useDetectionStats()

  const [active, setActive] = useState(false)
  const [condition, setCondition] = useState<Condition>('S1_normal')
  const [detections, setDetections] = useState<Detection[]>([])
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [peakRiskMoments, setPeakRiskMoments] = useState<PeakRiskMoment[]>([])
  const heatmapHistoryRef = useRef<HeatmapPoint[]>([])

  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const captureRef = useRef<HTMLCanvasElement>(null)  // offscreen, not in DOM

  const rafRef = useRef<number | null>(null)
  const isRunningRef = useRef(false)
  const configRef = useRef({ condition })
  const tracksRef = useRef<Track[]>([])
  const pendingRef = useRef<Detection[] | null>(null)
  const pendingHeatmapRef = useRef<DetectionHeatmapPayload | null>(null)
  const riskZonesRef = useRef<DetectionRiskZone[]>([])
  const frameWRef = useRef(1)
  const frameHRef = useRef(1)
  // Throttle label/sidebar state updates to ≤ 1 Hz (same as VideoUploadMode)
  const lastLabelUpdateRef = useRef<number>(0)

  useEffect(() => {
    configRef.current = { condition }
  }, [condition])

  // ── RAF draw loop (identical logic to VideoUploadMode) ───────────────────
  const drawOverlay = useCallback(() => {
    const canvas = overlayRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const dw = video.clientWidth
    const dh = video.clientHeight
    if (canvas.width !== dw) canvas.width = dw
    if (canvas.height !== dh) canvas.height = dh

    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, dw, dh)

    const { rw, rh, ox, oy } = _letterbox(video.videoWidth, video.videoHeight, dw, dh)
    const fw = frameWRef.current
    const fh = frameHRef.current

    if (showHeatmap && heatmapHistoryRef.current.length > 0) {
      heatmapHistoryRef.current = drawHeatmap(
        ctx,
        heatmapHistoryRef.current,
        performance.now(),
        rw,
        rh,
        ox,
        oy
      )
    }

    if (showHeatmap) {
      drawRiskZones(ctx, riskZonesRef.current, fw, fh, rw, rh, ox, oy)
    }

    if (pendingRef.current !== null) {
      const dets = pendingRef.current
      tracksRef.current = mergeTracks(tracksRef.current, dets)

      const backendHeatmap = pendingHeatmapRef.current
      riskZonesRef.current = backendHeatmap?.zones ?? []

      // Add backend risk points to heatmap history. Fall back to local worker
      // centroids for older backend responses.
      const now = performance.now()
      const backendPoints = pointsFromHeatmap(backendHeatmap ?? undefined, now)
      if (backendPoints.length > 0) {
        heatmapHistoryRef.current.push(...backendPoints)
      } else {
        dets.forEach(d => {
          const isWorker = d.class === 'worker' || d.class === 'person'
          const hasViolation = isWorker && (!d.has_helmet || !d.has_vest)
          heatmapHistoryRef.current.push({
            x: (d.box[0] + d.box[2]) / 2 / fw,
            y: (d.box[1] + d.box[3]) / 2 / fh,
            time: now,
            type: hasViolation ? 'violation' : 'worker',
            value: hasViolation ? 0.78 : 0.28,
            riskLevel: hasViolation ? 'HIGH' : 'LOW',
          })
        })
      }

      pendingRef.current = null
      pendingHeatmapRef.current = null
    }
    const LERP = 0.40  // faster snap to new position reduces perceived lag

    for (const t of tracksRef.current) {
      const [x1, y1, x2, y2] = t.frameBox
      const tx1 = (x1 / fw) * rw + ox, ty1 = (y1 / fh) * rh + oy
      const tx2 = (x2 / fw) * rw + ox, ty2 = (y2 / fh) * rh + oy

      if (!t.initialized) {
        t.sx1 = tx1; t.sy1 = ty1; t.sx2 = tx2; t.sy2 = ty2
        t.initialized = true
      } else {
        t.sx1 += (tx1 - t.sx1) * LERP
        t.sy1 += (ty1 - t.sy1) * LERP
        t.sx2 += (tx2 - t.sx2) * LERP
        t.sy2 += (ty2 - t.sy2) * LERP
      }

      ctx.globalAlpha = t.missed > 0 ? 0.45 : 1.0

      // Fixed class colours: blue=worker, green=helmet, yellow=safety_vest
      const col = resolveCssVar(CLASS_COLORS[t.cls] ?? '#aaaaaa')

      ctx.strokeStyle = col
      ctx.lineWidth = 2.5
      ctx.strokeRect(t.sx1, t.sy1, t.sx2 - t.sx1, t.sy2 - t.sy1)

      const label = `${t.cls} ${(t.confidence * 100).toFixed(0)}%`
      ctx.font = 'bold 11px monospace'
      const tw = ctx.measureText(label).width
      ctx.fillStyle = col
      ctx.fillRect(t.sx1, t.sy1 - 17, tw + 8, 17)
      ctx.fillStyle = '#000'
      ctx.fillText(label, t.sx1 + 4, t.sy1 - 3)

      ctx.globalAlpha = 1
    }

    rafRef.current = requestAnimationFrame(drawOverlay)
  }, [])

  // ── Inference loop — same self-scheduling pattern as VideoUploadMode ─────
  const startInferenceLoop = useCallback(() => {
    const MIN_GAP_MS = 100  // was 450 — let GPU run at full speed

    const loop = async () => {
      if (!isRunningRef.current) return

      const video = videoRef.current
      const capture = captureRef.current
      if (!video || !capture || video.readyState < 2) {
        if (isRunningRef.current) setTimeout(loop, 100)
        return
      }

      const MAX_DIM = 640
      const vw = video.videoWidth, vh = video.videoHeight
      const scale = Math.min(1, MAX_DIM / Math.max(vw, vh, 1))
      const fw = Math.round(vw * scale), fh = Math.round(vh * scale)
      capture.width = fw; capture.height = fh
      capture.getContext('2d')!.drawImage(video, 0, 0, fw, fh)
      frameWRef.current = fw; frameHRef.current = fh

      const b64 = capture.toDataURL('image/jpeg', 0.72)
      const t0 = performance.now()
      const cfg = configRef.current

      try {
        const form = new FormData()
        form.append('image_b64', b64)
        form.append('condition', cfg.condition)
        const res = await fetch(`${API}/detect/frame`, { method: 'POST', body: form })
        if (res.ok && isRunningRef.current) {
          const data = await res.json()
          const frameElapsed = Math.round(performance.now() - t0)
          pendingRef.current = data.detections
          pendingHeatmapRef.current = data.heatmap ?? null
          pushDetections(data.detections, frameElapsed)

          // Throttle React state updates to ≤ 1 Hz — canvas RAF loop draws at
          // full inference speed; sidebar counts don't need to update every frame.
          const nowMs = performance.now()
          if (nowMs - lastLabelUpdateRef.current >= 1000) {
            lastLabelUpdateRef.current = nowMs
            setDetections([...data.detections])
            setElapsed(frameElapsed)
          }

          const violations = data.detections.filter((d: Detection) =>
            (d.class === 'worker' || d.class === 'person') && (!d.has_helmet || !d.has_vest)
          )
          if (violations.length >= 1) {
            setPeakRiskMoments(prev => {
              if (prev.some(moment => Math.abs(moment.time - video.currentTime) < 1.5)) return prev
              return [
                ...prev,
                { time: video.currentTime, score: violations.length, type: 'LIVE VIOLATION' },
              ].sort((left, right) => right.score - left.score).slice(0, 6)
            })
          }
        }
      } catch { /* network hiccup — continue */ }

      const gap = Math.max(0, MIN_GAP_MS - (performance.now() - t0))
      if (isRunningRef.current) setTimeout(loop, gap)
    }

    loop()
  }, [pushDetections])

  // ── Camera lifecycle ─────────────────────────────────────────────────────
  const startCamera = async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      })
      const video = videoRef.current
      if (!video) return
      video.srcObject = stream
      await video.play()
      setActive(true)
      setRunning(true)
      isRunningRef.current = true
      rafRef.current = requestAnimationFrame(drawOverlay)
      startInferenceLoop()
    } catch {
      setError('Camera access denied or unavailable.')
    }
  }

  const stopCamera = useCallback(() => {
    isRunningRef.current = false
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    tracksRef.current = []
    pendingRef.current = null
    pendingHeatmapRef.current = null
    riskZonesRef.current = []
    heatmapHistoryRef.current = []
    const video = videoRef.current
    if (video?.srcObject) {
      (video.srcObject as MediaStream).getTracks().forEach(t => t.stop())
      video.srcObject = null
    }
    const canvas = overlayRef.current
    canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
    setActive(false); setDetections([]); setElapsed(0)
    setRunning(false)
  }, [setRunning])

  useEffect(() => () => stopCamera(), [stopCamera])

  return (
    <div className="det-live">
      {error && <p className="det-error">{error}</p>}

      {/* Status bar */}
      <div className="det-video__status">
        <span className={`det-video__indicator ${active ? 'det-video__indicator--live' : ''}`} />
        <span>{active ? 'LIVE DETECTION' : 'CAMERA OFF'}</span>
        {active && elapsed > 0 && (
          <>
            <span className="det-video__status-sep" />
            <span>{elapsed}ms / inference</span>
            <span className="det-video__status-sep" />
            <span>{detections.length} detection{detections.length !== 1 ? 's' : ''}</span>
          </>
        )}
      </div>

      {/* Feed + sidebar */}
      <div className="det-video__layout">
        <div className="det-video__col">
          <div className="det-video__player-wrap" style={{ cursor: 'default' }}>
            <video
              ref={videoRef}
              className="det-video__player"
              muted
              playsInline
            />
            <canvas ref={overlayRef} className="det-video__overlay" />
            {!active && (
              <div className="det-live__placeholder" style={{ position: 'absolute', inset: 0 }}>
                Camera Off — click START CAMERA
              </div>
            )}
          </div>
          {active && (
            <div className="det-video__seekbar" style={{ border: 'none', background: 'none' }}>
              <button
                className={`det-run-btn ${showHeatmap ? 'det-run-btn--active' : ''}`}
                onClick={() => setShowHeatmap(!showHeatmap)}
                style={{ width: 'auto', padding: '0.4rem 1rem' }}
              >
                {showHeatmap ? 'HIDE HEATMAP' : 'SHOW HEATMAP'}
              </button>
              <div style={{ flex: 1 }} />
              <span className="det-video__time" style={{ color: 'var(--color-safe)' }}>LIVE STREAMING</span>
            </div>
          )}
        </div>

        <DetectionSidebar
          detections={detections}
          elapsed={active ? elapsed : undefined}
          mode={active ? 'live-ensemble' : undefined}
          renderControls={
            <>
              <ConditionPicker value={condition} onChange={setCondition} />
              <button
                className={`det-run-btn ${active ? 'det-run-btn--stop' : ''}`}
                onClick={active ? stopCamera : startCamera}
                style={{ width: '100%', marginTop: '1.25rem' }}
              >
                {active ? 'STOP CAMERA' : 'START CAMERA'}
              </button>
            </>
          }
        />

        {/* Peak Risk Moments Sidebar (Live) */}
        <div className="det-risk-sidebar">
          <div className="det-risk-sidebar__header">
            <span>PEAK RISK MOMENTS</span>
          </div>
          <div className="det-risk-sidebar__list">
            {peakRiskMoments.length === 0 ? (
              <div style={{ padding: '1rem', fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>
                Waiting for high-risk detection data...
              </div>
            ) : (
              peakRiskMoments.map((moment, idx) => (
                <div key={idx} className="det-risk-item" style={{ cursor: 'default' }}>
                  <div className="det-risk-item__meta">
                    <span className="det-risk-item__label">{moment.type}</span>
                    <span className="det-risk-item__score">RISK: {moment.score}</span>
                  </div>
                  <div className="det-risk-item__time">LIVE CAPTURE #{idx + 1}</div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Hidden offscreen capture canvas */}
      <canvas ref={captureRef} style={{ display: 'none' }} />
    </div>
  )
}

// ── Main Panel ─────────────────────────────────────────────────────────────────
type DetectionPanelMode = 'VIDEO' | 'IMAGE'

interface DetectionPanelProps {
  mode: DetectionPanelMode
}

export function DetectionPanel({ mode }: DetectionPanelProps) {
  const [apiOk, setApiOk] = useState<boolean | null>(null)
  const [modelMode, setModelMode] = useState('')
  const { setModelName } = useDetectionStats()
  const title = mode === 'VIDEO' ? 'PPE Detection - Video Workspace' : 'PPE Detection - Image Workspace'
  const meta = mode === 'VIDEO'
    ? 'Review uploaded construction footage with live overlays and export annotated MP4 evidence.'
    : 'Inspect still captures for PPE compliance, incident snapshots, and audit evidence.'

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => { setApiOk(true); setModelMode(d.mode); setModelName(d.mode || 'ensemble-wbf') })
      .catch(() => setApiOk(false))
  }, [setModelName])

  return (
    <div className="det-panel panel">
      <div className="panel-heading">
        <div>
          <p className="section-label">AI Detection Engine</p>
          <p className="det-panel__mode">{title}</p>
          <h3>PPE Detection — Image · Video · Live</h3>
        </div>
        <div className="det-panel__status">
          <p className="panel-meta">{meta}</p>
          <span className={`api-badge ${apiOk === true ? 'api-badge--ok' : apiOk === false ? 'api-badge--err' : 'api-badge--pending'}`}>
            {apiOk === true ? `BACKEND OK · ${modelMode.toUpperCase()}` : apiOk === false ? 'BACKEND OFFLINE' : 'CONNECTING...'}
          </span>
        </div>
      </div>

      <div className="det-body">
        {mode === 'IMAGE' ? <ImageUploadMode /> : <VideoUploadMode />}
      </div>
    </div>
  )
}
