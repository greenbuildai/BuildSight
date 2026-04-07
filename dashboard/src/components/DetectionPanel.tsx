import {
  useRef, useState, useEffect, useCallback, useMemo,
} from 'react'
import { useDetectionStats } from '../DetectionStatsContext'
import { useSettings } from '../SettingsContext'

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

interface DetectResult {
  detections: Detection[]
  class_counts: Record<string, number>
  total: number
  image_b64: string
  mode: string
  condition: Condition
  elapsed_ms: number
}

const CLASS_COLORS: Record<string, string> = {
  helmet:        '#00ff55', // Green
  safety_vest:   '#ffdd00', // Yellow
  worker:        '#0088ff', // Blue
  person:        '#0088ff', // Blue (same as worker)
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
  id:          number
  cls:         string
  confidence:  number
  /** Bounding box in the sent-frame's pixel space (e.g. 640×360) */
  frameBox:    [number, number, number, number]
  /** Smoothed display coords on the overlay canvas (lerped each RAF tick) */
  sx1: number; sy1: number; sx2: number; sy2: number
  initialized: boolean
  /** How many consecutive inference cycles had no match */
  missed: number
  /** PPE association flags — only set on worker/person tracks */
  has_helmet?: boolean
  has_vest?: boolean
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
      t.frameBox   = nd.box
      t.confidence = nd.confidence
      t.missed     = 0
      // Carry forward PPE association flags from the latest detection
      t.has_helmet = nd.has_helmet
      t.has_vest   = nd.has_vest
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
      has_vest:   d.has_vest,
    })
  })
  return alive
}

/** Compute letterbox offsets for object-fit:contain video rendering */
function _letterbox(vw: number, vh: number, dw: number, dh: number) {
  if (!vw || !vh) return { rw: dw, rh: dh, ox: 0, oy: 0 }
  const va = vw / vh, da = dw / dh
  return va > da
    ? { rw: dw,      rh: dw / va,  ox: 0,           oy: (dh - dw / va) / 2 }
    : { rw: dh * va, rh: dh,       ox: (dw - dh * va) / 2, oy: 0 }
}

function _fmtTime(s: number): string {
  const m = Math.floor(s / 60)
  return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`
}

// ── Shared condition picker ─────────────────────────────────────────────────────
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
function ImageUploadMode() {
  const [file, setFile]         = useState<File | null>(null)
  const [preview, setPreview]   = useState<string | null>(null)
  const [result, setResult]     = useState<DetectResult | null>(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string | null>(null)
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
  const [file, setFile]              = useState<File | null>(null)
  const [condition, setCondition]    = useState<Condition>('S1_normal')
  const [detections, setDetections]  = useState<Detection[]>([])
  const [elapsed, setElapsed]        = useState(0)
  const [isPlaying, setIsPlaying]    = useState(false)
  const [duration, setDuration]      = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [downloading, setDownloading] = useState(false)
  const [error, setError]            = useState<string | null>(null)
  /** Consecutive backend failures — shown in status bar so user knows why 0 detections */
  const [inferError, setInferError]  = useState<string | null>(null)
  const { settings } = useSettings()
  const { pushDetections, setRunning } = useDetectionStats()

  const videoRef   = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const captureRef = useRef<HTMLCanvasElement>(null)  // offscreen, not in DOM
  const inputRef   = useRef<HTMLInputElement>(null)

  // Refs for state that inference loop / RAF must read without stale closures
  const rafRef       = useRef<number | null>(null)
  const isRunningRef = useRef(false)
  const configRef    = useRef({
    condition,
    model:        settings.selectedModel,
    workerConf:   settings.workerConf,
    helmetConf:   settings.helmetConf,
    vestConf:     settings.vestConf,
    workerNmsIou: settings.workerNmsIou,
    helmetNmsIou: settings.helmetNmsIou,
    vestNmsIou:   settings.vestNmsIou,
    wbfWorkerIou: settings.wbfWorkerIou,
    wbfHelmetIou: settings.wbfHelmetIou,
    wbfVestIou:   settings.wbfVestIou,
    enableClahe:  settings.enableClahe,
    claheClip:    settings.claheClip,
  })
  const tracksRef    = useRef<Track[]>([])
  const pendingRef   = useRef<Detection[] | null>(null)  // set by inference, consumed by RAF
  const frameWRef    = useRef(1)   // dimensions of last sent frame (for coord transform)
  const frameHRef    = useRef(1)
  // ROI polygon in letterbox-normalised content coords [0-1].
  // null = no zone filter. Set via the Draw Zone tool.
  const roiRef       = useRef<[number, number][] | null>(null)
  const [roiPoly, setRoiPoly]       = useState<[number, number][] | null>(null)
  const [roiDrawMode, setRoiDrawMode] = useState(false)
  const [roiDraft, setRoiDraft]     = useState<[number, number][]>([])

  useEffect(() => {
    configRef.current = {
      condition,
      model:        settings.selectedModel,
      workerConf:   settings.workerConf,
      helmetConf:   settings.helmetConf,
      vestConf:     settings.vestConf,
      workerNmsIou: settings.workerNmsIou,
      helmetNmsIou: settings.helmetNmsIou,
      vestNmsIou:   settings.vestNmsIou,
      wbfWorkerIou: settings.wbfWorkerIou,
      wbfHelmetIou: settings.wbfHelmetIou,
      wbfVestIou:   settings.wbfVestIou,
      enableClahe:  settings.enableClahe,
      claheClip:    settings.claheClip,
    }
  }, [condition, settings])

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
    const video  = videoRef.current
    if (!canvas || !video) return

    const dw = video.clientWidth
    const dh = video.clientHeight
    if (canvas.width  !== dw) canvas.width  = dw
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
      ctx.lineWidth   = 2
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

    // Consume pending detections → filter by ROI → update tracker
    if (pendingRef.current !== null) {
      let dets = pendingRef.current
      if (roi && roi.length >= 3) {
        dets = dets.filter(d => {
          // Centre of detection box in frame-space → normalised to content area
          const cx = ((d.box[0] + d.box[2]) / 2 / fh)  // note: use fw below
          const nx = (d.box[0] + d.box[2]) / 2 / fw
          const ny = (d.box[1] + d.box[3]) / 2 / fh
          void cx  // suppress unused warning
          return pointInRoi(nx, ny, roi)
        })
      }
      tracksRef.current = mergeTracks(tracksRef.current, dets)
      pendingRef.current = null
    }

    // α=0.40 per RAF tick: boxes reach the new position faster than α=0.25,
    // reducing perceived lag on fast-moving workers.
    const LERP = 0.40

    for (const t of tracksRef.current) {
      const [x1, y1, x2, y2] = t.frameBox
      const tx1 = (x1 / fw) * rw + ox,  ty1 = (y1 / fh) * rh + oy
      const tx2 = (x2 / fw) * rw + ox,  ty2 = (y2 / fh) * rh + oy

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
      ctx.lineWidth   = 2.5
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
    const video  = videoRef.current
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
    const video  = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')!
    const { rw, rh, ox, oy } = _letterbox(video.videoWidth, video.videoHeight, canvas.width, canvas.height)
    ctx.save()
    ctx.strokeStyle = '#ffcc00'
    ctx.fillStyle   = 'rgba(255,204,0,0.8)'
    ctx.lineWidth   = 2
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

      const video   = videoRef.current
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
      const t0  = performance.now()
      const cfg = configRef.current

      try {
        const form = new FormData()
        form.append('image_b64', b64)
        form.append('condition',  cfg.condition)
        form.append('model',      cfg.model)
        // Per-class thresholds sent as JSON so the backend can apply them per-class
        form.append('class_conf', JSON.stringify({
          worker: cfg.workerConf, helmet: cfg.helmetConf, vest: cfg.vestConf,
        }))
        form.append('nms_iou', JSON.stringify({
          worker: cfg.workerNmsIou, helmet: cfg.helmetNmsIou, vest: cfg.vestNmsIou,
        }))
        form.append('wbf_iou', JSON.stringify({
          worker: cfg.wbfWorkerIou, helmet: cfg.wbfHelmetIou, vest: cfg.wbfVestIou,
        }))
        form.append('clahe',      cfg.enableClahe ? '1' : '0')
        form.append('clahe_clip', cfg.claheClip.toString())

        const res = await fetch(`${API}/detect/frame`, { method: 'POST', body: form })
        if (res.ok && isRunningRef.current) {
          const data = await res.json()
          pendingRef.current = data.detections
          setDetections([...data.detections])
          setElapsed(Math.round(performance.now() - t0))
          setInferError(null)
          pushDetections(data.detections, Math.round(performance.now() - t0))
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
    tracksRef.current  = []
    pendingRef.current = null
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
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
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
            <span>{elapsed}ms / frame</span>
            <span className="det-video__status-sep" />
            <span>{detections.length} det</span>
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
        </div>

        <DetectionSidebar
          detections={detections}
          elapsed={isPlaying ? elapsed : undefined}
          mode={isPlaying ? 'live-ensemble' : undefined}
          renderControls={
            <>
              <ConditionPicker value={condition} onChange={setCondition} />
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
  const { settings } = useSettings()
  const { pushDetections, setRunning } = useDetectionStats()

  const [active, setActive]       = useState(false)
  const [condition, setCondition] = useState<Condition>('S1_normal')
  const [detections, setDetections] = useState<Detection[]>([])
  const [elapsed, setElapsed]     = useState(0)
  const [error, setError]         = useState<string | null>(null)

  const videoRef   = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const captureRef = useRef<HTMLCanvasElement>(null)  // offscreen, not in DOM

  const rafRef       = useRef<number | null>(null)
  const isRunningRef = useRef(false)
  const configRef    = useRef({
    condition,
    workerConf:   settings.workerConf,
    helmetConf:   settings.helmetConf,
    vestConf:     settings.vestConf,
    workerNmsIou: settings.workerNmsIou,
    helmetNmsIou: settings.helmetNmsIou,
    vestNmsIou:   settings.vestNmsIou,
    wbfWorkerIou: settings.wbfWorkerIou,
    wbfHelmetIou: settings.wbfHelmetIou,
    wbfVestIou:   settings.wbfVestIou,
    enableClahe:  settings.enableClahe,
    claheClip:    settings.claheClip,
  })
  const tracksRef    = useRef<Track[]>([])
  const pendingRef   = useRef<Detection[] | null>(null)
  const frameWRef    = useRef(1)
  const frameHRef    = useRef(1)

  useEffect(() => {
    configRef.current = {
      condition,
      workerConf:   settings.workerConf,
      helmetConf:   settings.helmetConf,
      vestConf:     settings.vestConf,
      workerNmsIou: settings.workerNmsIou,
      helmetNmsIou: settings.helmetNmsIou,
      vestNmsIou:   settings.vestNmsIou,
      wbfWorkerIou: settings.wbfWorkerIou,
      wbfHelmetIou: settings.wbfHelmetIou,
      wbfVestIou:   settings.wbfVestIou,
      enableClahe:  settings.enableClahe,
      claheClip:    settings.claheClip,
    }
  }, [condition, settings])

  // ── RAF draw loop (identical logic to VideoUploadMode) ───────────────────
  const drawOverlay = useCallback(() => {
    const canvas = overlayRef.current
    const video  = videoRef.current
    if (!canvas || !video) return

    const dw = video.clientWidth
    const dh = video.clientHeight
    if (canvas.width  !== dw) canvas.width  = dw
    if (canvas.height !== dh) canvas.height = dh

    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, dw, dh)

    if (pendingRef.current !== null) {
      tracksRef.current = mergeTracks(tracksRef.current, pendingRef.current)
      pendingRef.current = null
    }

    const { rw, rh, ox, oy } = _letterbox(video.videoWidth, video.videoHeight, dw, dh)
    const fw = frameWRef.current, fh = frameHRef.current
    const LERP = 0.40  // faster snap to new position reduces perceived lag

    for (const t of tracksRef.current) {
      const [x1, y1, x2, y2] = t.frameBox
      const tx1 = (x1 / fw) * rw + ox,  ty1 = (y1 / fh) * rh + oy
      const tx2 = (x2 / fw) * rw + ox,  ty2 = (y2 / fh) * rh + oy

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
      ctx.lineWidth   = 2.5
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

      const video   = videoRef.current
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
      const t0  = performance.now()
      const cfg = configRef.current

      try {
        const form = new FormData()
        form.append('image_b64', b64)
        form.append('condition',  cfg.condition)
        form.append('class_conf', JSON.stringify({
          worker: cfg.workerConf, helmet: cfg.helmetConf, vest: cfg.vestConf,
        }))
        form.append('nms_iou', JSON.stringify({
          worker: cfg.workerNmsIou, helmet: cfg.helmetNmsIou, vest: cfg.vestNmsIou,
        }))
        form.append('wbf_iou', JSON.stringify({
          worker: cfg.wbfWorkerIou, helmet: cfg.wbfHelmetIou, vest: cfg.wbfVestIou,
        }))
        form.append('clahe',      cfg.enableClahe ? '1' : '0')
        form.append('clahe_clip', cfg.claheClip.toString())
        const res = await fetch(`${API}/detect/frame`, { method: 'POST', body: form })
        if (res.ok && isRunningRef.current) {
          const data = await res.json()
          pendingRef.current = data.detections
          setDetections([...data.detections])
          setElapsed(Math.round(performance.now() - t0))
          pushDetections(data.detections, Math.round(performance.now() - t0))
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
    tracksRef.current = []; pendingRef.current = null
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
