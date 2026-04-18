import { useState, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { useDetectionStats } from '../DetectionStatsContext'

/* ── Types ────────────────────────────────────────────────────────────────── */
interface DailyCompliance {
  day: string
  helmet_pct: number
  vest_pct: number
  workers: number
  violations: number
}

interface ZoneRisk {
  zone_name: string
  risk_level: string
  risk_score: number
  activity: number
  violations?: number
  compliance_score?: number
}

interface DailyReport {
  date: string
  site_id: string
  total_workers: number
  peak_workers: number
  total_violations: number
  avg_compliance: number
  violation_breakdown: Record<string, number>
  top_risk_zones: ZoneRisk[]
}

interface AnalyticsData {
  compliance_trend: DailyCompliance[]
  zone_risks: ZoneRisk[] // For the radar chart (historical or real-time)
  session: {
    frames_scanned: number
    total_workers: number
    helmet_violations: number
    vest_violations: number
    proximity_violations: number
    total_violations: number
    alerts_generated: number
    avg_confidence: number
    model_name: string
    uptime_seconds: number
  }
  violation_breakdown: {
    no_helmet: number
    no_vest: number
    proximity: number
  }
}

/* ── Demo fallback data ───────────────────────────────────────────────────── */
const DEMO_DATA: AnalyticsData = {
  compliance_trend: [
    { day: 'Mon', helmet_pct: 92, vest_pct: 84, workers: 48, violations: 12 },
    { day: 'Tue', helmet_pct: 88, vest_pct: 79, workers: 52, violations: 18 },
    { day: 'Wed', helmet_pct: 95, vest_pct: 88, workers: 45, violations: 8  },
    { day: 'Thu', helmet_pct: 91, vest_pct: 86, workers: 50, violations: 11 },
    { day: 'Fri', helmet_pct: 97, vest_pct: 92, workers: 55, violations: 5  },
    { day: 'Sat', helmet_pct: 89, vest_pct: 82, workers: 38, violations: 14 },
    { day: 'Sun', helmet_pct: 94, vest_pct: 90, workers: 30, violations: 6  },
  ],
  zone_risks: [
    { zone_name: 'LOW RISK PARKING',        risk_score: 22, violations: 4,  workers: 31, compliance_score: 98 },
    { zone_name: 'MODERATE RISK INTERIOR',  risk_score: 45, violations: 8,  workers: 18, compliance_score: 92 },
    { zone_name: 'CRITICAL STAIRCASE',      risk_score: 78, violations: 19, workers: 12, compliance_score: 75 },
    { zone_name: 'CAM-01 VIEW HEIGHT ZONE', risk_score: 65, violations: 14, workers: 22, compliance_score: 82 },
    { zone_name: 'SCAFFOLD PERIMETER',      risk_score: 35, violations: 6,  workers: 15, compliance_score: 88 },
  ],
  session: {
    frames_scanned: 0,
    total_workers: 0,
    helmet_violations: 0,
    vest_violations: 0,
    proximity_violations: 0,
    total_violations: 0,
    alerts_generated: 0,
    avg_confidence: 0,
    model_name: 'BS-ENSEMBLE-WBF',
    uptime_seconds: 0,
  },
  violation_breakdown: { no_helmet: 0, no_vest: 0, proximity: 0 },
}

/* ── Bar Animation ────────────────────────────────────────────────────────── */
function AnimBar({ pct, delay, color }: { pct: number; delay: number; color: string }) {
  return (
    <motion.div
      className="analytics-bar"
      initial={{ height: 0 }}
      animate={{ height: `${pct}%` }}
      transition={{ delay, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      style={{ background: color }}
    />
  )
}

/* ── Radar / Spider Chart (SVG) ───────────────────────────────────────────── */
function RadarChart({ zones }: { zones: ZoneRisk[] }) {
  const cx = 200, cy = 180, r = 100
  const n = zones.length
  if (n === 0) return null

  const angleStep = (2 * Math.PI) / n
  const gridLevels = [0.25, 0.5, 0.75, 1.0]

  const dataPoints = zones.map((z, i) => {
    const angle = i * angleStep - Math.PI / 2
    const pct = Math.min(z.risk_score / 100, 1)
    
    // Dynamically increase offset for bottom labels to prevent overlap
    const isBottom = Math.sin(angle) > 0.5
    const labelOffset = isBottom ? 50 : 42

    return {
      x: cx + r * pct * Math.cos(angle),
      y: cy + r * pct * Math.sin(angle),
      label: z.zone_name,
      lx: cx + (r + labelOffset) * Math.cos(angle),
      ly: cy + (r + labelOffset) * Math.sin(angle),
    }
  })

  const polygonPoints = dataPoints.map(p => `${p.x},${p.y}`).join(' ')

  return (
    <svg viewBox="0 0 400 400" className="radar-chart">
      {/* Grid rings */}
      {gridLevels.map((level) => (
        <polygon
          key={level}
          points={
            Array.from({ length: n }, (_, i) => {
              const a = i * angleStep - Math.PI / 2
              return `${cx + r * level * Math.cos(a)},${cy + r * level * Math.sin(a)}`
            }).join(' ')
          }
          fill="none"
          stroke="var(--color-accent)"
          strokeWidth="0.8"
          strokeOpacity="0.25"
        />
      ))}

      {/* Axis lines */}
      {Array.from({ length: n }, (_, i) => {
        const a = i * angleStep - Math.PI / 2
        return (
          <line
            key={i}
            x1={cx} y1={cy}
            x2={cx + r * Math.cos(a)}
            y2={cy + r * Math.sin(a)}
            stroke="var(--color-accent)"
            strokeWidth="0.8"
            strokeOpacity="0.15"
          />
        )
      })}

      {/* Data polygon */}
      <motion.polygon
        points={polygonPoints}
        fill="var(--accent-a18)"
        stroke="var(--color-accent)"
        strokeWidth="2"
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3, duration: 0.5 }}
        style={{ transformOrigin: `${cx}px ${cy}px` }}
      />

      {/* Data points */}
      {dataPoints.map((p, i) => (
        <motion.circle
          key={i}
          cx={p.x} cy={p.y} r="3.5"
          fill="var(--color-accent)"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.4 + i * 0.08 }}
        />
      ))}

      {/* Labels */}
      {zones.map((z, i) => {
        const angle = i * angleStep - Math.PI / 2
        const isBottom = Math.sin(angle) > 0.5
        const isRight = Math.cos(angle) > 0.2
        const isLeft = Math.cos(angle) < -0.2
        
        let anchor = "middle"
        if (isRight) anchor = "start"
        if (isLeft) anchor = "end"

        let baseline = "central"
        if (Math.sin(angle) < -0.8) baseline = "auto" // Top
        if (Math.sin(angle) > 0.8) baseline = "hanging" // Bottom

        const p = dataPoints[i]
        
        return (
          <text
            key={`label-${i}`}
            x={p.lx} y={p.ly}
            textAnchor={anchor}
            dominantBaseline={baseline}
            className="radar-label"
          >
            {p.label}
          </text>
        )
      })}
    </svg>
  )
}

/* ── Main Analytics Page ──────────────────────────────────────────────────── */
export function AnalyticsPage() {
  const { stats } = useDetectionStats()
  const [data, setData] = useState<AnalyticsData>(DEMO_DATA)
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0])
  const [report, setReport] = useState<DailyReport | null>(null)
  const [isExporting, setIsExporting] = useState(false)

  // Fetch Dashbord Trend (7-day compliance etc)
  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/analytics/dashboard')
        if (res.ok) {
          const json = await res.json()
          setData(prev => ({
            ...prev,
            compliance_trend: json.compliance_trend.map((d: any) => ({
              day: d.date.split('-').pop(), // Just the day number or short date
              helmet_pct: d.avg_compliance,
              vest_pct: d.avg_compliance - 5, // Simulated offset if data is combined
              workers: d.peak_workers,
              violations: d.total_incidents
            })),
            zone_risks: json.zone_risks
          }))
        }
      } catch (err) {
        console.warn('Dashboard API not available', err)
      }
    }
    fetchAnalytics()
  }, [])

  // Fetch Daily Report when date changes
  useEffect(() => {
    const fetchReport = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/analytics/daily-report?date=${selectedDate}`)
        if (res.ok) {
          const json = await res.json()
          setReport(json)
        }
      } catch (err) {
        console.warn('Daily Report API not available')
      }
    }
    fetchReport()
  }, [selectedDate])

  const handleDownloadPDF = async () => {
    setIsExporting(true)
    try {
      window.open(`http://localhost:8000/api/analytics/export/pdf?date=${selectedDate}`, '_blank')
    } finally {
      setIsExporting(false)
    }
  }

  // Merge live detection stats into session summary
  const session = useMemo(() => {
    if (stats.isRunning && stats.framesScanned > 0) {
      const workers = stats.totalWorkers || 1
      return {
        frames_scanned: stats.framesScanned,
        total_workers: stats.totalWorkers,
        helmet_violations: Math.max(0, stats.totalWorkers - stats.helmetsDetected),
        vest_violations: Math.max(0, stats.totalWorkers - stats.vestsDetected),
        proximity_violations: stats.proximityViolations,
        total_violations: Math.max(0, workers - stats.helmetsDetected) + Math.max(0, workers - stats.vestsDetected) + stats.proximityViolations,
        alerts_generated: 0,
        avg_confidence: stats.avgConfidence * 100,
        model_name: stats.modelName || data.session.model_name,
        uptime_seconds: Math.floor(stats.elapsedMs / 1000),
      }
    }
    return data.session
  }, [stats, data.session])

  const violations = useMemo(() => {
    if (stats.isRunning && stats.framesScanned > 0) {
      return {
        no_helmet: Math.max(0, stats.totalWorkers - stats.helmetsDetected),
        no_vest: Math.max(0, stats.totalWorkers - stats.vestsDetected),
        proximity: stats.proximityViolations,
      }
    }
    return data.violation_breakdown
  }, [stats, data.violation_breakdown])

  const totalViolations = violations.no_helmet + violations.no_vest + violations.proximity
  const maxViolation = Math.max(violations.no_helmet, violations.no_vest, violations.proximity, 1)

  return (
    <motion.div 
      className="analytics-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <div className="analytics-header-actions">
        <div className="report-selector">
          <label>End-of-Day Report:</label>
          <input 
            type="date" 
            value={selectedDate} 
            max={new Date().toISOString().split('T')[0]}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="report-date-picker"
          />
        </div>
        <button 
          className={`btn-export ${isExporting ? 'loading' : ''}`}
          onClick={handleDownloadPDF}
        >
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
          </svg>
          {isExporting ? 'Generating...' : 'Download PDF Report'}
        </button>
      </div>

      <div className="analytics-grid">
        {/* ── 7-Day Compliance Drift ──────────────────────────────────────────── */}
        <div className="panel analytics-panel analytics-panel--drift">
          <div className="panel-heading">
            <div>
              <p className="section-label">Historical Performance</p>
              <h3>7-Day Compliance Drift</h3>
            </div>
          </div>
          <div className="analytics-chart-legend">
            <span className="analytics-legend-item"><span className="analytics-legend-swatch analytics-legend-swatch--helmet" /> Helmet</span>
            <span className="analytics-legend-item"><span className="analytics-legend-swatch analytics-legend-swatch--vest" /> Vest</span>
          </div>
          <div className="compliance-chart">
            {data.compliance_trend.map((day, i) => (
              <div className="compliance-chart__day" key={day.day}>
                <div className="compliance-chart__bars">
                  <AnimBar pct={day.helmet_pct} delay={i * 0.08} color="var(--status-ok)" />
                  <AnimBar pct={day.vest_pct} delay={i * 0.08 + 0.04} color="var(--color-accent)" />
                </div>
                <span className="compliance-chart__label">{day.day}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Zone Risk Heatmap ───────────────────────────────────────────────── */}
        <div className="panel analytics-panel analytics-panel--heatmap">
          <div className="panel-heading">
            <div>
              <p className="section-label">Spatial Intelligence</p>
              <h3>Zone Risk Heatmap</h3>
            </div>
          </div>
          <div className="analytics-heatmap-main">
            <RadarChart zones={report?.top_risk_zones || data.zone_risks} />
          </div>
          <div className="zone-risk-table">
            {(report?.top_risk_zones || data.zone_risks).map((z) => (
              <div className="zone-risk-row" key={z.zone_name}>
                <span className="zone-risk-name">{z.zone_name}</span>
                <div className="zone-risk-bar-track">
                  <motion.div
                    className="zone-risk-bar-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${z.risk_score}%` }}
                    transition={{ delay: 0.4, duration: 0.5 }}
                    style={{
                      background: z.risk_score > 60
                        ? 'var(--status-critical)'
                        : z.risk_score > 30
                          ? 'var(--status-warning)'
                          : 'var(--status-ok)',
                    }}
                  />
                </div>
                <span className="zone-risk-score">{z.risk_score}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Session Summary ─────────────────────────────────────────────────── */}
        <div className="panel analytics-panel analytics-panel--session">
          <div className="panel-heading">
            <div>
              <p className="section-label">Current Session</p>
              <h3>Inference Summary</h3>
            </div>
          </div>
          <div className="session-grid">
            {[
              { label: 'Frames Scanned',        value: session.frames_scanned.toLocaleString() },
              { label: 'Total Workers Seen',     value: session.total_workers.toLocaleString() },
              { label: 'Helmet Violations',      value: session.helmet_violations.toString(), tone: 'critical' },
              { label: 'Vest Violations',        value: session.vest_violations.toString(), tone: 'warning' },
              { label: 'Proximity Violations',   value: session.proximity_violations.toString(), tone: 'alert' },
              { label: 'Total Violations',       value: session.total_violations.toString(), tone: 'critical' },
              { label: 'Alerts Generated',       value: session.alerts_generated.toString() },
              { label: 'AI Confidence',          value: `${session.avg_confidence.toFixed(1)}%` },
              { label: 'Active Model',           value: session.model_name },
              { label: 'Uptime',                 value: `${Math.floor(session.uptime_seconds / 60)}m ${session.uptime_seconds % 60}s` },
            ].map((item, i) => (
              <motion.div
                className={`session-stat ${item.tone ? `session-stat--${item.tone}` : ''}`}
                key={item.label}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + i * 0.04 }}
              >
                <span className="session-stat__label">{item.label}</span>
                <strong className="session-stat__value">{item.value}</strong>
              </motion.div>
            ))}
          </div>
        </div>

        {/* ── Violation Breakdown ─────────────────────────────────────────────── */}
        <div className="panel analytics-panel analytics-panel--violations">
          <div className="panel-heading">
            <div>
              <p className="section-label">Compliance Gap</p>
              <h3>Violation Breakdown</h3>
            </div>
            <span className="violation-total">{totalViolations} total</span>
          </div>
          <div className="violation-bars">
            {[
              { label: 'No Helmet',  count: violations.no_helmet,  color: 'var(--status-critical)' },
              { label: 'No Vest',    count: violations.no_vest,    color: 'var(--status-warning)' },
              { label: 'Proximity',  count: violations.proximity,  color: 'var(--status-neutral)' },
            ].map((v, i) => (
              <div className="violation-row" key={v.label}>
                <span className="violation-label">{v.label}</span>
                <div className="violation-bar-track">
                  <motion.div
                    className="violation-bar-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${(v.count / maxViolation) * 100}%` }}
                    transition={{ delay: 0.3 + i * 0.1, duration: 0.5 }}
                    style={{ background: v.color }}
                  />
                </div>
                <span className="violation-count">{v.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  )
}
