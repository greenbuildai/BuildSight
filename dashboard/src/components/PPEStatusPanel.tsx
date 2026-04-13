import { memo } from 'react'
import './PPEStatusPanel.css'

export interface WorkerPPEStatus {
  worker_id:         number
  confidence:        number
  has_helmet:        boolean | null
  has_vest:          boolean | null
  helmet_confidence: number
  vest_confidence:   number
  ppe_compliant:     boolean
  ppe_violation:     boolean
  violation_type:    string[]
  box:               number[]
}

interface PPEStatusPanelProps {
  workers:        WorkerPPEStatus[]
  sceneCondition: string
}

const WorkerRow = memo(({ worker, index }: { worker: WorkerPPEStatus; index: number }) => {
  const helmetStatus = worker.has_helmet === null ? 'unknown' : worker.has_helmet ? 'compliant' : 'violation'
  const vestStatus   = worker.has_vest   === null ? 'unknown' : worker.has_vest   ? 'compliant' : 'violation'
  const rowClass     = worker.ppe_compliant ? 'row-compliant' : worker.ppe_violation ? 'row-violation' : 'row-unknown'

  return (
    <div className={`ppe-worker-row ${rowClass}`}>
      <div className="ppe-worker-id">
        <span className="worker-index">W{index + 1}</span>
        <span className="worker-conf">{Math.round(worker.confidence * 100)}%</span>
      </div>

      <div className="ppe-item">
        <span className="ppe-label">HELMET</span>
        <span className={`ppe-status ppe-${helmetStatus}`}>
          {worker.has_helmet === null ? '—'
            : worker.has_helmet
              ? `\u2713 ${Math.round(worker.helmet_confidence * 100)}%`
              : '\u2717 MISSING'}
        </span>
      </div>

      <div className="ppe-item">
        <span className="ppe-label">VEST</span>
        <span className={`ppe-status ppe-${vestStatus}`}>
          {worker.has_vest === null ? '—'
            : worker.has_vest
              ? `\u2713 ${Math.round(worker.vest_confidence * 100)}%`
              : '\u2717 MISSING'}
        </span>
      </div>

      <div className="ppe-compliance-badge">
        {worker.ppe_compliant
          ? <span className="badge-ok">OK</span>
          : worker.violation_type.length > 0
            ? <span className="badge-violation">{worker.violation_type.join(' ')}</span>
            : <span className="badge-unknown">—</span>}
      </div>
    </div>
  )
})

WorkerRow.displayName = 'WorkerRow'

const PPEStatusPanel = memo(({ workers, sceneCondition }: PPEStatusPanelProps) => {
  const total      = workers.length
  const compliant  = workers.filter(w => w.ppe_compliant).length
  const violations = workers.filter(w => w.ppe_violation).length
  const rate       = total > 0 ? Math.round((compliant / total) * 100) : 0
  const sceneSlug  = sceneCondition.toLowerCase().replace(/_/g, '-')

  return (
    <div className="ppe-status-panel">
      {/* Header */}
      <div className="ppe-panel-header">
        <span className="ppe-panel-title">PPE COMPLIANCE STATUS</span>
        <span className={`ppe-scene-tag ppe-scene-${sceneSlug}`}>
          {sceneCondition.replace(/_/g, ' ')}
        </span>
      </div>

      {/* Summary counts */}
      <div className="ppe-summary-bar">
        <div className="ppe-summary-item">
          <span className="summary-value summary-total">{total}</span>
          <span className="summary-label">DETECTED</span>
        </div>
        <div className="ppe-summary-item">
          <span className="summary-value summary-compliant">{compliant}</span>
          <span className="summary-label">COMPLIANT</span>
        </div>
        <div className="ppe-summary-item">
          <span className="summary-value summary-violation">{violations}</span>
          <span className="summary-label">VIOLATIONS</span>
        </div>
        <div className="ppe-summary-item">
          <span className="summary-value summary-rate">{rate}%</span>
          <span className="summary-label">RATE</span>
        </div>
      </div>

      {/* Compliance progress bar */}
      <div className="ppe-progress-track">
        <div className="ppe-progress-fill" style={{ width: `${rate}%` }} />
      </div>

      {/* Per-worker rows */}
      <div className="ppe-worker-list">
        {workers.length === 0
          ? <div className="ppe-empty">No workers detected</div>
          : workers.map((w, i) => <WorkerRow key={w.worker_id ?? i} worker={w} index={i} />)
        }
      </div>
    </div>
  )
})

PPEStatusPanel.displayName = 'PPEStatusPanel'
export default PPEStatusPanel
