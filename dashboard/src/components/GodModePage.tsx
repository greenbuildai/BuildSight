import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { DEFAULT_SETTINGS, useSettings } from '../SettingsContext'
import { useDetectionStats } from '../DetectionStatsContext'
import './GodModePage.css'

type AccessRole = 'admin' | 'reviewer'
type EnvironmentName = 'Demo' | 'Staging' | 'Production'
type SiteMode = 'Auto' | 'Normal' | 'Dusty' | 'Low Light' | 'Crowded'
type SectionKey = 'brain' | 'models' | 'thresholds' | 'site' | 'governance' | 'observability' | 'danger'
type ModelKey = 'visionary' | 'samurai' | 'sentinel'

interface GodModePageProps {
  onBack: () => void
}

interface CameraConfig {
  id: string
  name: string
  location: string
  health: 'healthy' | 'warning' | 'critical'
  fps: number
  enabled: boolean
}

interface ZoneConfig {
  id: string
  name: string
  type: 'Restricted' | 'Warning' | 'Safe Corridor'
  riskScore: number
  active: boolean
}

interface UserRoleConfig {
  id: string
  name: string
  role: AccessRole | 'operator'
  canManageZones: boolean
  canTuneModels: boolean
  canExportConfig: boolean
}

interface GodModeDraft {
  environment: EnvironmentName
  livePreview: boolean
  siteMode: SiteMode
  workerThreshold: number
  helmetThreshold: number
  vestThreshold: number
  restrictedZoneThreshold: number
  unsafeProximityThreshold: number
  alertSensitivity: 'Low' | 'Medium' | 'High'
  escalationMinutes: number
  confidenceCalibration: number
  ensembleSelection: 'Balanced Ensemble' | 'Precision Lock' | 'Speed Priority'
  ensembleWeights: Record<ModelKey, number>
  geoLayers: Record<'riskHeatmap' | 'workerPaths' | 'craneRadius' | 'restrictedZones' | 'emergencyRoutes', boolean>
  cameras: CameraConfig[]
  zones: ZoneConfig[]
  websocketHealthy: boolean
  apiHealthy: boolean
  databaseHealthy: boolean
  cameraStreamsHealthy: boolean
  geoAiHealthy: boolean
  queueLoad: number
  latencyMs: number
  fps: number
  backendKeyStatus: 'Connected' | 'Expiring' | 'Missing'
  users: UserRoleConfig[]
}

interface AuditEntry {
  id: string
  actor: string
  role: AccessRole
  action: string
  detail: string
  severity: 'info' | 'warn' | 'critical'
  timestamp: string
}

interface SnapshotEntry {
  id: string
  name: string
  createdAt: string
  createdBy: string
  config: GodModeDraft
}

interface SectionMeta {
  lastModifiedBy: string
  lastUpdated: string
}

const GATE_KEY = 'buildsight_god_mode_access'
const AUDIT_KEY = 'buildsight_god_mode_audit'
const SNAPSHOT_KEY = 'buildsight_god_mode_snapshots'
const META_KEY = 'buildsight_god_mode_meta'

const ACCESS_CODES: Record<AccessRole, string> = {
  admin: 'jovi#2748',
  reviewer: 'BS-REVIEW-042',
}

const sectionMotion = {
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.18 },
  transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] as const },
}

const SECTION_LABELS: Record<SectionKey, string> = {
  brain: 'System Brain',
  models: 'Model Control',
  thresholds: 'Threshold Control',
  site: 'Site Operations',
  governance: 'Governance',
  observability: 'Observability',
  danger: 'Danger Zone',
}

const FIELD_SECTION_MAP: Record<string, SectionKey> = {
  environment: 'brain',
  livePreview: 'brain',
  siteMode: 'brain',
  workerThreshold: 'thresholds',
  helmetThreshold: 'thresholds',
  vestThreshold: 'thresholds',
  restrictedZoneThreshold: 'thresholds',
  unsafeProximityThreshold: 'thresholds',
  alertSensitivity: 'thresholds',
  escalationMinutes: 'governance',
  confidenceCalibration: 'thresholds',
  ensembleSelection: 'models',
  ensembleWeights: 'models',
  geoLayers: 'site',
  cameras: 'site',
  zones: 'site',
  backendKeyStatus: 'governance',
  users: 'governance',
}

function makeId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
}

function nowIso() {
  return new Date().toISOString()
}

function fmtStamp(ts: string) {
  return new Date(ts).toLocaleString([], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) as T : fallback
  } catch {
    return fallback
  }
}

function cloneDraft(draft: GodModeDraft) {
  return JSON.parse(JSON.stringify(draft)) as GodModeDraft
}

function createInitialDraft(): GodModeDraft {
  return {
    environment: 'Production',
    livePreview: true,
    siteMode: 'Auto',
    workerThreshold: DEFAULT_SETTINGS.workerConf,
    helmetThreshold: DEFAULT_SETTINGS.helmetConf,
    vestThreshold: DEFAULT_SETTINGS.vestConf,
    restrictedZoneThreshold: 0.72,
    unsafeProximityThreshold: 0.66,
    alertSensitivity: 'Medium',
    escalationMinutes: DEFAULT_SETTINGS.escalationMinutes,
    confidenceCalibration: 0.92,
    ensembleSelection: 'Balanced Ensemble',
    ensembleWeights: { visionary: 0.4, samurai: 0.35, sentinel: 0.25 },
    geoLayers: { riskHeatmap: true, workerPaths: true, craneRadius: true, restrictedZones: true, emergencyRoutes: false },
    cameras: [
      { id: 'cam-01', name: 'Tower A Mast', location: 'North Elevation', health: 'healthy', fps: 18, enabled: true },
      { id: 'cam-02', name: 'Main Gate', location: 'Access Control', health: 'healthy', fps: 16, enabled: true },
      { id: 'cam-03', name: 'East Scaffold', location: 'Level 06', health: 'warning', fps: 12, enabled: true },
      { id: 'cam-04', name: 'Crane Blindspot', location: 'South Yard', health: 'critical', fps: 0, enabled: false },
    ],
    zones: [
      { id: 'zone-01', name: 'Hoist Landing', type: 'Restricted', riskScore: 91, active: true },
      { id: 'zone-02', name: 'Crane Radius', type: 'Warning', riskScore: 87, active: true },
      { id: 'zone-03', name: 'Pedestrian Corridor', type: 'Safe Corridor', riskScore: 22, active: true },
    ],
    websocketHealthy: true,
    apiHealthy: true,
    databaseHealthy: true,
    cameraStreamsHealthy: false,
    geoAiHealthy: true,
    queueLoad: 41,
    latencyMs: 48,
    fps: 17,
    backendKeyStatus: 'Connected',
    users: [
      { id: 'u-01', name: 'Jovi', role: 'admin', canManageZones: true, canTuneModels: true, canExportConfig: true },
      { id: 'u-02', name: 'Turner', role: 'reviewer', canManageZones: false, canTuneModels: false, canExportConfig: true },
      { id: 'u-03', name: 'Field Ops Bot', role: 'operator', canManageZones: false, canTuneModels: false, canExportConfig: false },
    ],
  }
}

function defaultSectionMeta(): Record<SectionKey, SectionMeta> {
  const ts = nowIso()
  return {
    brain: { lastModifiedBy: 'BuildSight System', lastUpdated: ts },
    models: { lastModifiedBy: 'BuildSight System', lastUpdated: ts },
    thresholds: { lastModifiedBy: 'BuildSight System', lastUpdated: ts },
    site: { lastModifiedBy: 'BuildSight System', lastUpdated: ts },
    governance: { lastModifiedBy: 'BuildSight System', lastUpdated: ts },
    observability: { lastModifiedBy: 'BuildSight System', lastUpdated: ts },
    danger: { lastModifiedBy: 'BuildSight System', lastUpdated: ts },
  }
}

export function GodModePage({ onBack }: GodModePageProps) {
  const { settings, update } = useSettings()
  const { stats, liveAlerts } = useDetectionStats()
  const [gateRole, setGateRole] = useState<AccessRole>('admin')
  const [accessCode, setAccessCode] = useState('')
  const [gateError, setGateError] = useState<string | null>(null)
  const [session, setSession] = useState<{ role: AccessRole; actor: string } | null>(() => {
    try {
      const raw = sessionStorage.getItem(GATE_KEY)
      return raw ? JSON.parse(raw) as { role: AccessRole; actor: string } : null
    } catch {
      return null
    }
  })
  const [activeSection, setActiveSection] = useState<SectionKey>('brain')
  const [applied, setApplied] = useState<GodModeDraft>(() => {
    const base = createInitialDraft()
    base.workerThreshold = settings.workerConf
    base.helmetThreshold = settings.helmetConf
    base.vestThreshold = settings.vestConf
    base.escalationMinutes = settings.escalationMinutes
    return base
  })
  const [draft, setDraft] = useState<GodModeDraft>(() => cloneDraft(applied))
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>(() => loadJson<AuditEntry[]>(AUDIT_KEY, []))
  const [snapshots, setSnapshots] = useState<SnapshotEntry[]>(() => loadJson<SnapshotEntry[]>(SNAPSHOT_KEY, []))
  const [sectionMeta, setSectionMeta] = useState<Record<SectionKey, SectionMeta>>(() => loadJson(META_KEY, defaultSectionMeta()))

  const readonly = session?.role === 'reviewer'
  const changedKeys = useMemo(() => {
    const before = applied as unknown as Record<string, unknown>
    const after = draft as unknown as Record<string, unknown>
    return Object.keys(after).filter(key => JSON.stringify(before[key]) !== JSON.stringify(after[key]))
  }, [applied, draft])
  const isDirty = changedKeys.length > 0

  const confidenceBins = useMemo(() => {
    const base = Math.max(0.18, stats.avgConfidence)
    return [
      { label: '0.50-0.60', value: Math.round((1 - base) * 38) + 8 },
      { label: '0.60-0.70', value: Math.round((1 - base) * 26) + 14 },
      { label: '0.70-0.80', value: Math.round(base * 32) + 20 },
      { label: '0.80-0.90', value: Math.round(base * 44) + 18 },
      { label: '0.90-1.00', value: Math.round(base * 56) + 12 },
    ]
  }, [stats.avgConfidence])

  const latencySeries = useMemo(() => {
    const base = draft.latencyMs
    return [base + 12, base + 4, base - 6, base + 8, base - 2, base + 5, base - 4, base]
  }, [draft.latencyMs])

  const confidenceSeries = useMemo(() => {
    const base = Math.round(draft.confidenceCalibration * 100)
    return [base - 8, base - 4, base - 2, base + 1, base - 1, base + 3, base + 5, base + 2]
  }, [draft.confidenceCalibration])

  const pipelineStages = [
    {
      label: 'Ingress',
      detail: `${draft.cameras.filter(camera => camera.enabled).length} cameras`,
      state: draft.cameraStreamsHealthy ? 'healthy' : 'warning',
    },
    {
      label: 'Ensemble',
      detail: draft.ensembleSelection,
      state: 'healthy',
    },
    {
      label: 'GeoAI',
      detail: draft.geoLayers.riskHeatmap ? 'Layers active' : 'Reduced layers',
      state: draft.geoAiHealthy ? 'healthy' : 'critical',
    },
    {
      label: 'Escalation',
      detail: `${draft.alertSensitivity} sensitivity`,
      state: draft.queueLoad > 70 ? 'warning' : 'healthy',
    },
  ] as const

  useEffect(() => { localStorage.setItem(AUDIT_KEY, JSON.stringify(auditEntries)) }, [auditEntries])
  useEffect(() => { localStorage.setItem(SNAPSHOT_KEY, JSON.stringify(snapshots)) }, [snapshots])
  useEffect(() => { localStorage.setItem(META_KEY, JSON.stringify(sectionMeta)) }, [sectionMeta])

  function logAudit(action: string, detail: string, severity: AuditEntry['severity'] = 'info') {
    if (!session) return
    setAuditEntries(prev => [{ id: makeId('audit'), actor: session.actor, role: session.role, action, detail, severity, timestamp: nowIso() }, ...prev].slice(0, 60))
  }

  function touchSections(keys: string[]) {
    if (!session) return
    const touched = new Set<SectionKey>()
    keys.forEach(key => { const section = FIELD_SECTION_MAP[key]; if (section) touched.add(section) })
    if (touched.size === 0) return
    const ts = nowIso()
    setSectionMeta(prev => {
      const next = { ...prev }
      touched.forEach(section => { next[section] = { lastModifiedBy: session.actor, lastUpdated: ts } })
      return next
    })
  }

  function unlock() {
    if (accessCode.trim() !== ACCESS_CODES[gateRole]) {
      setGateError('Access code rejected')
      return
    }
    const next = { role: gateRole, actor: gateRole === 'admin' ? 'Jovi' : 'Turner' }
    sessionStorage.setItem(GATE_KEY, JSON.stringify(next))
    setSession(next)
    setAccessCode('')
    setGateError(null)
  }

  function lock() {
    sessionStorage.removeItem(GATE_KEY)
    setSession(null)
  }

  function setDraftField<K extends keyof GodModeDraft>(key: K, value: GodModeDraft[K]) {
    setDraft(prev => ({ ...prev, [key]: value }))
  }

  function updateCamera(cameraId: string, patch: Partial<CameraConfig>) {
    setDraft(prev => ({ ...prev, cameras: prev.cameras.map(camera => camera.id === cameraId ? { ...camera, ...patch } : camera) }))
  }

  function updateZone(zoneId: string, patch: Partial<ZoneConfig>) {
    setDraft(prev => ({ ...prev, zones: prev.zones.map(zone => zone.id === zoneId ? { ...zone, ...patch } : zone) }))
  }

  function updateUser(userId: string, patch: Partial<UserRoleConfig>) {
    setDraft(prev => ({ ...prev, users: prev.users.map(user => user.id === userId ? { ...user, ...patch } : user) }))
  }

  function applyChanges() {
    if (!session || readonly || !isDirty) return
    if (!window.confirm(`Apply ${changedKeys.length} God Mode changes?`)) return
    update('workerConf', draft.workerThreshold)
    update('helmetConf', draft.helmetThreshold)
    update('vestConf', draft.vestThreshold)
    update('escalationMinutes', draft.escalationMinutes)
    update('alertSensitivity', draft.alertSensitivity.toLowerCase() as 'low' | 'medium' | 'high')
    update('selectedModel', draft.ensembleSelection)
    update('siteCondition', draft.siteMode)
    setApplied(cloneDraft(draft))
    touchSections(changedKeys)
    logAudit('CONFIG_APPLIED', `Applied ${changedKeys.length} fields: ${changedKeys.join(', ')}`)
  }

  function discardChanges() {
    if (!isDirty) return
    if (!window.confirm('Discard unsaved God Mode changes?')) return
    setDraft(cloneDraft(applied))
    logAudit('DRAFT_DISCARDED', 'Reverted pending God Mode edits', 'warn')
  }

  function saveSnapshot() {
    if (!session) return
    const name = window.prompt('Snapshot name', `Snapshot ${new Date().toLocaleDateString()}`)
    if (!name) return
    setSnapshots(prev => [{ id: makeId('snapshot'), name, createdAt: nowIso(), createdBy: session.actor, config: cloneDraft(draft) }, ...prev].slice(0, 20))
    logAudit('SNAPSHOT_SAVED', `Saved snapshot "${name}"`)
  }

  function restoreSnapshot(snapshot: SnapshotEntry) {
    if (readonly) return
    if (!window.confirm(`Restore snapshot "${snapshot.name}" into draft?`)) return
    setDraft(cloneDraft(snapshot.config))
    logAudit('SNAPSHOT_RESTORED', `Restored snapshot "${snapshot.name}"`, 'warn')
  }

  function exportConfig() {
    if (!session) return
    const payload = JSON.stringify({ exportedAt: nowIso(), exportedBy: session.actor, config: draft }, null, 2)
    const blob = new Blob([payload], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `buildsight-god-mode-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    logAudit('CONFIG_EXPORTED', 'Exported God Mode configuration')
  }

  function resetThresholds() {
    if (readonly || !window.confirm('Reset threshold controls to defaults?')) return
    setDraft(prev => ({
      ...prev,
      workerThreshold: DEFAULT_SETTINGS.workerConf,
      helmetThreshold: DEFAULT_SETTINGS.helmetConf,
      vestThreshold: DEFAULT_SETTINGS.vestConf,
      restrictedZoneThreshold: 0.72,
      unsafeProximityThreshold: 0.66,
      confidenceCalibration: 0.92,
    }))
    logAudit('THRESHOLDS_RESET', 'Reset threshold controls to defaults', 'critical')
  }

  function disableAlerts() {
    if (readonly || !window.confirm('Reduce alert sensitivity and escalation pressure?')) return
    setDraft(prev => ({ ...prev, alertSensitivity: 'Low', escalationMinutes: 30 }))
    logAudit('ALERTS_DISABLED', 'Reduced alert pressure', 'critical')
  }

  function removeOfflineCameras() {
    if (readonly || !window.confirm('Remove offline cameras from the draft?')) return
    setDraft(prev => ({ ...prev, cameras: prev.cameras.filter(camera => camera.enabled && camera.health !== 'critical') }))
    logAudit('CAMERAS_REMOVED', 'Removed offline cameras from draft', 'critical')
  }

  function rollbackModels() {
    if (readonly || !window.confirm('Rollback model routing to the stable ensemble?')) return
    setDraft(prev => ({
      ...prev,
      ensembleSelection: 'Balanced Ensemble',
      ensembleWeights: { visionary: 0.34, samurai: 0.33, sentinel: 0.33 },
    }))
    logAudit('MODEL_ROLLBACK', 'Rolled routing back to stable ensemble', 'critical')
  }

  const previewCounts = {
    workers: Math.max(0, stats.totalWorkers - Math.round(draft.workerThreshold * 10)),
    helmets: Math.max(0, stats.helmetsDetected - Math.round(draft.helmetThreshold * 8)),
    vests: Math.max(0, stats.vestsDetected - Math.round(draft.vestThreshold * 8)),
  }

  const sectionNav = (Object.keys(SECTION_LABELS) as SectionKey[]).map(id => ({ id, label: SECTION_LABELS[id] }))

  if (!session) {
    return (
      <main className="dashboard dashboard--godmode">
        <header className="topbar">
          <div className="topbar__brand">
            <p className="eyebrow">Restricted Platform Control</p>
            <h2>God Mode</h2>
          </div>
          <div className="topbar__status">
            <button className="stg-back-btn" onClick={onBack}>
              <span className="stg-back-btn__icon">&larr;</span>
              <span>Back to Dashboard</span>
            </button>
          </div>
        </header>
        <section className="god-mode-shell god-mode-shell--gate">
          <motion.div className="panel god-gate" initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}>
            <div className="panel-heading"><div><p className="section-label">Admin Access</p><h3>Secure Entry Gate</h3></div></div>
            <div className="god-gate__body">
              <p className="god-gate__copy">God Mode exposes BuildSight thresholds, model routing, zones, roles, rollback controls, and audit history. This workspace is restricted to admins and reviewers.</p>
              <div className="god-grid">
                <label className="gm-field"><span>Access Role</span><select value={gateRole} onChange={event => setGateRole(event.target.value as AccessRole)}><option value="admin">Admin</option><option value="reviewer">Reviewer (read only)</option></select></label>
                <label className="gm-field"><span>Access Code</span><input type="password" value={accessCode} placeholder="Enter secure token" onChange={event => setAccessCode(event.target.value)} /></label>
              </div>
              {gateError && <p className="gm-error">{gateError}</p>}
              <div className="god-action-row">
                <button className="gm-btn gm-btn--ghost" onClick={onBack}>Exit</button>
                <button className="gm-btn gm-btn--primary" onClick={unlock}>Unlock God Mode</button>
              </div>
            </div>
          </motion.div>
        </section>
      </main>
    )
  }

  return (
    <main className="dashboard dashboard--godmode">
      <header className="topbar">
        <div className="topbar__brand">
          <p className="eyebrow">BuildSight Private Operating Center</p>
          <h2>God Mode</h2>
        </div>
        <div className="topbar__status">
          <div className={`gm-env-pill gm-env-pill--${draft.environment.toLowerCase()}`}>{draft.environment}</div>
          <div className="telemetry-block"><span>Role</span><strong>{readonly ? 'Reviewer' : 'Admin'}</strong></div>
          <div className="telemetry-block"><span>Active Cameras</span><strong>{draft.cameras.filter(camera => camera.enabled).length}</strong></div>
          <button className="gm-btn gm-btn--ghost" onClick={lock}>Lock</button>
          <button className="stg-back-btn" onClick={onBack}><span className="stg-back-btn__icon">&larr;</span><span>Back to Dashboard</span></button>
        </div>
      </header>

      <section className="god-mode-shell">
        <motion.aside className="panel god-nav" initial={{ opacity: 0, x: -18 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}>
          <div className="panel-heading"><div><p className="section-label">Internal Navigation</p><h3>Command Layers</h3></div></div>
          <nav className="god-nav__links">{sectionNav.map(section => <a key={section.id} href={`#${section.id}`} className={`god-nav__link ${activeSection === section.id ? 'god-nav__link--active' : ''}`} onClick={() => setActiveSection(section.id)}>{section.label}</a>)}</nav>
          <div className="god-nav__status">
            <div className="gm-health-pill"><span>WebSocket</span><strong>{draft.websocketHealthy ? 'Connected' : 'Dropped'}</strong></div>
            <div className="gm-health-pill"><span>API</span><strong>{draft.apiHealthy ? 'Healthy' : 'Retrying'}</strong></div>
            <div className="gm-health-pill"><span>DB</span><strong>{draft.databaseHealthy ? 'Synced' : 'Write failure'}</strong></div>
          </div>
        </motion.aside>

        <motion.div className="god-content" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.35 }}>
          <motion.section id="brain" className="panel god-section" {...sectionMotion}>
            <div className="panel-heading"><div><p className="section-label">Brain of BuildSight</p><h3>System Brain</h3></div></div>
            <div className="gm-meta-bar"><span>Last modified by {sectionMeta.brain.lastModifiedBy}</span><span>{fmtStamp(sectionMeta.brain.lastUpdated)}</span></div>
            <div className="god-grid">
              <div className="god-card god-card--wide">
                <div className="god-card__eyebrow">Current model pipeline flow</div>
                <div className="god-pipeline">
                  {pipelineStages.map((stage, index) => (
                    <div key={stage.label} className={`god-pipeline__stage god-pipeline__stage--${stage.state}`}>
                      <span className="god-pipeline__pulse" />
                      <em>{stage.label}</em>
                      <strong>{stage.detail}</strong>
                      {index < pipelineStages.length - 1 && <i className="god-pipeline__arrow">-&gt;</i>}
                    </div>
                  ))}
                </div>
              </div>
              <div className="god-card"><span>Queue Load</span><strong>{draft.queueLoad}%</strong><em>{draft.queueLoad > 70 ? 'critical' : draft.queueLoad > 45 ? 'warning' : 'healthy'}</em></div>
              <div className="god-card"><span>Latency</span><strong>{draft.latencyMs}ms</strong><em>{draft.latencyMs > 90 ? 'critical' : draft.latencyMs > 55 ? 'warning' : 'healthy'}</em></div>
              <div className="god-card"><span>System FPS</span><strong>{draft.fps}</strong><em>{draft.fps < 10 ? 'critical' : draft.fps < 16 ? 'warning' : 'healthy'}</em></div>
              <div className="god-card"><span>Site Mode</span><strong>{draft.siteMode}</strong><em>{draft.livePreview ? 'preview enabled' : 'applied only'}</em></div>
              <div className="god-card god-card--wide">
                <div className="god-card__eyebrow">Live health indicators</div>
                <div className="god-health-ribbon">
                  <div className={`god-health-ribbon__item ${draft.websocketHealthy ? 'is-healthy' : 'is-critical'}`}><span className="god-health-ribbon__dot" />WebSocket</div>
                  <div className={`god-health-ribbon__item ${draft.apiHealthy ? 'is-healthy' : 'is-critical'}`}><span className="god-health-ribbon__dot" />API</div>
                  <div className={`god-health-ribbon__item ${draft.databaseHealthy ? 'is-healthy' : 'is-critical'}`}><span className="god-health-ribbon__dot" />Database</div>
                  <div className={`god-health-ribbon__item ${draft.cameraStreamsHealthy ? 'is-healthy' : 'is-warning'}`}><span className="god-health-ribbon__dot" />Camera Streams</div>
                  <div className={`god-health-ribbon__item ${draft.geoAiHealthy ? 'is-healthy' : 'is-critical'}`}><span className="god-health-ribbon__dot" />GeoAI</div>
                </div>
                <div className="god-inline-duo">
                  <label className="gm-field"><span>Environment</span><select disabled={readonly} value={draft.environment} onChange={event => setDraftField('environment', event.target.value as EnvironmentName)}><option>Demo</option><option>Staging</option><option>Production</option></select></label>
                  <label className="gm-toggle"><input type="checkbox" checked={draft.livePreview} disabled={readonly} onChange={event => setDraftField('livePreview', event.target.checked)} /><span>Live preview mode</span></label>
                </div>
              </div>
            </div>
          </motion.section>

          <motion.section id="models" className="panel god-section" {...sectionMotion}>
            <div className="panel-heading"><div><p className="section-label">Model routing and rollback</p><h3>Model Control</h3></div></div>
            <div className="gm-meta-bar"><span>Last modified by {sectionMeta.models.lastModifiedBy}</span><span>{fmtStamp(sectionMeta.models.lastUpdated)}</span></div>
            <div className="god-grid">
              <div className="god-card">
                <label className="gm-field"><span>Selection profile</span><select disabled={readonly} value={draft.ensembleSelection} onChange={event => setDraftField('ensembleSelection', event.target.value as GodModeDraft['ensembleSelection'])}><option>Balanced Ensemble</option><option>Precision Lock</option><option>Speed Priority</option></select></label>
                {(['visionary', 'samurai', 'sentinel'] as ModelKey[]).map(key => <label key={key} className="gm-slider"><div><span>{key}</span><strong>{Math.round(draft.ensembleWeights[key] * 100)}%</strong></div><input type="range" min={0} max={1} step={0.01} value={draft.ensembleWeights[key]} disabled={readonly} onChange={event => setDraftField('ensembleWeights', { ...draft.ensembleWeights, [key]: Number(event.target.value) })} /></label>)}
              </div>
              <div className="god-card">
                {[
                  ['Visionary PPE', `${Math.round(draft.ensembleWeights.visionary * 100)}% weight`, 'Active'],
                  ['Samurai Context', `${Math.round(draft.ensembleWeights.samurai * 100)}% weight`, 'Active'],
                  ['Sentinel Guard', `${Math.round(draft.ensembleWeights.sentinel * 100)}% weight`, 'Standby'],
                ].map(([name, detail, status]) => <div key={name} className="god-table__row"><div><strong>{name}</strong><span>{detail}</span></div><em>{status}</em></div>)}
                <div className="god-action-row"><button className="gm-btn gm-btn--ghost" disabled={readonly} onClick={rollbackModels}>Rollback to stable</button><button className="gm-btn gm-btn--ghost" disabled={readonly}>Restore previous bundle</button></div>
              </div>
            </div>
          </motion.section>

          <motion.section id="thresholds" className="panel god-section" {...sectionMotion}>
            <div className="panel-heading"><div><p className="section-label">Detection and alert tuning</p><h3>Threshold Control</h3></div></div>
            <div className="gm-meta-bar"><span>Last modified by {sectionMeta.thresholds.lastModifiedBy}</span><span>{fmtStamp(sectionMeta.thresholds.lastUpdated)}</span></div>
            <div className="god-grid">
              <div className="god-card">
                {[
                  ['Worker detection', 'workerThreshold'],
                  ['Helmet detection', 'helmetThreshold'],
                  ['Safety vest detection', 'vestThreshold'],
                  ['Restricted zone alerts', 'restrictedZoneThreshold'],
                  ['Unsafe proximity alerts', 'unsafeProximityThreshold'],
                ].map(([label, key]) => <label key={key} className="gm-slider"><div><span>{label}</span><strong>{Math.round((draft[key as keyof GodModeDraft] as number) * 100)}%</strong></div><input type="range" min={0} max={1} step={0.01} value={draft[key as keyof GodModeDraft] as number} disabled={readonly} onChange={event => setDraftField(key as keyof GodModeDraft, Number(event.target.value) as never)} /></label>)}
              </div>
              <div className="god-card">
                <label className="gm-field"><span>Alert sensitivity</span><select disabled={readonly} value={draft.alertSensitivity} onChange={event => setDraftField('alertSensitivity', event.target.value as GodModeDraft['alertSensitivity'])}><option>Low</option><option>Medium</option><option>High</option></select></label>
                <label className="gm-slider"><div><span>Confidence calibration</span><strong>{Math.round(draft.confidenceCalibration * 100)}%</strong></div><input type="range" min={0} max={1} step={0.01} value={draft.confidenceCalibration} disabled={readonly} onChange={event => setDraftField('confidenceCalibration', Number(event.target.value))} /></label>
                <label className="gm-slider"><div><span>Escalation timeout</span><strong>{draft.escalationMinutes} min</strong></div><input type="range" min={1} max={30} step={1} value={draft.escalationMinutes} disabled={readonly} onChange={event => setDraftField('escalationMinutes', Number(event.target.value))} /></label>
              </div>
              <div className="god-card god-card--wide">
                <div className="god-preview-grid">
                  <div><span>Workers</span><strong>{draft.livePreview ? previewCounts.workers : stats.totalWorkers}</strong><em>{draft.livePreview ? 'draft preview' : 'applied'}</em></div>
                  <div><span>Helmets</span><strong>{draft.livePreview ? previewCounts.helmets : stats.helmetsDetected}</strong><em>{draft.livePreview ? 'draft preview' : 'applied'}</em></div>
                  <div><span>Vests</span><strong>{draft.livePreview ? previewCounts.vests : stats.vestsDetected}</strong><em>{draft.livePreview ? 'draft preview' : 'applied'}</em></div>
                </div>
              </div>
            </div>
          </motion.section>

          <motion.section id="site" className="panel god-section" {...sectionMotion}>
            <div className="panel-heading"><div><p className="section-label">Cameras, GeoAI, and zones</p><h3>Site Operations</h3></div></div>
            <div className="gm-meta-bar"><span>Last modified by {sectionMeta.site.lastModifiedBy}</span><span>{fmtStamp(sectionMeta.site.lastUpdated)}</span></div>
            <div className="god-grid">
              <div className="god-card">
                <div className="gm-chip-row">{(['Auto', 'Normal', 'Dusty', 'Low Light', 'Crowded'] as SiteMode[]).map(mode => <button key={mode} className={`gm-chip ${draft.siteMode === mode ? 'gm-chip--active' : ''}`} disabled={readonly} onClick={() => setDraftField('siteMode', mode)}>{mode}</button>)}</div>
                {Object.entries(draft.geoLayers).map(([key, value]) => <label key={key} className="gm-toggle"><input type="checkbox" checked={value} disabled={readonly} onChange={event => setDraftField('geoLayers', { ...draft.geoLayers, [key]: event.target.checked })} /><span>{key}</span></label>)}
              </div>
              <div className="god-card god-card--wide">{draft.cameras.map(camera => <div key={camera.id} className="god-table__row"><div><strong>{camera.name}</strong><span>{camera.location}</span></div><span>{camera.fps} fps</span><em>{camera.health}</em><label className="gm-toggle"><input type="checkbox" checked={camera.enabled} disabled={readonly} onChange={event => updateCamera(camera.id, { enabled: event.target.checked })} /><span>{camera.enabled ? 'Enabled' : 'Disabled'}</span></label></div>)}</div>
              <div className="god-card god-card--wide">
                <div className="god-action-row"><button className="gm-btn gm-btn--ghost" disabled={readonly} onClick={() => setDraft(prev => ({ ...prev, zones: [...prev.zones, { id: makeId('zone'), name: `New Zone ${prev.zones.length + 1}`, type: 'Warning', riskScore: 50, active: true }] }))}>Create zone</button></div>
                {draft.zones.map(zone => <div key={zone.id} className="god-table__row"><input value={zone.name} disabled={readonly} onChange={event => updateZone(zone.id, { name: event.target.value })} /><select value={zone.type} disabled={readonly} onChange={event => updateZone(zone.id, { type: event.target.value as ZoneConfig['type'] })}><option>Restricted</option><option>Warning</option><option>Safe Corridor</option></select><input type="number" min={0} max={100} value={zone.riskScore} disabled={readonly} onChange={event => updateZone(zone.id, { riskScore: Number(event.target.value) })} /><label className="gm-toggle"><input type="checkbox" checked={zone.active} disabled={readonly} onChange={event => updateZone(zone.id, { active: event.target.checked })} /><span>{zone.active ? 'Active' : 'Paused'}</span></label></div>)}
              </div>
            </div>
          </motion.section>

          <motion.section id="governance" className="panel god-section" {...sectionMotion}>
            <div className="panel-heading"><div><p className="section-label">Audit, snapshots, and access control</p><h3>Governance</h3></div></div>
            <div className="gm-meta-bar"><span>Last modified by {sectionMeta.governance.lastModifiedBy}</span><span>{fmtStamp(sectionMeta.governance.lastUpdated)}</span></div>
            <div className="god-grid">
              <div className="god-card"><div className="gm-health-pill"><span>Backend key</span><strong>{draft.backendKeyStatus}</strong></div><div className="gm-health-pill"><span>FastAPI</span><strong>{draft.apiHealthy ? 'localhost:8000' : 'Offline'}</strong></div><div className="gm-health-pill"><span>Database</span><strong>{draft.databaseHealthy ? 'Audit writes healthy' : 'Write failures'}</strong></div></div>
              <div className="god-card"><div className="god-action-row"><button className="gm-btn gm-btn--ghost" onClick={saveSnapshot}>Save snapshot</button><button className="gm-btn gm-btn--ghost" onClick={exportConfig}>Export config</button></div>{snapshots.length === 0 && <div className="god-empty">No snapshots saved yet.</div>}{snapshots.map(snapshot => <div key={snapshot.id} className="god-table__row"><div><strong>{snapshot.name}</strong><span>{snapshot.createdBy}</span></div><span>{fmtStamp(snapshot.createdAt)}</span><button className="gm-btn gm-btn--ghost" disabled={readonly} onClick={() => restoreSnapshot(snapshot)}>Restore</button></div>)}</div>
              <div className="god-card god-card--wide">{draft.users.map(user => <div key={user.id} className="god-table__row"><div><strong>{user.name}</strong><span>{user.id}</span></div><select value={user.role} disabled={readonly} onChange={event => updateUser(user.id, { role: event.target.value as UserRoleConfig['role'] })}><option value="admin">Admin</option><option value="reviewer">Reviewer</option><option value="operator">Operator</option></select><label className="gm-toggle"><input type="checkbox" checked={user.canManageZones} disabled={readonly} onChange={event => updateUser(user.id, { canManageZones: event.target.checked })} /><span>Zones</span></label><label className="gm-toggle"><input type="checkbox" checked={user.canTuneModels} disabled={readonly} onChange={event => updateUser(user.id, { canTuneModels: event.target.checked })} /><span>Models</span></label><label className="gm-toggle"><input type="checkbox" checked={user.canExportConfig} disabled={readonly} onChange={event => updateUser(user.id, { canExportConfig: event.target.checked })} /><span>Export</span></label></div>)}</div>
            </div>
          </motion.section>

          <motion.section id="observability" className="panel god-section" {...sectionMotion}>
            <div className="panel-heading"><div><p className="section-label">Runtime insight</p><h3>Observability</h3></div></div>
            <div className="gm-meta-bar"><span>Last modified by {sectionMeta.observability.lastModifiedBy}</span><span>{fmtStamp(sectionMeta.observability.lastUpdated)}</span></div>
            <div className="god-grid">
              <div className="god-card">
                <div className="god-card__eyebrow">Real-time confidence distribution</div>
                {confidenceBins.map(bin => (
                  <div key={bin.label} className="god-bar">
                    <span>{bin.label}</span>
                    <div className="god-bar__track">
                      <div className="god-bar__fill" style={{ width: `${Math.min(100, bin.value)}%` }} />
                    </div>
                    <strong>{bin.value}</strong>
                  </div>
                ))}
              </div>
              <div className="god-card">
                <div className="god-card__eyebrow">Engine trend traces</div>
                <div className="god-spark-grid">
                  <div className="god-spark-card">
                    <span>Latency trend</span>
                    <svg viewBox="0 0 220 72" className="god-spark">
                      <polyline
                        fill="none"
                        stroke="var(--color-accent)"
                        strokeWidth="3"
                        points={latencySeries.map((value, index) => `${index * 31},${68 - Math.min(60, value * 0.7)}`).join(' ')}
                      />
                    </svg>
                    <strong>{draft.latencyMs}ms live</strong>
                  </div>
                  <div className="god-spark-card">
                    <span>Confidence trend</span>
                    <svg viewBox="0 0 220 72" className="god-spark">
                      <polyline
                        fill="none"
                        stroke="var(--status-ok)"
                        strokeWidth="3"
                        points={confidenceSeries.map((value, index) => `${index * 31},${72 - Math.min(64, value * 0.62)}`).join(' ')}
                      />
                    </svg>
                    <strong>{Math.round(draft.confidenceCalibration * 100)}% calibrated</strong>
                  </div>
                </div>
                <div className="god-preview-grid"><div><span>False positive trend</span><strong>{Math.max(4, Math.round((1 - draft.confidenceCalibration) * 40))}%</strong></div><div><span>False negative trend</span><strong>{Math.max(5, Math.round(draft.workerThreshold * 20))}%</strong></div></div>
              </div>
              <div className="god-card god-card--wide">{[...liveAlerts.slice(0, 3).map(alert => ({ id: alert.id, message: `${alert.severity.toUpperCase()} ${alert.title} @ ${alert.camera}`, timestamp: alert.time })), ...auditEntries.slice(0, 5).map(entry => ({ id: entry.id, message: `${entry.action}: ${entry.detail}`, timestamp: fmtStamp(entry.timestamp) }))].map(entry => <div key={entry.id} className="god-log-entry"><span>{entry.timestamp}</span><strong>{entry.message}</strong></div>)}</div>
            </div>
          </motion.section>

          <motion.section id="danger" className="panel god-section god-section--danger" {...sectionMotion}>
            <div className="panel-heading"><div><p className="section-label">Critical controls</p><h3>Danger Zone</h3></div></div>
            <div className="gm-meta-bar"><span>Last modified by {sectionMeta.danger.lastModifiedBy}</span><span>{fmtStamp(sectionMeta.danger.lastUpdated)}</span></div>
            <div className="god-danger-grid">
              <button className="gm-danger-card" disabled={readonly} onClick={resetThresholds}><strong>Reset thresholds</strong><span>Revert all detection and alert tuning to defaults.</span></button>
              <button className="gm-danger-card" disabled={readonly} onClick={disableAlerts}><strong>Disable alerts</strong><span>Reduce sensitivity and slow escalation pressure.</span></button>
              <button className="gm-danger-card" disabled={readonly} onClick={removeOfflineCameras}><strong>Remove cameras</strong><span>Drop all offline or critical feeds from the draft.</span></button>
              <button className="gm-danger-card" disabled={readonly} onClick={rollbackModels}><strong>Rollback model versions</strong><span>Revert routing to the last stable ensemble profile.</span></button>
            </div>
          </motion.section>
        </motion.div>
      </section>

      {isDirty && <motion.div className="gm-sticky-bar" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}><div><strong>{changedKeys.length} unsaved changes</strong><span>{readonly ? 'Reviewer mode cannot apply changes.' : 'Draft changes stay in God Mode until applied.'}</span></div><div className="god-action-row"><button className="gm-btn gm-btn--ghost" onClick={discardChanges}>Discard changes</button><button className="gm-btn gm-btn--primary" disabled={readonly} onClick={applyChanges}>Apply changes</button></div></motion.div>}
    </main>
  )
}
