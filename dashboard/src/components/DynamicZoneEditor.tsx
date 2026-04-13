/* ─────────────────────────────────────────────────────────────────────────────
 * DynamicZoneEditor — operator-controlled geofence management panel
 *
 * Features:
 *  - List active/inactive dynamic zones with risk badge and colour swatch
 *  - Toggle zone active state
 *  - Delete zones
 *  - Create new zones via preset templates + name / risk-level form
 *  - Trigger SAM auto-detect (calls /api/geoai/sam/detect) and add results
 * ──────────────────────────────────────────────────────────────────────────── */

import { useState, useEffect, useCallback } from 'react'
import type { DynamicZone, DynamicZoneCreate, ZoneRiskLevel, ZoneType } from '../types/geoai'
import './DynamicZoneEditor.css'

const API_BASE = 'http://localhost:8000/api/geoai'

// ── Risk colour map (mirrors backend _RISK_COLORS) ────────────────────────
const RISK_COLORS: Record<string, string> = {
  low:      '#00e676',
  moderate: '#ffd600',
  high:     '#ff7b00',
  critical: '#ff3b3b',
  safe:     '#00b4d8',
}

// ── Zone type labels ──────────────────────────────────────────────────────
const ZONE_TYPE_LABELS: Record<string, string> = {
  restricted: 'RESTRICTED',
  work:       'WORK ZONE',
  hazard:     'HAZARD',
  safe:       'SAFE ZONE',
  custom:     'CUSTOM',
}

// ── Template list fetched once on mount ───────────────────────────────────
type TemplateMap = Record<string, [number, number][]>

interface Props {
  /** Called whenever the zone list changes so GeoAIMap can re-render. */
  onZonesChange?: (zones: DynamicZone[]) => void
}

export function DynamicZoneEditor({ onZonesChange }: Props) {
  const [zones, setZones]           = useState<DynamicZone[]>([])
  const [templates, setTemplates]   = useState<TemplateMap>({})
  const [loading, setLoading]       = useState(true)
  const [samRunning, setSamRunning] = useState(false)
  const [samStatus, setSamStatus]   = useState<boolean | null>(null)
  const [error, setError]           = useState<string | null>(null)

  // ── New zone form state ────────────────────────────────────────────────
  const [formOpen, setFormOpen]     = useState(false)
  const [formName, setFormName]     = useState('')
  const [formRisk, setFormRisk]     = useState<ZoneRiskLevel>('moderate')
  const [formType, setFormType]     = useState<ZoneType>('restricted')
  const [formTemplate, setFormTemplate] = useState('')
  const [formSaving, setFormSaving] = useState(false)

  // ── Load zones + templates ─────────────────────────────────────────────
  const loadZones = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/dynamic-zones`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: DynamicZone[] = await res.json()
      setZones(data)
      onZonesChange?.(data)
      setError(null)
    } catch {
      setError('Could not load zones — backend may be offline')
    } finally {
      setLoading(false)
    }
  }, [onZonesChange])

  useEffect(() => {
    loadZones()
    // Load templates and SAM status in parallel
    fetch(`${API_BASE}/dynamic-zones/templates`)
      .then(r => r.json())
      .then(d => setTemplates(d.templates || {}))
      .catch(() => {})

    fetch(`${API_BASE}/sam/status`)
      .then(r => r.json())
      .then(d => setSamStatus(d.available))
      .catch(() => setSamStatus(false))
  }, [loadZones])

  // ── Toggle active state ────────────────────────────────────────────────
  const toggleActive = async (zone: DynamicZone) => {
    try {
      const res = await fetch(`${API_BASE}/dynamic-zones/${zone.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_active: !zone.is_active }),
      })
      if (!res.ok) throw new Error()
      await loadZones()
    } catch {
      setError(`Failed to update zone "${zone.name}"`)
    }
  }

  // ── Delete zone ────────────────────────────────────────────────────────
  const deleteZone = async (zone: DynamicZone) => {
    if (!confirm(`Delete zone "${zone.name}"?`)) return
    try {
      const res = await fetch(`${API_BASE}/dynamic-zones/${zone.id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error()
      await loadZones()
    } catch {
      setError(`Failed to delete zone "${zone.name}"`)
    }
  }

  // ── Create zone ────────────────────────────────────────────────────────
  const createZone = async () => {
    const name = formName.trim()
    if (!name) { setError('Zone name is required'); return }
    if (!formTemplate && Object.keys(templates).length > 0) {
      setError('Select a template polygon'); return
    }

    const coords: [number, number][] = formTemplate
      ? templates[formTemplate]
      : [[78.66873, 10.81660], [78.66884, 10.81660], [78.66884, 10.81666], [78.66873, 10.81666]]

    const payload: DynamicZoneCreate = {
      name,
      risk_level: formRisk,
      zone_type:  formType,
      coordinates: coords,
      description: `${ZONE_TYPE_LABELS[formType]} zone — ${formRisk} risk`,
    }

    setFormSaving(true)
    try {
      const res = await fetch(`${API_BASE}/dynamic-zones`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setFormName('')
      setFormRisk('moderate')
      setFormType('restricted')
      setFormTemplate('')
      setFormOpen(false)
      setError(null)
      await loadZones()
    } catch (err) {
      setError(`Create failed — ${err}`)
    } finally {
      setFormSaving(false)
    }
  }

  // ── SAM auto-detect ────────────────────────────────────────────────────
  const runSamDetect = async () => {
    setSamRunning(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/sam/detect`, { method: 'POST' })
      const data = await res.json()
      if (!res.ok || data.status !== 'ok') {
        setError(data.message || 'SAM detect failed')
        return
      }
      if (data.count === 0) {
        setError('SAM found no segments in current frame')
        return
      }
      // Create a dynamic zone for each SAM feature
      const creates = (data.features as any[]).map((f, i) =>
        fetch(`${API_BASE}/dynamic-zones`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: `SAM Zone ${i + 1}`,
            risk_level: 'moderate',
            zone_type: 'custom',
            coordinates: f.geometry.coordinates[0],
            description: `Auto-detected by SAM (score ${f.properties.confidence?.toFixed(2) ?? '?'})`,
          } as DynamicZoneCreate),
        })
      )
      await Promise.all(creates)
      await loadZones()
    } catch {
      setError('SAM detect request failed')
    } finally {
      setSamRunning(false)
    }
  }

  return (
    <div className="dze">
      {/* Header */}
      <div className="dze__header">
        <span className="dze__title">DYNAMIC ZONES</span>
        <div className="dze__header-actions">
          <button
            className={`dze__btn dze__btn--sam ${samRunning ? 'dze__btn--loading' : ''}`}
            onClick={runSamDetect}
            disabled={samRunning || samStatus === false}
            title={samStatus === false ? 'SAM not available — place sam_vit_b.pth in weights/' : 'Auto-detect zones with SAM'}
          >
            {samRunning ? 'DETECTING…' : 'SAM DETECT'}
          </button>
          <button className="dze__btn dze__btn--add" onClick={() => setFormOpen(o => !o)}>
            {formOpen ? 'CANCEL' : '+ ADD ZONE'}
          </button>
        </div>
      </div>

      {/* SAM status note */}
      {samStatus === false && (
        <div className="dze__info">
          SAM not loaded — place <code>sam_vit_b.pth</code> in <code>backend/weights/</code>
        </div>
      )}

      {/* Error */}
      {error && <div className="dze__error">{error}</div>}

      {/* Create form */}
      {formOpen && (
        <div className="dze__form">
          <input
            className="dze__input"
            type="text"
            placeholder="Zone name (e.g. North Scaffold)"
            value={formName}
            onChange={e => setFormName(e.target.value)}
          />
          <div className="dze__form-row">
            <select className="dze__select" value={formRisk} onChange={e => setFormRisk(e.target.value as ZoneRiskLevel)}>
              <option value="low">Low risk</option>
              <option value="moderate">Moderate risk</option>
              <option value="high">High risk</option>
              <option value="critical">Critical risk</option>
              <option value="safe">Safe zone</option>
            </select>
            <select className="dze__select" value={formType} onChange={e => setFormType(e.target.value as ZoneType)}>
              <option value="restricted">Restricted</option>
              <option value="work">Work zone</option>
              <option value="hazard">Hazard</option>
              <option value="safe">Safe zone</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          {Object.keys(templates).length > 0 && (
            <select className="dze__select" value={formTemplate} onChange={e => setFormTemplate(e.target.value)}>
              <option value="">— Select polygon template —</option>
              {Object.keys(templates).map(k => (
                <option key={k} value={k}>{k.replace(/_/g, ' ').toUpperCase()}</option>
              ))}
            </select>
          )}
          <button className="dze__btn dze__btn--save" onClick={createZone} disabled={formSaving}>
            {formSaving ? 'SAVING…' : 'CREATE ZONE'}
          </button>
        </div>
      )}

      {/* Zone list */}
      <div className="dze__list">
        {loading && (
          <div className="dze__empty">
            <div className="dze__spinner" />
          </div>
        )}
        {!loading && zones.length === 0 && !error && (
          <div className="dze__empty">No dynamic zones yet. Add one above or run SAM detect.</div>
        )}
        {zones.map(zone => (
          <div key={zone.id} className={`dze__zone ${!zone.is_active ? 'dze__zone--inactive' : ''}`}>
            <div className="dze__zone-color" style={{ background: zone.color }} />
            <div className="dze__zone-info">
              <span className="dze__zone-name">{zone.name}</span>
              <span className="dze__zone-meta">
                <span className="dze__zone-risk" style={{ color: RISK_COLORS[zone.risk_level] ?? '#aaa' }}>
                  {zone.risk_level.toUpperCase()}
                </span>
                &nbsp;·&nbsp;
                {ZONE_TYPE_LABELS[zone.zone_type] ?? zone.zone_type}
              </span>
              {zone.description && (
                <span className="dze__zone-desc">{zone.description}</span>
              )}
            </div>
            <div className="dze__zone-actions">
              <button
                className={`dze__icon-btn ${zone.is_active ? 'dze__icon-btn--on' : 'dze__icon-btn--off'}`}
                onClick={() => toggleActive(zone)}
                title={zone.is_active ? 'Deactivate' : 'Activate'}
              >
                {zone.is_active ? '●' : '○'}
              </button>
              <button
                className="dze__icon-btn dze__icon-btn--del"
                onClick={() => deleteZone(zone)}
                title="Delete zone"
              >
                ✕
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
