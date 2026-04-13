import { useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { HeatmapUpdatePayload, GeoAIVisualMode } from '../types/geoai'

interface GeoAIHUDProps {
  data: HeatmapUpdatePayload | null
  connectionState: 'connecting' | 'live' | 'demo' | 'disconnected'
  cycle: number
  activeMode: GeoAIVisualMode
  onAcknowledge?: (eventId: string) => void
  onResolve?: (eventId: string) => void
  statusData?: {
    modeMeta: { label: string; detail: string; badge: string }
    topWorker: any
    topEvent: any
    mapperStatus: string
  }
}

type TabType = 'STATUS' | 'CRITICAL' | 'WARNING' | 'INFO'

function formatTime(isoString: string) {
  const d = new Date(isoString)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function GeoAIHUD({ data, connectionState, cycle, activeMode, onAcknowledge, onResolve, statusData }: GeoAIHUDProps) {
  const [activeTab, setActiveTab] = useState<TabType>('STATUS')

  const isStale = connectionState === 'disconnected'

  const events = data?.events ?? []
  
  const filteredEvents = useMemo(() => {
    return events.filter(e => e.priority === activeTab).sort((a, b) => {
      return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    })
  }, [events, activeTab])

  const criticalCount = events.filter(e => e.priority === 'CRITICAL').length
  const warningCount = events.filter(e => e.priority === 'WARNING').length
  const infoCount = events.filter(e => e.priority === 'INFO').length

  const sysStatusClass =
    connectionState === 'live' ? 'geoai-hud-status--live' :
    connectionState === 'demo' ? 'geoai-hud-status--demo' :
    connectionState === 'connecting' ? 'geoai-hud-status--connecting' :
    'geoai-hud-status--disconnected'

  return (
    <div className="geoai-hud">
      <div className="geoai-hud__header">
        <div>
          <span className="geoai-hud__section-label">Incident Telemetry</span>
          <h2 className="geoai-hud__title">Escalation Queue</h2>
        </div>
        <div className={`geoai-hud-status ${sysStatusClass}`}>
          <span className="geoai-hud-status-indicator" />
          {connectionState.toUpperCase()}
        </div>
      </div>

      <div className="geoai-hud__tabs">
        <button 
          type="button" 
          className={`geoai-hud-tab ${activeTab === 'STATUS' ? 'geoai-hud-tab--active' : ''}`}
          onClick={() => setActiveTab('STATUS')}
        >
          SITE STATUS
        </button>
        <button 
          type="button" 
          className={`geoai-hud-tab ${activeTab === 'CRITICAL' ? 'geoai-hud-tab--active' : ''}`}
          onClick={() => setActiveTab('CRITICAL')}
        >
          CRITICAL <span className="geoai-hud-badge geoai-hud-badge--critical">{criticalCount}</span>
        </button>
        <button 
          type="button" 
          className={`geoai-hud-tab ${activeTab === 'WARNING' ? 'geoai-hud-tab--active' : ''}`}
          onClick={() => setActiveTab('WARNING')}
        >
          WARNING <span className="geoai-hud-badge geoai-hud-badge--warning">{warningCount}</span>
        </button>
        <button 
          type="button" 
          className={`geoai-hud-tab ${activeTab === 'INFO' ? 'geoai-hud-tab--active' : ''}`}
          onClick={() => setActiveTab('INFO')}
        >
          INFO <span className="geoai-hud-badge geoai-hud-badge--info">{infoCount}</span>
        </button>
      </div>

      <div className="geoai-hud__content">
        <AnimatePresence mode="wait">
          {activeTab === 'STATUS' ? (
            <motion.div
              key="status"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="geoai-hud-status-stack"
            >
              <article className="geoai-summary-card">
                <span className="geoai-summary-card__label">Mode focus</span>
                <strong>{statusData?.modeMeta.label}</strong>
                <p>{statusData?.modeMeta.detail}</p>
                <span className="geoai-summary-card__foot">{statusData?.modeMeta.badge}</span>
              </article>

              <article className="geoai-summary-card">
                <span className="geoai-summary-card__label">Operational hotspot</span>
                <strong>{statusData?.topWorker ? `Worker W${statusData.topWorker.worker_id ?? '?'}` : 'No active worker'}</strong>
                <p>
                  {statusData?.topWorker
                    ? `${statusData.topWorker.risk} risk in ${statusData.topWorker.zone.replace(/_/g, ' ')} at ${statusData.topWorker.height_m.toFixed(1)}m.`
                    : 'Telemetry feed stable.'}
                </p>
                <span className="geoai-summary-card__foot">{statusData?.topWorker?.status ?? 'Monitoring'}</span>
              </article>

              <article className="geoai-summary-card">
                <span className="geoai-summary-card__label">Escalation posture</span>
                <strong>{statusData?.topEvent ? statusData.topEvent.event_type.replace(/_/g, ' ') : 'No live escalation'}</strong>
                <p>{statusData?.topEvent?.message ?? 'Critical events surface here automatically.'}</p>
                <span className="geoai-summary-card__foot">{statusData?.mapperStatus}</span>
              </article>
            </motion.div>
          ) : (
            <motion.div
              key="events"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="geoai-event-log"
            >
              {filteredEvents.length === 0 ? (
                <div className="geoai-hud-empty">
                  <div className="geoai-hud-empty__icon">✓</div>
                  <span>No escalation records found for {activeTab}.</span>
                </div>
              ) : (
                filteredEvents.map(event => (
                  <motion.div
                    key={event.event_id}
                    layout
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    className={`geoai-event-card geoai-event-card--${event.priority.toLowerCase()} geoai-event-card--${event.state.toLowerCase()}`}
                  >
                    <div className="geoai-event-card__header">
                      <span className="geoai-event-card__type">
                        {event.event_type.replace(/_/g, ' ')}
                      </span>
                      <span className="geoai-event-card__time">
                        {formatTime(event.timestamp)}
                      </span>
                    </div>
                    
                    <div className="geoai-event-card__body">
                      {event.message}
                    </div>
                    
                    <div className="geoai-event-card__meta">
                      <span className="geoai-event-card__zone">📍 {event.zone.replace(/_/g, ' ')}</span>
                      <span className="geoai-event-card__cam">📷 {event.camera_id}</span>
                    </div>

                    {event.state === 'ACTIVE' && (
                      <div className="geoai-event-card__actions">
                        <button
                          type="button"
                          className="geoai-event-btn geoai-event-btn--ack"
                          onClick={() => onAcknowledge?.(event.event_id)}
                        >
                          Acknowledge
                        </button>
                        <button
                          type="button"
                          className="geoai-event-btn geoai-event-btn--resolve"
                          onClick={() => onResolve?.(event.event_id)}
                        >
                          Resolve
                        </button>
                      </div>
                    )}

                    {event.state === 'ACKNOWLEDGED' && (
                      <div className="geoai-event-card__actions">
                        <button
                          type="button"
                          className="geoai-event-btn geoai-event-btn--resolve"
                          onClick={() => onResolve?.(event.event_id)}
                        >
                          Resolve Event
                        </button>
                      </div>
                    )}
                  </motion.div>
                ))
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="geoai-hud__footer">
        <div className="geoai-hud-metric">
          <span className="geoai-hud-metric-label">Update Cycle</span>
          <span className="geoai-hud-metric-val">{cycle}</span>
        </div>
        <div className="geoai-hud-metric">
          <span className="geoai-hud-metric-label">Renderer</span>
          <span className="geoai-hud-metric-val">{activeMode.toUpperCase()}</span>
        </div>
        <div className="geoai-hud-metric">
          <span className="geoai-hud-metric-label">Data age</span>
          <span className={`geoai-hud-metric-val ${isStale ? 'geoai-hud-metric-val--alert' : ''}`}>
            {isStale ? 'STALE' : 'LIVE'}
          </span>
        </div>
      </div>
    </div>
  )
}
