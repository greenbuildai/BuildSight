import { useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { HeatmapUpdatePayload, SpatialEvent, GeoAIVisualMode } from '../types/geoai'

interface GeoAIHUDProps {
  data: HeatmapUpdatePayload | null
  connectionState: 'connecting' | 'live' | 'demo' | 'disconnected'
  cycle: number
  activeMode: GeoAIVisualMode
  onAcknowledge?: (eventId: string) => void
  onResolve?: (eventId: string) => void
}

type TabType = 'CRITICAL' | 'WARNING' | 'INFO'

function formatTime(isoString: string) {
  const d = new Date(isoString)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function GeoAIHUD({ data, connectionState, cycle, activeMode, onAcknowledge, onResolve }: GeoAIHUDProps) {
  const [activeTab, setActiveTab] = useState<TabType>('CRITICAL')

  const isDemo = connectionState === 'demo'
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
        <div className="geoai-event-log">
          <AnimatePresence mode="popLayout">
            {filteredEvents.length === 0 ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="geoai-hud-empty"
              >
                <div className="geoai-hud-empty__icon">✓</div>
                <span>No escalation records found for {activeTab}.</span>
              </motion.div>
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
          </AnimatePresence>
        </div>
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
