/**
 * DetectionStatusBar
 * Persistent global status bar shown on every tab while the background
 * detection service is running. Reads from the Zustand detectionStore so
 * it reflects live state regardless of which React view is active.
 */

import { memo } from 'react'
import { useDetectionStore } from '../store/detectionStore'
import './DetectionStatusBar.css'

const DetectionStatusBar = memo(() => {
  const isRunning       = useDetectionStore(s => s.isRunning)
  const isPaused        = useDetectionStore(s => s.isPaused)
  const isConnected     = useDetectionStore(s => s.isConnected)
  const fps             = useDetectionStore(s => s.fps)
  const latencyMs       = useDetectionStore(s => s.latencyMs)
  const workerCount     = useDetectionStore(s => s.workerCount)
  const sceneCondition  = useDetectionStore(s => s.sceneCondition)
  const progressPercent = useDetectionStore(s => s.progressPercent)
  const violations      = useDetectionStore(s => s.violations)

  // Only render while detection is active
  if (!isRunning && !isPaused) return null

  const critical = violations.filter(v => v.severity === 'CRITICAL')
  const hasCritical = critical.length > 0

  return (
    <div className={`dsb ${hasCritical ? 'dsb--critical' : 'dsb--normal'}`}>
      {/* Status pill */}
      <div className="dsb-status">
        <span className={`dsb-dot ${isPaused ? 'dsb-dot--paused' : 'dsb-dot--running'}`} />
        <span className="dsb-label">{isPaused ? 'PAUSED' : 'DETECTING'}</span>
      </div>

      {/* Progress bar */}
      <div className="dsb-progress-track">
        <div className="dsb-progress-fill" style={{ width: `${progressPercent}%` }} />
      </div>
      <span className="dsb-pct">{progressPercent}%</span>

      {/* Metrics — dimmed and frozen while paused */}
      <div className={`dsb-metrics ${isPaused ? 'dsb-metrics--frozen' : ''}`}>
        <span className="dsb-metric">
          <span className="dsb-metric-label">FPS</span>
          <span className="dsb-metric-value">{isPaused ? '—' : fps}</span>
        </span>
        <span className="dsb-metric">
          <span className="dsb-metric-label">LAT</span>
          <span className="dsb-metric-value">{isPaused ? '—' : `${Math.round(latencyMs)}ms`}</span>
        </span>
        <span className="dsb-metric">
          <span className="dsb-metric-label">WORKERS</span>
          <span className="dsb-metric-value">{workerCount}</span>
        </span>
        <span className={`dsb-metric ${sceneCondition.includes('crowded') ? 'dsb-metric--crowded' : ''}`}>
          <span className="dsb-metric-label">SCENE</span>
          <span className="dsb-metric-value">{sceneCondition.replace('_', ' ')}</span>
        </span>
      </div>

      {/* Critical violation ticker */}
      {hasCritical && (
        <div className="dsb-violation-ticker">
          <span className="dsb-violation-icon">⚠</span>
          <span className="dsb-violation-text">
            {critical.length} CRITICAL — {critical[0].zone_name}: {critical[0].details}
          </span>
        </div>
      )}

      {/* Connection indicator */}
      <div className={`dsb-conn ${isConnected ? 'dsb-conn--live' : 'dsb-conn--lost'}`}>
        {isConnected ? '● LIVE' : '○ RECONNECTING'}
      </div>
    </div>
  )
})

DetectionStatusBar.displayName = 'DetectionStatusBar'
export default DetectionStatusBar
