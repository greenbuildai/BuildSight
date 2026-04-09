import { useMemo, useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import type { HeatmapUpdatePayload } from '../types/geoai'

interface GeoAI3DViewProps {
  data: HeatmapUpdatePayload | null
  showLabels: boolean
  showCameraFOV: boolean
  onReturn2D?: () => void
}

const VIEWER_SRC = '/buildsight_3d_v5.html'
const LOAD_TIMEOUT_MS = 15000 // 3D models can be heavy

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`
}

export function GeoAI3DView({ data, showLabels, showCameraFOV, onReturn2D }: GeoAI3DViewProps) {
  const [frameKey, setFrameKey] = useState(0)
  const [isLoaded, setIsLoaded] = useState(false)
  const [hasError, setHasError] = useState(false)
  const [isTimeout, setIsTimeout] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const criticalCells = useMemo(
    () => data?.cells?.filter(cell => cell.risk_level === 'CRITICAL').length ?? 0,
    [data]
  )

  const maxRisk = useMemo(
    () => Math.max(...(data?.cells?.map(cell => cell.risk_score) ?? [0])),
    [data]
  )

  const zoneCount = data?.site_stats?.critical_zones ?? 6
  const cameraCount = useMemo(() => {
    const cameraIds = new Set(
      (data?.workers ?? [])
        .map(worker => worker.camera_id)
        .filter((cameraId): cameraId is string => Boolean(cameraId))
    )
    return cameraIds.size || 3
  }, [data])
  const workerCount = data?.site_stats?.total_workers ?? data?.workers?.length ?? 0

  useEffect(() => {
    setIsTimeout(false)
    const timer = setTimeout(() => {
      if (!isLoaded) {
        setIsTimeout(true)
        setHasError(true)
      }
    }, LOAD_TIMEOUT_MS)
    return () => clearTimeout(timer)
  }, [frameKey, isLoaded])

  const handleReload = () => {
    setHasError(false)
    setIsLoaded(false)
    setIsTimeout(false)
    setFrameKey(prev => prev + 1)
  }

  /* ── Fullscreen toggle ─────────────────────────────────────────────── */
  const toggleFullscreen = () => {
    if (!containerRef.current) return
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {})
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {})
    }
  }

  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', handler)
    return () => document.removeEventListener('fullscreenchange', handler)
  }, [])

  return (
    <div className={`geoai-3d ${isFullscreen ? 'geoai-3d--fullscreen' : ''}`} ref={containerRef}>
      <div className="geoai-3d__topbar">
        <div>
          <span className="section-label">3D spatial intelligence</span>
          <h3>BuildSight Site Volume</h3>
        </div>
        <div className="geoai-3d__actions">
          <span className="geoai-3d__chip">{showLabels ? 'Labels on' : 'Labels off'}</span>
          <span className="geoai-3d__chip">{showCameraFOV ? 'Camera FOV on' : 'Camera FOV off'}</span>
          
          <button type="button" className="geoai-3d__button" onClick={handleReload}>
            Reload 3D
          </button>
          <button type="button" className="geoai-3d__button" onClick={toggleFullscreen}>
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </button>
          <button
            type="button"
            className="geoai-3d__button geoai-3d__button--ghost"
            onClick={() => window.open(VIEWER_SRC, '_blank', 'noopener,noreferrer')}
          >
            Open standalone
          </button>
          {onReturn2D && (
            <button type="button" className="geoai-3d__button geoai-3d__button--ghost" onClick={onReturn2D}>
              Return to 2D
            </button>
          )}
        </div>
      </div>

      <div className="geoai-3d__canvas">
        {!isLoaded && !hasError && (
          <div className="geoai-3d__state">
            <div className="geoai-map-loading__spinner geoai-map-loading__spinner--3d" />
            <span>Loading 3D volumetric model...</span>
            <div className="geoai-3d__warning">
              This model requires hardware acceleration and may consume significant memory.
            </div>
          </div>
        )}

        {hasError && (
          <div className="geoai-3d__state geoai-3d__state--error geoai-toast">
            <strong>{isTimeout ? '3D Render Timeout' : '3D model failed to load'}</strong>
            <span>The Plotly scene was not reachable or took too long to load ({LOAD_TIMEOUT_MS / 1000}s).</span>
            <button type="button" className="geoai-3d__button" onClick={handleReload}>
              Retry
            </button>
            {onReturn2D && (
              <button type="button" className="geoai-3d__button geoai-3d__button--ghost" onClick={onReturn2D}>
                Return to 2D
              </button>
            )}
          </div>
        )}

        <iframe
          key={frameKey}
          title="BuildSight 3D GeoAI model"
          src={VIEWER_SRC}
          className={`geoai-3d__frame ${isLoaded ? 'geoai-3d__frame--loaded' : ''}`}
          onLoad={() => {
            setIsLoaded(true)
            setHasError(false)
          }}
          onError={() => {
            setHasError(true)
            setIsLoaded(false)
          }}
        />

        <motion.div
          className="geoai-3d__overlay"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: isLoaded && !hasError ? 1 : 0, y: isLoaded && !hasError ? 0 : 16 }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
        >
          <div className="geoai-3d__stats">
            <div className="geoai-3d__stat">
              <span className="geoai-3d__stat-label">Workers</span>
              <strong>{workerCount}</strong>
            </div>
            <div className="geoai-3d__stat">
              <span className="geoai-3d__stat-label">Cameras</span>
              <strong>{cameraCount}</strong>
            </div>
            <div className="geoai-3d__stat">
              <span className="geoai-3d__stat-label">Active zones</span>
              <strong>{zoneCount}</strong>
            </div>
            <div className="geoai-3d__stat">
              <span className="geoai-3d__stat-label">Critical cells</span>
              <strong>{criticalCells}</strong>
            </div>
          </div>

          <div className="geoai-3d__legend">
            <div className="geoai-3d__legend-row">
              <span>Peak risk</span>
              <strong>{formatPercent(maxRisk)}</strong>
            </div>
            <div className="geoai-3d__legend-row">
              <span>Render mode</span>
              <strong>Plotly volumetric</strong>
            </div>
            <div className="geoai-3d__legend-row">
              <span>Viewport sync</span>
              <strong>{showCameraFOV ? 'Camera-aware' : 'Free orbit'}</strong>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
