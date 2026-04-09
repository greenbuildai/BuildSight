import { useState, useEffect } from 'react'

interface GeoAIPlan2DProps {
  showLabels?: boolean
}

const VIEWER_SRC = '/buildsight_2d_plan.html'
const LOAD_TIMEOUT_MS = 10000

export function GeoAIPlan2D({ showLabels }: GeoAIPlan2DProps) {
  const [frameKey, setFrameKey] = useState(0)
  const [isLoaded, setIsLoaded] = useState(false)
  const [hasError, setHasError] = useState(false)
  const [isTimeout, setIsTimeout] = useState(false)

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
    setIsLoaded(false)
    setHasError(false)
    setFrameKey(k => k + 1)
  }

  return (
    <div className="geoai-3d">
      <div className="geoai-3d__topbar">
        <div>
          <span className="section-label">2D Spatial Intelligence</span>
          <h3>BuildSight Tactical Plan</h3>
        </div>
        <div className="geoai-3d__actions">
          <button type="button" className="geoai-3d__button" onClick={handleReload}>
            Retry Load
          </button>
          <button
            type="button"
            className="geoai-3d__button geoai-3d__button--ghost"
            onClick={() => window.open(VIEWER_SRC, '_blank', 'noopener,noreferrer')}
          >
            Open in new tab
          </button>
        </div>
      </div>

      <div className="geoai-3d__canvas">
        {!isLoaded && !hasError && (
          <div className="geoai-3d__state">
            <div className="geoai-map-loading__spinner" />
            <span>Rendering 2D Tactical Plan...</span>
          </div>
        )}

        {hasError && (
          <div className="geoai-3d__state geoai-3d__state--error geoai-toast">
            <strong>{isTimeout ? 'Plan render timeout' : 'Plan render failed'}</strong>
            <span>The 2D visualization could not be loaded within {LOAD_TIMEOUT_MS / 1000}s.</span>
            <button type="button" className="geoai-3d__button" onClick={handleReload}>Retry</button>
          </div>
        )}

        <iframe
          key={frameKey}
          title="BuildSight 2D Plan"
          src={VIEWER_SRC}
          className={`geoai-3d__frame ${isLoaded ? 'geoai-3d__frame--loaded' : ''}`}
          onLoad={() => {
            setIsLoaded(true)
            setHasError(false)
            setIsTimeout(false)
          }}
          onError={() => {
            setHasError(true)
            setIsLoaded(false)
          }}
        />
      </div>
    </div>
  )
}
