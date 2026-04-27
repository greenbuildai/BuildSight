import React, { Suspense, useEffect, useState } from 'react'
import Spline from '@splinetool/react-spline'
import './TurnerAvatar3D.css'

interface TurnerAvatar3DProps {
  isSpeaking?: boolean
  isThinking?: boolean
  size?: 'sm' | 'md' | 'lg'
}

const TurnerAvatar3D: React.FC<TurnerAvatar3DProps> = ({
  isSpeaking = false,
  isThinking = false,
  size = 'md'
}) => {
  const sceneUrl = '/assets/models/turner_avatar.splinecode'
  const [assetState, setAssetState] = useState<'checking' | 'ready' | 'missing'>('checking')

  useEffect(() => {
    let active = true

    fetch(sceneUrl, { method: 'HEAD' })
      .then((response) => {
        if (!active) return
        setAssetState(response.ok ? 'ready' : 'missing')
      })
      .catch(() => {
        if (active) setAssetState('missing')
      })

    return () => {
      active = false
    }
  }, [sceneUrl])

  return (
    <div className={`turner-avatar-3d turner-avatar-3d--${size} ${isSpeaking ? 'turner-avatar-3d--speaking' : ''} ${isThinking ? 'turner-avatar-3d--thinking' : ''}`}>
      <Suspense fallback={<FallbackIcon />}>
        <div className="turner-avatar-3d__container">
          {assetState === 'ready' ? (
            <SplineBoundary fallback={<FallbackIcon />}>
              <Spline
                scene={sceneUrl}
                className="turner-avatar-3d__canvas"
              />
            </SplineBoundary>
          ) : (
            <FallbackIcon />
          )}
        </div>
      </Suspense>
      <div className="turner-avatar-3d__overlay" />
    </div>
  )
}

class SplineBoundary extends React.Component<
  { children: React.ReactNode; fallback: React.ReactNode },
  { failed: boolean }
> {
  public constructor(props: { children: React.ReactNode; fallback: React.ReactNode }) {
    super(props)
    this.state = { failed: false }
  }

  public static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true }
  }

  public componentDidCatch(error: Error) {
    console.error('[TurnerAvatar3D] Spline render failure:', error)
  }

  public render() {
    if (this.state.failed) return this.props.fallback
    return this.props.children
  }
}

const FallbackIcon = () => (
  <div className="turner-avatar-3d__fallback">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="12" cy="12" r="9" strokeOpacity="0.2" />
      <circle cx="12" cy="12" r="5" strokeDasharray="2 2" />
      <path d="M12 7v2m0 6v2M7 12h2m6 0h2" strokeLinecap="round" />
    </svg>
  </div>
)

export default TurnerAvatar3D
