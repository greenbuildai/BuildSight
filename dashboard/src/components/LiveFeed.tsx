import { useEffect } from 'react'
import { useDetectionStats } from '../DetectionStatsContext'

const detections = [
  { label: 'NO HARDHAT', top: '20%', left: '18%', width: '24%', height: '28%', tone: 'critical', conf: 0.91 },
  { label: 'VEST LOW', top: '46%', left: '57%', width: '18%', height: '22%', tone: 'warning', conf: 0.76 },
  { label: 'RESTRICTED ARC', top: '14%', left: '62%', width: '26%', height: '44%', tone: 'neutral', conf: 0.68 },
  { label: 'WORKER / HELMET OK', top: '54%', left: '22%', width: '12%', height: '20%', tone: 'neutral', conf: 0.59 },
  { label: 'WORKER / VEST OK', top: '50%', left: '38%', width: '11%', height: '19%', tone: 'neutral', conf: 0.42 },
  { label: 'NO HARDHAT', top: '30%', left: '72%', width: '10%', height: '18%', tone: 'critical', conf: 0.31 },
]

interface LiveFeedProps {
  confidenceThreshold: number
}

export function LiveFeed({ confidenceThreshold }: LiveFeedProps) {
  const { pushDetections, setRunning, setModelName } = useDetectionStats()
  const visible = detections.filter((d) => d.conf >= confidenceThreshold)

  useEffect(() => {
    setRunning(true)
    setModelName('BS-ENSEMBLE-WBF')
    
    // Simulate a recurring "inference" from the live stream
    const interval = setInterval(() => {
      const mapped = visible.map(d => ({
        class: d.label.includes('HARDHAT') ? 'helmet' : d.label.includes('VEST') ? 'vest' : 'worker',
        confidence: d.conf
      }))
      pushDetections(mapped, 12) // low latency for "live"
    }, 1000)

    return () => {
      clearInterval(interval)
      setRunning(false)
    }
  }, [visible, pushDetections, setRunning, setModelName])

  return (
    <div className="live-feed">
      <div className="live-feed__hud">
        <span>REC / 04 FEEDS</span>
        <span>MODEL: BS-ENSEMBLE-WBF</span>
        <span>CONF GATE: {(confidenceThreshold * 100).toFixed(0)}%</span>
      </div>

      <div className="live-feed__viewport" role="img" aria-label="Live construction site feed with AI overlays">
        <div className="feed-structure feed-structure--beam" />
        <div className="feed-structure feed-structure--mast" />
        <div className="feed-structure feed-structure--deck" />

        {visible.map((item) => (
          <div
            key={item.label + item.top}
            className={`detection detection--${item.tone}`}
            style={{
              top: item.top,
              left: item.left,
              width: item.width,
              height: item.height,
            }}
          >
            <span>{item.label} {(item.conf * 100).toFixed(0)}%</span>
          </div>
        ))}

        <div className="crosshair crosshair--horizontal" aria-hidden="true" />
        <div className="crosshair crosshair--vertical" aria-hidden="true" />
        <div className="feed-scanline" aria-hidden="true" />
      </div>

      <div className="live-feed__legend">
        <div>
          <span className="legend-swatch legend-swatch--critical" aria-hidden="true" />
          Critical
        </div>
        <div>
          <span className="legend-swatch legend-swatch--warning" aria-hidden="true" />
          Warning
        </div>
        <div>
          <span className="legend-swatch legend-swatch--neutral" aria-hidden="true" />
          Zone Tracking
        </div>
      </div>
    </div>
  )
}
