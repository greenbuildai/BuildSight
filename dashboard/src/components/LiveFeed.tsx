import { useEffect, useMemo, useRef, useState } from 'react'
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
  const visible = useMemo(() => 
    detections.filter((d) => d.conf >= confidenceThreshold),
    [confidenceThreshold]
  )

  const [showHeatmap, setShowHeatmap] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const heatmapHistory = useRef<{ x: number, y: number, time: number, tone: string }[]>([])

  // Initialization
  useEffect(() => {
    setRunning(true)
    setModelName('BS-ENSEMBLE-WBF')
    return () => {
      setRunning(false)
    }
  }, [setRunning, setModelName])

  // Polling and Drawing
  useEffect(() => {
    const interval = setInterval(() => {
      const mapped = visible.map(d => ({
        class: d.label.includes('HARDHAT') ? 'helmet' : d.label.includes('VEST') ? 'vest' : 'worker',
        confidence: d.conf
      }))
      pushDetections(mapped, 12)

      // Add to heatmap history
      const now = performance.now()
      visible.forEach(d => {
        const x = parseFloat(d.left) / 100
        const y = parseFloat(d.top) / 100
        heatmapHistory.current.push({ x, y, time: now, tone: d.tone })
      })
    }, 1000)

    let raf: number
    const draw = () => {
      const canvas = canvasRef.current
      if (canvas) {
        const ctx = canvas.getContext('2d')
        if (ctx) {
          const w = canvas.clientWidth
          const h = canvas.clientHeight
          if (canvas.width !== w) canvas.width = w
          if (canvas.height !== h) canvas.height = h
          
          ctx.clearRect(0, 0, w, h)
          
          if (showHeatmap) {
            const now = performance.now()
            heatmapHistory.current = heatmapHistory.current.filter(p => now - p.time < 1500)
            
            heatmapHistory.current.forEach(p => {
              const hx = p.x * w
              const hy = p.y * h
              const age = (now - p.time) / 1500
              const opacity = Math.max(0, (1 - age) * 0.05)
              
              const grad = ctx.createRadialGradient(hx, hy, 0, hx, hy, 20)
              const color = p.tone === 'critical' ? '255, 42, 42' : '0, 255, 136'
              grad.addColorStop(0, `rgba(${color}, ${opacity})`)
              grad.addColorStop(1, `rgba(${color}, 0)`)
              
              ctx.fillStyle = grad
              ctx.beginPath()
              ctx.arc(hx, hy, 20, 0, Math.PI * 2)
              ctx.fill()
            })
          }
        }
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)

    return () => {
      clearInterval(interval)
      cancelAnimationFrame(raf)
    }
  }, [pushDetections, visible, showHeatmap])

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
        
        {/* Heatmap Overlay Layer */}
        <canvas 
          ref={canvasRef} 
          className="live-feed__heatmap" 
          style={{ 
            position: 'absolute', 
            top: 0, 
            left: 0, 
            width: '100%', 
            height: '100%', 
            pointerEvents: 'none',
            zIndex: 5
          }} 
        />
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
        <button 
          className={`live-feed__toggle ${showHeatmap ? 'active' : ''}`}
          onClick={() => setShowHeatmap(!showHeatmap)}
          style={{
            background: 'none',
            border: '1px solid var(--color-border)',
            padding: '2px 8px',
            fontSize: '10px',
            color: showHeatmap ? 'var(--color-accent)' : 'var(--color-text-muted)',
            cursor: 'pointer',
            marginLeft: 'auto'
          }}
        >
          {showHeatmap ? 'HEATMAP ON' : 'HEATMAP OFF'}
        </button>
      </div>
    </div>
  )
}
