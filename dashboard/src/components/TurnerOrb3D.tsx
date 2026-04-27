import React, { useEffect, useRef, useMemo } from 'react'

interface TurnerOrb3DProps {
  amplitude: number
  state: 'idle' | 'presenting' | 'thinking' | 'speaking' | 'listening'
  size?: number
}

// ── Types for Neural Geometry ──
interface NeuralPoint {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  size: number;
  brightness: number;
  connections: number[]; // Indices of connected points
}

interface SynapticPath {
  points: { x: number; y: number; z: number }[];
  speed: number;
  offset: number;
  width: number;
  opacity: number;
}

const TurnerOrb3D: React.FC<TurnerOrb3DProps> = ({ 
  amplitude, 
  state, 
  size = 400 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const requestRef = useRef<number>(0)
  const rotationRef = useRef({ x: 0, y: 0, z: 0 })
  
  // ── Theme Mapping (BuildSight Professional) ──
  const theme = useMemo(() => {
    const colors = {
      idle: { primary: '#00ffff', secondary: '#0088ff', core: '#ffffff', glow: 'rgba(0, 255, 255, 0.08)' },
      speaking: { primary: '#00ffff', secondary: '#00f2ff', core: '#ffffff', glow: 'rgba(0, 255, 255, 0.25)' },
      listening: { primary: '#4d79ff', secondary: '#2962ff', core: '#ffffff', glow: 'rgba(77, 121, 255, 0.2)' },
      thinking: { primary: '#ffcc00', secondary: '#ff8800', core: '#ffffff', glow: 'rgba(255, 204, 0, 0.15)' },
      presenting: { primary: '#00ffff', secondary: '#00ffcc', core: '#ffffff', glow: 'rgba(0, 255, 255, 0.15)' }
    }
    return colors[state] || colors.idle
  }, [state])

  // ── Generate Neural Geometry once ──
  const neuralData = useMemo(() => {
    const points: NeuralPoint[] = []
    const paths: SynapticPath[] = []
    const sphereRadius = size * 0.38
    const nodeCount = 120 // Increased density

    // 1. Generate nodes on sphere surface
    for (let i = 0; i < nodeCount; i++) {
      const phi = Math.acos(-1 + (2 * i) / nodeCount)
      const theta = Math.sqrt(nodeCount * Math.PI) * phi
      
      const p = {
        x: sphereRadius * Math.cos(theta) * Math.sin(phi),
        y: sphereRadius * Math.sin(theta) * Math.sin(phi),
        z: sphereRadius * Math.cos(phi),
        vx: (Math.random() - 0.5) * 0.01,
        vy: (Math.random() - 0.5) * 0.01,
        vz: (Math.random() - 0.5) * 0.01,
        size: Math.random() * 1.5 + 0.5,
        brightness: Math.random(),
        connections: [] as number[]
      }
      points.push(p)
    }

    // Assign connections to nearby points
    points.forEach((p, i) => {
      for (let j = i + 1; j < points.length; j++) {
        const other = points[j]
        const dist = Math.sqrt((p.x-other.x)**2 + (p.y-other.y)**2 + (p.z-other.z)**2)
        if (dist < sphereRadius * 0.4 && p.connections.length < 3) {
          p.connections.push(j)
        }
      }
    })

    // 2. Generate synaptic paths (arcs)
    for (let i = 0; i < 18; i++) {
      const pathPoints = []
      const segments = 24
      const startPhi = Math.random() * Math.PI * 2
      const startTheta = Math.random() * Math.PI
      const length = Math.random() * Math.PI * 1.2 + 0.4
      
      for (let j = 0; j <= segments; j++) {
        const t = j / segments
        const phi = startPhi + t * length
        const theta = startTheta + Math.sin(t * Math.PI) * 0.3
        pathPoints.push({
          x: sphereRadius * Math.cos(phi) * Math.sin(theta),
          y: sphereRadius * Math.sin(phi) * Math.sin(theta),
          z: sphereRadius * Math.cos(theta)
        })
      }
      paths.push({
        points: pathPoints,
        speed: Math.random() * 0.03 + 0.01,
        offset: Math.random() * Math.PI * 2,
        width: Math.random() * 1.2 + 0.3,
        opacity: Math.random() * 0.4 + 0.15
      })
    }

    return { points, paths, sphereRadius }
  }, [size])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    ctx.scale(dpr, dpr)

    const draw = (time: number) => {
      ctx.clearRect(0, 0, size, size)
      const centerX = size / 2
      const centerY = size / 2
      
      // Rotation logic
      const baseRotation = state === 'idle' ? 0.0015 : (state === 'thinking' ? 0.02 : 0.006)
      rotationRef.current.y += baseRotation + (amplitude * 0.04)
      rotationRef.current.x = Math.sin(time * 0.0004) * 0.15
      const { x: rotX, y: rotY } = rotationRef.current

      // Projection Helper
      const project = (p: { x: number; y: number; z: number }) => {
        let x1 = p.x * Math.cos(rotY) - p.z * Math.sin(rotY)
        let z1 = p.x * Math.sin(rotY) + p.z * Math.cos(rotY)
        let y2 = p.y * Math.cos(rotX) - z1 * Math.sin(rotX)
        let z2 = p.y * Math.sin(rotX) + z1 * Math.cos(rotX)
        
        const perspective = 1 + (z2 / (size * 0.6)) * 0.5
        return {
          sx: x1 * perspective + centerX,
          sy: y2 * perspective + centerY,
          sz: z2,
          alpha: (z2 + neuralData.sphereRadius) / (neuralData.sphereRadius * 2.5)
        }
      }

      // 1. Global Glow
      const bgGlow = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, size * 0.5)
      bgGlow.addColorStop(0, theme.glow)
      bgGlow.addColorStop(1, 'rgba(0,0,0,0)')
      ctx.fillStyle = bgGlow
      ctx.fillRect(0, 0, size, size)

      // 2. Synaptic Pathways (Background)
      neuralData.paths.forEach(path => {
        ctx.beginPath()
        ctx.lineWidth = path.width + (amplitude * 4)
        path.points.forEach((p, idx) => {
          const { sx, sy, alpha } = project(p)
          ctx.globalAlpha = path.opacity * alpha * (0.4 + amplitude * 0.6)
          ctx.strokeStyle = theme.primary
          if (idx === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        })
        ctx.stroke()

        // Data pulses
        const pulseT = (time * path.speed + path.offset) % 1
        const pulseIdx = Math.floor(pulseT * (path.points.length - 1))
        const { sx: psx, sy: psy, alpha: palpha } = project(path.points[pulseIdx])
        if (palpha > 0.3) {
          ctx.beginPath()
          ctx.arc(psx, psy, 1.5 + (amplitude * 5), 0, Math.PI * 2)
          ctx.fillStyle = '#fff'
          ctx.globalAlpha = 0.8 * palpha
          ctx.fill()
        }
      })

      // 3. Neural Mesh (Connections between nodes)
      ctx.lineWidth = 0.5
      neuralData.points.forEach(p => {
        const proj1 = project(p)
        if (proj1.sz < -20) return // Culling back-facing connections
        
        p.connections.forEach(targetIdx => {
          const target = neuralData.points[targetIdx]
          const proj2 = project(target)
          
          const opacity = (Math.sin(time * 0.002 + p.brightness * 10) + 1) * 0.5
          ctx.globalAlpha = proj1.alpha * opacity * 0.2 * (0.5 + amplitude * 1.5)
          ctx.strokeStyle = theme.primary
          ctx.beginPath()
          ctx.moveTo(proj1.sx, proj1.sy)
          ctx.lineTo(proj2.sx, proj2.sy)
          ctx.stroke()
        })
      })

      // 4. Neural Nodes
      neuralData.points.forEach(p => {
        const { sx, sy, alpha } = project(p)
        if (alpha < 0.2) return

        const flicker = Math.sin(time * 0.01 + p.brightness * 100) > 0.8 ? 1 : 0.4
        const finalSize = p.size * (1 + amplitude * 3) * flicker
        
        ctx.globalAlpha = alpha * (0.5 + amplitude * 0.5)
        ctx.fillStyle = theme.primary
        ctx.beginPath()
        ctx.arc(sx, sy, finalSize, 0, Math.PI * 2)
        ctx.fill()

        if (alpha > 0.8 && amplitude > 0.3 && Math.random() > 0.98) {
          ctx.beginPath()
          ctx.arc(sx, sy, finalSize * 3, 0, Math.PI * 2)
          ctx.strokeStyle = '#fff'
          ctx.globalAlpha = 0.3
          ctx.stroke()
        }
      })

      // 5. Central Brain Core
      const corePulse = 1 + Math.sin(time * 0.01) * 0.05 + (amplitude * 1.2)
      const coreRadius = size * 0.07 * corePulse
      
      const coreGrad = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, coreRadius)
      coreGrad.addColorStop(0, '#fff')
      coreGrad.addColorStop(0.3, theme.primary)
      coreGrad.addColorStop(0.7, theme.secondary)
      coreGrad.addColorStop(1, 'rgba(0,0,0,0)')
      
      ctx.globalAlpha = 1
      ctx.fillStyle = coreGrad
      ctx.beginPath()
      ctx.arc(centerX, centerY, coreRadius, 0, Math.PI * 2)
      ctx.fill()

      // Core Structural Rings
      ctx.lineWidth = 1
      for (let i = 0; i < 4; i++) {
        const r = coreRadius * (0.2 + i * 0.25)
        const rot = time * 0.002 * (i % 2 === 0 ? 1 : -1)
        ctx.setLineDash([2, 4])
        ctx.strokeStyle = theme.primary
        ctx.globalAlpha = 0.3
        ctx.beginPath()
        ctx.ellipse(centerX, centerY, r, r * 0.6, rot, 0, Math.PI * 2)
        ctx.stroke()
      }
      ctx.setLineDash([])

      requestRef.current = requestAnimationFrame(draw)
    }

    requestRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(requestRef.current)
  }, [amplitude, state, size, neuralData, theme])

  return (
    <canvas 
      ref={canvasRef} 
      style={{ 
        width: size, 
        height: size, 
        filter: 'drop-shadow(0 0 50px rgba(0, 255, 255, 0.15))' 
      }} 
    />
  )
}

export default TurnerOrb3D


