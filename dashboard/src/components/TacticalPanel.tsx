import { motion } from 'framer-motion'

interface TacticalPanelProps {
  title: string
  children: React.ReactNode
  className?: string
  priority?: 'low' | 'moderate' | 'high' | 'critical'
}

export function TacticalPanel({ title, children, className = '', priority = 'moderate' }: TacticalPanelProps) {
  const priorityColor = {
    low: '#00e676',
    moderate: '#ffd600',
    high: '#ff7b00',
    critical: '#ff3b3b'
  }[priority]

  return (
    <motion.div 
      className={`tactical-panel tactical-corner ${className}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* ── Header ────────────────────────────────────────────────── */}
      <div className="tactical-panel__header" style={{ borderBottomColor: `${priorityColor}44` }}>
        <div className="tactical-panel__title-wrap">
          <span className="geoai-status-pulse" style={{ background: priorityColor, boxShadow: `0 0 8px ${priorityColor}` }} />
          <span className="tactical-panel__title">{title}</span>
        </div>
        <div className="tactical-panel__ref">MX_NODE: PRO_01</div>
      </div>

      {/* ── Content ───────────────────────────────────────────────── */}
      <div className="tactical-panel__content">
        {children}
      </div>

      {/* ── Brackets Overlay ───────────────────────────────────────── */}
      <div className="tactical-brackets" style={{ borderColor: `${priorityColor}88` }} />
      
      {/* ── Scanlines Layer ────────────────────────────────────────── */}
      <div className="hud-scanline" style={{ opacity: 0.1 }} />
    </motion.div>
  )
}
