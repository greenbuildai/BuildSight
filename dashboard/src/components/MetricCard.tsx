import { motion } from 'framer-motion'

type MetricCardProps = {
  label: string
  value: string
  delta: string
  status: 'stable' | 'risk' | 'alert'
  progress: number
  footnote: string
  index?: number
}

export function MetricCard({
  label,
  value,
  delta,
  status,
  progress,
  footnote,
  index = 0,
}: MetricCardProps) {
  const isLive = delta === 'LIVE' || delta === 'TRACKING'
  
  return (
    <motion.article 
      className={`metric-card panel panel--metric panel--${status} ${isLive ? 'metric-card--live' : ''}`}
      initial={{ opacity: 0, scale: 0.98, y: 8 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ 
        type: 'spring', 
        stiffness: 260, 
        damping: 24, 
        delay: index * 0.05 
      }}
      whileHover={{ y: -2, transition: { duration: 0.2 } }}
    >
      <div className="metric-card__top">
        <p className="section-label">{label}</p>
        <span className="metric-card__delta">{delta}</span>
      </div>
      <strong className="metric-card__value">{value}</strong>
      <div
        className="metric-card__progress"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress)}
        aria-label={label}
      >
        <motion.span 
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 1, ease: "easeOut", delay: 0.2 + (index * 0.05) }}
        />
      </div>
      <p className="metric-card__footnote">{footnote}</p>
    </motion.article>
  )
}
