type MetricCardProps = {
  label: string
  value: string
  delta: string
  status: 'stable' | 'risk' | 'alert'
  progress: number
  footnote: string
}

export function MetricCard({
  label,
  value,
  delta,
  status,
  progress,
  footnote,
}: MetricCardProps) {
  const isLive = delta === 'LIVE' || delta === 'TRACKING'
  return (
    <article className={`metric-card panel panel--metric panel--${status} ${isLive ? 'metric-card--live' : ''}`}>
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
        <span style={{ width: `${progress}%` }} />
      </div>
      <p className="metric-card__footnote">{footnote}</p>
    </article>
  )
}
