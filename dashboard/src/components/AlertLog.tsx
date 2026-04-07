export type AlertItem = {
  id: string
  time: string
  camera: string
  severity: 'critical' | 'warning' | 'info'
  title: string
  detail: string
}

type AlertLogProps = {
  alerts: AlertItem[]
}

export function AlertLog({ alerts }: AlertLogProps) {
  return (
    <div className="alert-log">
      {alerts.map((alert) => (
        <article className={`alert-log__item alert-log__item--${alert.severity}`} key={alert.id}>
          <div className="alert-log__meta">
            <span>{alert.time}</span>
            <span>{alert.camera}</span>
            <span>{alert.id}</span>
          </div>
          <div className="alert-log__body">
            <h4>{alert.title}</h4>
            <p>{alert.detail}</p>
          </div>
        </article>
      ))}
    </div>
  )
}
