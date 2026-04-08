import { motion, AnimatePresence } from 'framer-motion'

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
      <AnimatePresence initial={false}>
        {alerts.map((alert, idx) => (
          <motion.article 
            className={`alert-log__item alert-log__item--${alert.severity}`} 
            key={alert.id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 10 }}
            transition={{ 
              type: 'spring', 
              stiffness: 300, 
              damping: 30, 
              delay: Math.min(idx * 0.05, 0.3) 
            }}
          >
            <div className="alert-log__meta">
              <span>{alert.time}</span>
              <span>{alert.camera}</span>
              <span>{alert.id}</span>
            </div>
            <div className="alert-log__body">
              <h4>{alert.title}</h4>
              <p>{alert.detail}</p>
            </div>
          </motion.article>
        ))}
      </AnimatePresence>
    </div>
  )
}
