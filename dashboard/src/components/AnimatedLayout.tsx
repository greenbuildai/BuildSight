import { type ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

/* ── Shared spring config ─────────────────────────────────────────────────── */
const springSnappy = { type: 'spring' as const, stiffness: 400, damping: 32 }
const springSmooth = { type: 'spring' as const, stiffness: 260, damping: 28 }

/* ── Page-level transition wrapper ─────────────────────────────────────────── */
export function PageTransition({
  children,
  viewKey,
}: {
  children: ReactNode
  viewKey: string
}) {
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={viewKey}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.22, ease: 'easeOut' }}
        style={{ width: '100%', height: '100%' }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  )
}

/* ── Staggered card reveal ─────────────────────────────────────────────────── */
export function CardReveal({
  children, index = 0, className = '',
}: {
  children: ReactNode
  index?: number
  className?: string
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, scale: 0.96, y: 12 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ delay: index * 0.06, ...springSmooth }}
    >
      {children}
    </motion.div>
  )
}

/* ── Sidebar slide-in ─────────────────────────────────────────────────────── */
export function SlideIn({
  children, direction = 'left', className = '',
}: {
  children: ReactNode
  direction?: 'left' | 'right'
  className?: string
}) {
  const x = direction === 'left' ? -20 : 20
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, x }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, ...springSnappy }}
      style={{ height: '100%' }}
    >
      {children}
    </motion.div>
  )
}

/* ── Generic fade-in ──────────────────────────────────────────────────────── */
export function FadeIn({
  children, delay = 0, className = '',
}: {
  children: ReactNode
  delay?: number
  className?: string
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.25, delay }}
    >
      {children}
    </motion.div>
  )
}

/* ── Animated bar (chart column) ──────────────────────────────────────────── */
export function AnimatedBar({
  heightPct, delay = 0, className = '',
}: {
  heightPct: number
  delay?: number
  className?: string
}) {
  return (
    <motion.span
      className={className}
      initial={{ height: 0 }}
      animate={{ height: `${heightPct}%` }}
      transition={{ delay, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    />
  )
}

/* ── List item stagger (for alert lists) ──────────────────────────────────── */
export function StaggerItem({
  children, index = 0,
}: {
  children: ReactNode
  index?: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 8 }}
      transition={{ delay: index * 0.04, duration: 0.2 }}
    >
      {children}
    </motion.div>
  )
}

/* ── Hover spring for interactive cards ───────────────────────────────────── */
export function HoverCard({
  children, className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <motion.div
      className={className}
      whileHover={{ scale: 1.015, y: -2 }}
      whileTap={{ scale: 0.99 }}
      transition={springSnappy}
    >
      {children}
    </motion.div>
  )
}
