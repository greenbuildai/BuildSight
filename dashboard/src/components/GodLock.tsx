import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import './GodLock.css'

interface GodLockProps {
  onUnlock: () => void
  onClose: () => void
}

export function GodLock({ onUnlock, onClose }: GodLockProps) {
  const [password, setPassword] = useState('')
  const [error, setError] = useState(false)
  const [attempts, setAttempts] = useState(0)
  const [lockout, setLockout] = useState(false)
  const [isDecrypting, setIsDecrypting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const CORRECT_PASSWORD = import.meta.env.VITE_GOD_MODE_PASSWORD || 'jovi#2748'

  useEffect(() => {
    // Auto-focus the input on mount
    const timer = setTimeout(() => inputRef.current?.focus(), 100)
    return () => clearTimeout(timer)
  }, [])

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault()
    if (lockout) return

    if (password.trim() === CORRECT_PASSWORD.trim()) {
      setIsDecrypting(true)
      setTimeout(() => {
        onUnlock()
      }, 1500)
    } else {
      setError(true)
      setAttempts(prev => prev + 1)
      setPassword('')
      
      if (attempts + 1 >= 5) {
        setLockout(true)
        setTimeout(() => {
          setLockout(false)
          setAttempts(0)
        }, 60000) // 1 minute lockout
      }

      setTimeout(() => setError(false), 500)
    }
  }

  return (
    <div className="god-lock-overlay">
      <div className="god-lock-background">
        <div className="god-lock-grid" />
        <div className="god-lock-red-glow" />
      </div>

      <motion.div 
        className="god-lock-card"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ 
          scale: 1, 
          opacity: 1,
          x: error ? [0, -10, 10, -10, 10, 0] : 0
        }}
        transition={{ duration: error ? 0.4 : 0.3 }}
      >
        <div className="god-lock-header">
          <div className="god-lock-alert-icon">⚠️</div>
          <div className="god-lock-titles">
            <h2>RESTRICTED EXECUTIVE ACCESS</h2>
            <p>BUILDSIGHT MASTER OVERRIDE INITIALIZED</p>
          </div>
        </div>

        <div className="god-lock-content">
          <AnimatePresence mode="wait">
            {isDecrypting ? (
              <motion.div 
                key="decrypting"
                className="god-lock-status"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="god-lock-loader" />
                <div className="god-lock-loader-text">
                  <motion.span
                    animate={{ opacity: [1, 0.4, 1] }}
                    transition={{ repeat: Infinity, duration: 0.5 }}
                  >
                    DECRYPTING NEURAL PARAMETERS...
                  </motion.span>
                  <span className="god-lock-loader-sub">AUTHORIZATION CONFIRMED</span>
                </div>
              </motion.div>
            ) : (
              <motion.div 
                key="input"
                className="god-lock-form-container"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <form className="god-lock-form" onSubmit={handleSubmit}>
                  <div className="god-lock-input-wrapper">
                    <label htmlFor="god-password">COMMAND OVERRIDE CODE</label>
                    <input
                      ref={inputRef}
                      id="god-password"
                      type="password"
                      placeholder={lockout ? "SYSTEM LOCKED" : "ENTER KEYCODE..."}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      disabled={lockout}
                      className={error ? 'god-input-error' : ''}
                      autoComplete="off"
                    />
                    {lockout && (
                      <div className="god-lock-timer">
                        LOCKOUT ACTIVE: SECURING SYSTEM...
                      </div>
                    )}
                  </div>
                  
                  <div className="god-lock-actions">
                    <button 
                      type="button" 
                      className="god-lock-btn god-lock-btn--cancel" 
                      onClick={onClose}
                    >
                      ABORT
                    </button>
                    <button 
                      type="submit" 
                      className="god-lock-btn god-lock-btn--execute"
                      disabled={lockout || !password}
                    >
                      EXECUTE
                    </button>
                  </div>
                </form>
                
                <div className="god-lock-attempts">
                  FAILED ATTEMPTS: {attempts}/5
                  {attempts > 0 && <span className="god-lock-warning"> · TRACING IP...</span>}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="god-lock-footer">
          <div className="god-lock-line" />
          <div className="god-lock-footer-text">
            <span>LEVEL 5 SECURED</span>
            <span>GOD_MODE_AUTH_v5.0</span>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
