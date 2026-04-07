import React, { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useDetectionStats } from '../DetectionStatsContext'
import { useSettings } from '../SettingsContext'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

const STORAGE_KEY = 'buildsight.turner.messages'

const QUICK_ACTIONS = [
  { label: 'Site Summary',    query: 'Give me a summary of the current site status.' },
  { label: 'Helmet Check',   query: 'How many workers are not wearing helmets?' },
  { label: 'Highest Risk',   query: 'Which area has the highest risk right now?' },
  { label: 'Top Violation',  query: 'Show me the most common safety violation.' },
  { label: 'PPE Compliance', query: 'Which zone has the lowest PPE compliance?' },
  { label: 'Active Workers', query: 'How many workers are currently active on site?' },
  { label: 'Safety Brief',   query: 'Give me a safety summary for today.' },
]

function createMessage(role: Message['role'], content: string): Message {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    content,
    timestamp: new Date().toISOString(),
  }
}

function loadInitialMessages(): Message[] {
  if (typeof window === 'undefined') {
    return []
  }

  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) {
      return []
    }

    return parsed.filter((item): item is Message => (
      typeof item?.id === 'string'
      && (item?.role === 'user' || item?.role === 'assistant')
      && typeof item?.content === 'string'
      && typeof item?.timestamp === 'string'
    ))
  } catch {
    return []
  }
}

function formatTime(timestamp: string) {
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function renderMessageContent(content: string) {
  return content.split('\n').filter(Boolean).map((line, index) => {
    const trimmed = line.trim()
    const isBullet = trimmed.startsWith('- ') || trimmed.startsWith('* ')

    return (
      <p
        key={`${line}-${index}`}
        className={`chat-bubble__line ${isBullet ? 'chat-bubble__line--bullet' : ''}`}
      >
        {isBullet ? trimmed.slice(2) : trimmed}
      </p>
    )
  })
}

export const TurnerAssistant: React.FC<{ isHero?: boolean }> = ({ isHero = false }) => {
  const { stats, liveAlerts } = useDetectionStats()
  const { settings } = useSettings()
  const [messages, setMessages] = useState<Message[]>(() => {
    const stored = loadInitialMessages()
    return stored.length > 0
      ? stored
      : [
          createMessage(
            'assistant',
            "Turner online. I am monitoring compliance, live detections, and escalation pressure. Ask for a site summary, PPE review, or immediate action plan.",
          ),
        ]
  })
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)
  const [requestError, setRequestError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const messagesRef = useRef<Message[]>(messages)

  useEffect(() => {
    messagesRef.current = messages
    if (typeof window !== 'undefined') {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages))
    }
  }, [messages])

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isTyping, isExpanded])

  useEffect(() => {
    if (!isExpanded) {
      return undefined
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsExpanded(false)
      }
    }

    document.body.classList.add('turner-modal-open')
    window.addEventListener('keydown', onKeyDown)

    return () => {
      document.body.classList.remove('turner-modal-open')
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [isExpanded])

  const workers = Math.max(stats.totalWorkers, 0)
  const helmetCompliance = workers > 0 ? Math.round((stats.helmetsDetected / workers) * 100) : 100
  const vestCompliance = workers > 0 ? Math.round((stats.vestsDetected / workers) * 100) : 100
  const confidencePct = Math.round(stats.avgConfidence * 100)
  const activeAlerts = liveAlerts.slice(0, 5)

  const buildContext = () => ({
    workers:  stats.totalWorkers,
    helmets:  stats.helmetsDetected,
    vests:    stats.vestsDetected,
    condition: settings.siteCondition,
    alerts: activeAlerts.map((a) => `${a.severity.toUpperCase()}: ${a.title} (${a.camera})`),
    telemetry: {
      totalWorkers:       stats.totalWorkers,
      helmetsDetected:    stats.helmetsDetected,
      vestsDetected:      stats.vestsDetected,
      proximityViolations: stats.proximityViolations,
      avgConfidence:      stats.avgConfidence,
      elapsedMs:          stats.elapsedMs,
      framesScanned:      stats.framesScanned,
      isRunning:          stats.isRunning,
      modelName:          stats.modelName,
    },
  })

  /** Streaming send — uses /api/ai/chat/stream (SSE), falls back to /api/ai/chat. */
  const handleSend = async (rawText: string = inputValue) => {
    const text = rawText.trim()
    if (!text || isTyping) return

    const historySnapshot = messagesRef.current.map((m) => ({ role: m.role, content: m.content }))
    setMessages((prev) => [...prev, createMessage('user', text)])
    setInputValue('')
    setIsTyping(true)
    setRequestError(null)

    const payload = JSON.stringify({ message: text, history: historySnapshot, context: buildContext() })
    const headers  = { 'Content-Type': 'application/json' }

    // ── Attempt streaming ────────────────────────────────────────────────────
    try {
      const streamResp = await fetch('http://localhost:8000/api/ai/chat/stream', {
        method: 'POST', headers, body: payload,
      })

      if (streamResp.ok && streamResp.headers.get('content-type')?.includes('text/event-stream')) {
        const assistantId = `assistant-${Date.now()}`
        // Seed the assistant bubble with empty content so it appears immediately
        setMessages((prev) => [...prev, { id: assistantId, role: 'assistant', content: '', timestamp: new Date().toISOString() }])
        setIsTyping(false)

        const reader = streamResp.body?.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        if (reader) {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() ?? ''

            for (const line of lines) {
              const trimmed = line.trim()
              if (!trimmed.startsWith('data:')) continue
              const payload = trimmed.slice(5).trim()
              if (payload === '[DONE]') break

              try {
                const obj = JSON.parse(payload) as { token?: string; error?: string }
                if (obj.error) { setRequestError(obj.error); break }
                if (obj.token) {
                  setMessages((prev) => prev.map((m) =>
                    m.id === assistantId ? { ...m, content: m.content + obj.token } : m
                  ))
                }
              } catch { /* ignore malformed SSE frame */ }
            }
          }
        }
        return
      }
    } catch { /* streaming failed — fall through to non-streaming */ }

    // ── Non-streaming fallback ───────────────────────────────────────────────
    try {
      const resp = await fetch('http://localhost:8000/api/ai/chat', {
        method: 'POST', headers, body: payload,
      })
      const data = await resp.json()
      const reply = typeof data?.response === 'string' && data.response.trim()
        ? data.response.trim()
        : "I couldn't complete that review. Check the backend log for the Turner request trace."

      setMessages((prev) => [...prev, createMessage('assistant', reply)])
      if (!resp.ok && typeof data?.error === 'string') setRequestError(data.error)
    } catch {
      setRequestError('BACKEND_UNREACHABLE')
      setMessages((prev) => [
        ...prev,
        createMessage('assistant', 'The Turner backend is unreachable. Verify the FastAPI service is running on port 8000.'),
      ])
    } finally {
      setIsTyping(false)
    }
  }

  const renderConversation = () => (
    <div className="turner-assistant__messages" ref={scrollRef}>
      {messages.map((message) => (
        <article key={message.id} className={`chat-bubble chat-bubble--${message.role}`}>
          <div className="chat-bubble__avatar" aria-hidden="true">
            {message.role === 'assistant' ? 'TR' : 'YO'}
          </div>
          <div className="chat-bubble__body">
            <div className="chat-bubble__meta">
              <strong>{message.role === 'assistant' ? 'Turner' : 'Operator'}</strong>
              <span>{formatTime(message.timestamp)}</span>
            </div>
            <div className="chat-bubble__content">
              {renderMessageContent(message.content)}
            </div>
          </div>
        </article>
      ))}

      {isTyping && (
        <article className="chat-bubble chat-bubble--assistant chat-bubble--thinking">
          <div className="chat-bubble__avatar" aria-hidden="true">TR</div>
          <div className="chat-bubble__body">
            <div className="chat-bubble__meta">
              <strong>Turner</strong>
              <span>Analyzing live telemetry</span>
            </div>
            <div className="turner-thinking">
              <div className="turner-thinking__pulse" aria-hidden="true" />
              <div>
                <strong>Turner is thinking...</strong>
                <p>Cross-checking PPE compliance, escalations, and live site conditions.</p>
              </div>
            </div>
          </div>
        </article>
      )}
    </div>
  )

  const renderComposer = () => (
    <div className="turner-assistant__footer">
      <div className="turner-assistant__chips">
        {QUICK_ACTIONS.map((action) => (
          <button
            key={action.label}
            type="button"
            className="turner-chip"
            onClick={() => void handleSend(action.query)}
            disabled={isTyping}
          >
            <span className="turner-chip__dot" aria-hidden="true" />
            {action.label}
          </button>
        ))}
      </div>

      <form
        className="turner-composer"
        onSubmit={(event) => {
          event.preventDefault()
          void handleSend()
        }}
      >
        <label className="turner-composer__field">
          <span className="sr-only">Ask Turner about site safety</span>
          <textarea
            rows={isExpanded ? 3 : 2}
            placeholder="Ask Turner for a site brief, PPE review, or next action..."
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
          />
        </label>
        <button type="submit" className="turner-composer__send" disabled={!inputValue.trim() || isTyping}>
          Dispatch
        </button>
      </form>

      {requestError && (
        <p className="turner-assistant__status">
          Request trace ended with `{requestError}`. The assistant response above reflects the backend fallback path.
        </p>
      )}
    </div>
  )

  const renderPanel = (inModal = false) => (
    <section className={`turner-assistant ${isHero ? 'turner-assistant--hero' : ''} ${inModal ? 'turner-assistant--modal' : ''}`}>
      <header className="turner-assistant__header">
        <div className="turner-assistant__identity">
          <div className="turner-assistant__badge">
            <span className="turner-assistant__icon" aria-hidden="true">TR</span>
            <div className="turner-assistant__title">
              <p>Turner AI Supervisor</p>
              <h4>Live Construction Safety Review</h4>
            </div>
          </div>

          <div className="turner-assistant__telemetry">
            <div className="turner-telemetry-card">
              <span>Workers</span>
              <strong className="monospaced-dataviz">{workers}</strong>
            </div>
            <div className="turner-telemetry-card">
              <span>Helmet</span>
              <strong className={`monospaced-dataviz ${helmetCompliance < 85 ? 'glow-critical' : 'glow-ok'}`}>
                {helmetCompliance}%
              </strong>
            </div>
            <div className="turner-telemetry-card">
              <span>Vest</span>
              <strong className={`monospaced-dataviz ${vestCompliance < 80 ? 'glow-critical' : 'glow-ok'}`}>
                {vestCompliance}%
              </strong>
            </div>
            <div className="turner-telemetry-card">
              <span>Confidence</span>
              <strong className={`monospaced-dataviz ${confidencePct < 70 ? 'glow-warning' : 'glow-ok'}`}>
                {confidencePct}%
              </strong>
            </div>
          </div>
        </div>

        <div className="turner-assistant__actions">
          <div className="turner-assistant__mode">
            <span className={`status-indicator status-indicator--${stats.isRunning ? 'active' : 'idle'}`} />
            {stats.isRunning ? 'Active Telemetry Sync' : 'Awaiting Live Scan'}
          </div>
          {!inModal && (
            <button
              type="button"
              className="turner-assistant__expand"
              onClick={() => setIsExpanded(true)}
            >
              Expand Review
            </button>
          )}
        </div>
      </header>

      <div className="turner-assistant__body">
        {renderConversation()}
        {renderComposer()}
      </div>
    </section>
  )

  const modal = isExpanded && typeof document !== 'undefined'
    ? createPortal(
        <div className="turner-modal" role="dialog" aria-modal="true" aria-label="Turner full review">
          <div className="turner-modal__backdrop" onClick={() => setIsExpanded(false)} />
          <div className="turner-modal__surface">
            <header className="turner-modal__header">
              <div>
                <p className="section-label">Expanded Review</p>
                <h3>Turner Intensive Safety Workspace</h3>
              </div>
              <button type="button" className="turner-modal__close" onClick={() => setIsExpanded(false)}>
                Close
              </button>
            </header>

            <div className="turner-modal__grid">
              <aside className="turner-modal__rail">
                <section className="turner-modal__snapshot">
                  <h4>Site Snapshot</h4>
                  <div className="turner-modal__stats">
                    <div>
                      <span>Condition</span>
                      <strong>{settings.siteCondition}</strong>
                    </div>
                    <div>
                      <span>Frames</span>
                      <strong>{stats.framesScanned}</strong>
                    </div>
                    <div>
                      <span>Latency</span>
                      <strong>{stats.elapsedMs}ms</strong>
                    </div>
                    <div>
                      <span>Model</span>
                      <strong>{stats.modelName || 'Pending'}</strong>
                    </div>
                  </div>
                </section>

                <section className="turner-modal__snapshot">
                  <h4>Recent Detections</h4>
                  <ul className="turner-modal__list">
                    <li>Total workers: {workers}</li>
                    <li>Helmets detected: {stats.helmetsDetected}</li>
                    <li>Vests detected: {stats.vestsDetected}</li>
                    <li>Proximity alerts: {stats.proximityViolations}</li>
                  </ul>
                </section>

                <section className="turner-modal__snapshot">
                  <h4>Escalation Queue</h4>
                  {activeAlerts.length > 0 ? (
                    <ul className="turner-modal__list turner-modal__list--alerts">
                      {activeAlerts.map((alert) => (
                        <li key={alert.id}>
                          <strong>{alert.title}</strong>
                          <span>{alert.camera}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="turner-modal__empty">No live escalations in the current review window.</p>
                  )}
                </section>
              </aside>

              <div className="turner-modal__chat">
                {renderPanel(true)}
              </div>
            </div>
          </div>
        </div>,
        document.body,
      )
    : null

  return (
    <>
      {renderPanel()}
      {modal}
    </>
  )
}
