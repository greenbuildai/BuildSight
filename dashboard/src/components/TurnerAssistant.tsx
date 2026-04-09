import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useDetectionStats } from '../DetectionStatsContext'
import { useSettings } from '../SettingsContext'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
}

type AssistantRoute = 'mistral' | 'gemini'
type TurnerIssueKind =
  | 'offline'
  | 'ai-unavailable'
  | 'delayed'
  | 'model-load-failure'
  | 'network'

interface TurnerIssue {
  kind: TurnerIssueKind
  title: string
  detail: string
}

const STORAGE_KEY = 'buildsight.turner.messages'

const QUICK_ACTIONS = [
  { label: 'Site Brief', query: 'I need a full site safety brief for the current shift.' },
  { label: 'PPE Compliance', query: 'Which zone has the lowest PPE compliance?' },
  { label: 'High-Vis Issues', query: 'List all active high-vis vest violations.' },
  { label: 'Hardhat Check', query: 'Identify all workers without hardhats.' },
  { label: 'Risk Zones', query: 'Generate a risk map of the current site.' },
  { label: 'Safety Drift', query: 'Analyze safety compliance drift in the last hour.' },
  { label: 'Active Alerts', query: 'Summarize the highest priority alerts now.' },
]

const ASSISTANT_MODELS: Array<{ value: AssistantRoute; label: string }> = [
  { value: 'mistral', label: 'Mistral' },
  { value: 'gemini', label: 'Gemini' },
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
  const lines = content.split('\n').map((line) => line.trim()).filter(Boolean)
  const source = lines.length > 0 ? lines : [content]

  return source.map((line, index) => {
    const isBullet = line.startsWith('- ') || line.startsWith('* ')

    return (
      <p
        key={`${line}-${index}`}
        className={`chat-bubble__line ${isBullet ? 'chat-bubble__line--bullet' : ''}`}
      >
        {isBullet ? line.slice(2) : line}
      </p>
    )
  })
}

function classifyIssue(status?: number, message?: string): TurnerIssue | null {
  const normalized = message?.toLowerCase() ?? ''

  if (status === 0) {
    return {
      kind: 'network',
      title: 'Network path unavailable',
      detail: 'Turner cannot reach the local AI service. Verify the dashboard backend is reachable.',
    }
  }

  if (status === 503 || normalized.includes('offline')) {
    return {
      kind: 'ai-unavailable',
      title: 'AI supervisor offline',
      detail: 'The Turner model route is unavailable. Live telemetry is still visible while the service recovers.',
    }
  }

  if (normalized.includes('model')) {
    return {
      kind: 'model-load-failure',
      title: 'Model load failure',
      detail: 'The requested Turner route could not initialize. Switch routing mode or retry after the backend stabilizes.',
    }
  }

  return null
}

function ChatAvatar({ role }: { role: Message['role'] }) {
  if (role === 'user') {
    return (
      <span className="chat-bubble__avatar-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" focusable="false">
          <path d="M12 3.5a4.5 4.5 0 0 0-4.5 4.5v1.1a2.8 2.8 0 0 0-1.8 2.6v1.1h12.6v-1.1a2.8 2.8 0 0 0-1.8-2.6V8A4.5 4.5 0 0 0 12 3.5Zm-6.1 10.8h12.2c.7 0 1.3.6 1.3 1.3v2.9h-2.1v-1.7H6.7v1.7H4.6v-2.9c0-.7.6-1.3 1.3-1.3Zm2.2 4h7.8v2.2H8.1v-2.2Z" fill="currentColor" />
        </svg>
      </span>
    )
  }

  return (
    <span className="chat-bubble__avatar-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" focusable="false">
        <path d="M12 3.2 5.8 6v3.2c0 4 2.6 7.7 6.2 8.9 3.6-1.2 6.2-4.9 6.2-8.9V6L12 3.2Zm0 2.1 3.8 1.7v2.2h-1.1v1.6a2.7 2.7 0 0 1-5.4 0V9.2H8.2V7l3.8-1.7Zm-1 5.5h2v.6a1 1 0 0 1-2 0v-.6Zm-2.8 8.1c1.1-.5 2.4-.8 3.8-.8s2.7.3 3.8.8v1.9H8.2v-1.9Z" fill="currentColor" />
      </svg>
    </span>
  )
}

function IssueBanner({ issue }: { issue: TurnerIssue }) {
  return (
    <div className={`turner-state-card turner-state-card--${issue.kind}`} role="status">
      <div className="turner-state-card__signal" aria-hidden="true" />
      <div>
        <strong>{issue.title}</strong>
        <p>{issue.detail}</p>
      </div>
    </div>
  )
}

export const TurnerAssistant: React.FC<{ isHero?: boolean; onOpenSettings?: () => void }> = ({
  isHero = false,
  onOpenSettings: _onOpenSettings,
}) => {
  const { stats, liveAlerts } = useDetectionStats()
  const { settings } = useSettings()
  const [messages, setMessages] = useState<Message[]>(() => {
    const stored = loadInitialMessages()
    return stored.length > 0
      ? stored
      : [
          createMessage(
            'assistant',
            'Turner online. I am monitoring compliance, live detections, and escalation pressure. Ask for a site summary, PPE review, or immediate action plan.',
          ),
        ]
  })
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)
  const [assistantModel, setAssistantModel] = useState<AssistantRoute>('mistral')
  const [networkOnline, setNetworkOnline] = useState<boolean>(() => (
    typeof navigator === 'undefined' ? true : navigator.onLine
  ))
  const [uiIssue, setUiIssue] = useState<TurnerIssue | null>(null)
  const [isDelayed, setIsDelayed] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
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
  }, [messages, isTyping, isExpanded, uiIssue])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined
    }

    const handleOnline = () => {
      setNetworkOnline(true)
      setUiIssue((current) => (current?.kind === 'offline' ? null : current))
    }
    const handleOffline = () => {
      setNetworkOnline(false)
      setUiIssue({
        kind: 'offline',
        title: 'No network connection',
        detail: 'The dashboard is offline. Turner can still show prior conversation context, but new requests will not send.',
      })
    }

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  useEffect(() => {
    if (!isTyping) {
      setIsDelayed(false)
      return undefined
    }

    const timer = window.setTimeout(() => {
      setIsDelayed(true)
      setUiIssue((current) => current ?? {
        kind: 'delayed',
        title: 'Response delayed',
        detail: 'Turner is still processing live telemetry and model context. The stream remains active.',
      })
    }, 4000)

    return () => window.clearTimeout(timer)
  }, [isTyping])

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
  const activeAlerts = liveAlerts.slice(0, 5)
  const currentIssue = useMemo(() => {
    if (!networkOnline) {
      return {
        kind: 'offline',
        title: 'No network connection',
        detail: 'The dashboard is offline. Turner can still show prior conversation context, but new requests will not send.',
      } satisfies TurnerIssue
    }

    return uiIssue
  }, [networkOnline, uiIssue])

  const buildContext = () => ({
    workers: stats.totalWorkers,
    helmets: stats.helmetsDetected,
    vests: stats.vestsDetected,
    condition: settings.siteCondition,
    assistantRoute: assistantModel,
    alerts: activeAlerts.map((a) => `${a.severity.toUpperCase()}: ${a.title} (${a.camera})`),
    telemetry: {
      totalWorkers: stats.totalWorkers,
      helmetsDetected: stats.helmetsDetected,
      vestsDetected: stats.vestsDetected,
      proximityViolations: stats.proximityViolations,
      avgConfidence: stats.avgConfidence,
      elapsedMs: stats.elapsedMs,
      framesScanned: stats.framesScanned,
      isRunning: stats.isRunning,
      modelName: stats.modelName,
    },
  })

  const handleSend = async (rawText: string = inputValue) => {
    const text = rawText.trim()
    if (!text || isTyping) {
      return
    }

    if (!networkOnline) {
      setUiIssue({
        kind: 'offline',
        title: 'No network connection',
        detail: 'The dashboard is offline. Restore connectivity before sending a new Turner request.',
      })
      return
    }

    const historySnapshot = messagesRef.current.map((m) => ({ role: m.role, content: m.content }))
    setMessages((prev) => [...prev, createMessage('user', text)])
    setInputValue('')
    setIsTyping(true)
    setIsDelayed(false)
    setUiIssue(null)

    const payload = JSON.stringify({
      message: text,
      history: historySnapshot,
      context: buildContext(),
    })

    const headers = { 'Content-Type': 'application/json' }

    try {
      const streamResp = await fetch('http://localhost:8000/api/ai/chat/stream', {
        method: 'POST',
        headers,
        body: payload,
      })

      if (streamResp.ok && streamResp.headers.get('content-type')?.includes('text/event-stream')) {
        const assistantId = `assistant-${Date.now()}`
        setStreamingMessageId(assistantId)
        setMessages((prev) => [
          ...prev,
          { id: assistantId, role: 'assistant', content: '', timestamp: new Date().toISOString() },
        ])
        setIsTyping(false)

        const reader = streamResp.body?.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let hasContent = false

        if (reader) {
          while (true) {
            const { done, value } = await reader.read()
            if (done) {
              break
            }

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() ?? ''

            for (const line of lines) {
              const trimmed = line.trim()
              if (!trimmed.startsWith('data:')) {
                continue
              }

              const data = trimmed.slice(5).trim()
              if (data === '[DONE]') {
                break
              }

              try {
                const parsed = JSON.parse(data) as { token?: string; error?: string }
                if (parsed.error) {
                  const issue = classifyIssue(503, parsed.error)
                  if (issue) {
                    setUiIssue(issue)
                  }
                  break
                }
                if (parsed.token) {
                  hasContent = true
                  setMessages((prev) => prev.map((message) => (
                    message.id === assistantId
                      ? { ...message, content: message.content + parsed.token }
                      : message
                  )))
                }
              } catch {
                // Ignore malformed SSE frames so the stream can continue.
              }
            }
          }
        }

        if (!hasContent) {
          setMessages((prev) => prev.map((message) => (
            message.id === assistantId
              ? {
                  ...message,
                  content: 'Turner returned an empty stream. Retry the request or switch the routing mode.',
                }
              : message
          )))
          setUiIssue({
            kind: 'model-load-failure',
            title: 'Empty AI stream',
            detail: 'The Turner route opened but did not return usable tokens. This typically indicates a model startup issue.',
          })
        }

        return
      }

      const fallbackData = await streamResp.json().catch(() => null)
      const issue = classifyIssue(streamResp.status, typeof fallbackData?.error === 'string' ? fallbackData.error : undefined)
      if (issue) {
        setUiIssue(issue)
      }
    } catch {
      // Fall through to non-streaming request.
    }

    try {
      const resp = await fetch('http://localhost:8000/api/ai/chat', {
        method: 'POST',
        headers,
        body: payload,
      })
      const data = await resp.json().catch(() => null)
      const reply = typeof data?.response === 'string' && data.response.trim()
        ? data.response.trim()
        : "I couldn't complete that review. Check the backend log for the Turner request trace."

      const issue = classifyIssue(resp.status, typeof data?.error === 'string' ? data.error : reply)
      if (issue) {
        setUiIssue(issue)
      }

      setMessages((prev) => [...prev, createMessage('assistant', reply)])
    } catch {
      setUiIssue({
        kind: 'network',
        title: 'Turner backend unreachable',
        detail: 'The local FastAPI service is not responding on port 8000. Monitoring remains available, but AI replies are paused.',
      })
      setMessages((prev) => [
        ...prev,
        createMessage('assistant', 'The Turner backend is unreachable. Verify the FastAPI service is running on port 8000.'),
      ])
    } finally {
      setIsTyping(false)
      setIsDelayed(false)
      setStreamingMessageId(null)
    }
  }

  const renderConversation = () => (
    <div className="turner-assistant__messages-container">
      <div className="turner-assistant__messages" ref={scrollRef} id="turner-scroll-container">
        {currentIssue && <IssueBanner issue={currentIssue} />}

        {messages.length === 0 && (
          <div className="turner-empty-state">
            <span className="turner-empty-state__eyebrow">Turner Ready</span>
            <strong>Request a tactical site readout</strong>
            <p>Use a quick action or enter a direct prompt for compliance, risk, or shift-level decisions.</p>
          </div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((message) => (
            <motion.article
              key={message.id}
              className={`chat-bubble chat-bubble--${message.role} ${message.id === streamingMessageId ? 'chat-bubble--streaming' : ''}`}
              initial={{ opacity: 0, scale: 0.985, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ type: 'spring', stiffness: 400, damping: 32 }}
            >
              <div className="chat-bubble__avatar" aria-hidden="true">
                <ChatAvatar role={message.role} />
              </div>
              <div className="chat-bubble__main">
                <div className="chat-bubble__meta">
                  <span className="chat-bubble__sender">
                    {message.role === 'assistant' ? 'TURNER-AI' : 'OPERATOR'}
                  </span>
                  <span className="chat-bubble__timestamp">{formatTime(message.timestamp)}</span>
                </div>
                <div className="chat-bubble__body">
                  <div className="chat-bubble__content">
                    {renderMessageContent(message.content)}
                  </div>
                </div>
              </div>
            </motion.article>
          ))}
        </AnimatePresence>

        <AnimatePresence>
          {isTyping && (
            <motion.article
              className="chat-bubble chat-bubble--assistant chat-bubble--thinking"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <div className="chat-bubble__avatar" aria-hidden="true">
                <ChatAvatar role="assistant" />
              </div>
              <div className="chat-bubble__main">
                <div className="chat-bubble__meta">
                  <span className="chat-bubble__sender">TURNER-AI</span>
                  <span className="chat-bubble__status">{isDelayed ? 'QUEUE HOLD' : 'ANALYZING'}</span>
                </div>
                <div className="chat-bubble__body">
                  <div className="turner-thinking">
                    <div className="turner-thinking__pulse" />
                    <div className="turner-thinking__bars" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                      <span />
                    </div>
                    <p>{isDelayed ? 'Compiling a delayed response from live site telemetry...' : 'Processing live site telemetry...'}</p>
                  </div>
                </div>
              </div>
            </motion.article>
          )}
        </AnimatePresence>
        <div className="turner-scroll-spacer" />
      </div>
    </div>
  )

  const renderComposer = () => (
    <div className="turner-assistant__footer">
      <form
        className="turner-composer"
        onSubmit={(event) => {
          event.preventDefault()
          void handleSend()
        }}
      >
        <div className="turner-composer__shell">
          <div className="turner-composer__row">
            <div className="turner-composer__field">
              <input
                type="text"
                placeholder="Query PPE drift, zone risk, live alerts, or next action..."
                value={inputValue}
                className="turner-composer__input"
                onChange={(event) => setInputValue(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault()
                    void handleSend()
                  }
                }}
              />
            </div>

            <label className="turner-composer__model-wrap">
              <span className="turner-composer__model-label">Model</span>
              <select
                className="turner-composer__select"
                value={assistantModel}
                onChange={(event) => {
                  setAssistantModel(event.target.value as AssistantRoute)
                  setUiIssue((current) => (current?.kind === 'model-load-failure' ? null : current))
                }}
                disabled={isTyping}
              >
                {ASSISTANT_MODELS.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>

            <div
              className={`turner-composer__wave ${isTyping ? 'turner-composer__wave--active' : ''}`}
              aria-hidden="true"
            >
              <span />
              <span />
              <span />
              <span />
              <span />
            </div>

            <button
              type="submit"
              className="turner-composer__send"
              disabled={!inputValue.trim() || isTyping}
              aria-label="Send prompt"
            >
              <span className="turner-composer__send-arrow" aria-hidden="true">
                <svg viewBox="0 0 24 24" focusable="false">
                  <path d="M12 18.8a.9.9 0 0 1-.9-.9V8.27l-3.02 3.02a.9.9 0 1 1-1.27-1.27l4.55-4.55a.9.9 0 0 1 1.27 0l4.55 4.55a.9.9 0 1 1-1.27 1.27L12.9 8.27v9.63a.9.9 0 0 1-.9.9Z" fill="currentColor" />
                </svg>
              </span>
            </button>
          </div>

          <div className="turner-composer__meta">
            <span className="turner-composer__status">
              <span className={`turner-composer__presence turner-composer__presence--${networkOnline ? 'online' : 'offline'}`} aria-hidden="true" />
              {networkOnline ? 'Link Stable' : 'Offline'}
            </span>
            <span className="turner-composer__model-readout">{stats.modelName || 'AI route standby'}</span>
            <span className="turner-composer__context">{activeAlerts.length} active alerts</span>
          </div>
        </div>
      </form>

      <div className="turner-assistant__chips">
        {QUICK_ACTIONS.map((action, index) => (
          <motion.button
            key={action.label}
            type="button"
            className="turner-chip"
            onClick={() => {
              void handleSend(action.query)
            }}
            disabled={isTyping}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 + index * 0.025 }}
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.98 }}
          >
            <span className="turner-chip__dot" aria-hidden="true" />
            {action.label}
          </motion.button>
        ))}
      </div>
    </div>
  )

  const renderPanel = (inModal = false) => (
    <section className={`turner-assistant ${isHero ? 'turner-assistant--hero' : ''} ${inModal ? 'turner-assistant--modal' : ''}`}>
      <div className="panel-content-wrapper turner-assistant__panel">
        <div className="panel-heading turner-assistant__heading">
          <div>
            <p className="section-label">AI Supervisor</p>
            <h3>Turner AI Chat</h3>
          </div>
          <div className="turner-assistant__heading-meta">
            {!inModal && (
              <p className="panel-meta">Live site questions, summaries, and follow-up actions</p>
            )}
            <button type="button" className="turner-assistant__expand" onClick={() => setIsExpanded(true)}>
              Expand
            </button>
          </div>
        </div>

        <div className="turner-assistant__body">
          {renderConversation()}
          {renderComposer()}
        </div>
      </div>
    </section>
  )

  const modal = isExpanded && typeof document !== 'undefined'
    ? createPortal(
        <div className="turner-modal" role="dialog" aria-modal="true" aria-label="Turner full review">
          <motion.div
            className="turner-modal__backdrop"
            onClick={() => setIsExpanded(false)}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />
          <motion.div
            className="turner-modal__surface"
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          >
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
          </motion.div>
        </div>,
        document.body,
      )
    : null

  return (
    <AnimatePresence>
      {renderPanel()}
      {modal}
    </AnimatePresence>
  )
}
