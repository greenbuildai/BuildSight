/* ─────────────────────────────────────────────────────────────────────────────
 * VLMActivityFeed — Moondream2 site-activity narration panel
 * Polls /api/geoai/vlm/latest every VLM_POLL_MS; lets operators ask custom
 * questions via /api/geoai/vlm/query.
 * ──────────────────────────────────────────────────────────────────────────── */

import { useState, useEffect, useRef, useCallback } from 'react'
import type { VLMEntry } from '../types/geoai'
import './VLMActivityFeed.css'

const API_BASE = 'http://localhost:8000/api/geoai'
const VLM_POLL_MS = 10_000

function isOpticalVlmSource(source: string): boolean {
  return source === 'florence2' || source === 'moondream2' || source === 'vlm_chained_with_turner_ai'
}

interface FeedItem {
  id: number
  entry: VLMEntry
}

let _feedIdCounter = 0

export function VLMActivityFeed() {
  const [feed, setFeed] = useState<FeedItem[]>([])
  const [vlmAvailable, setVlmAvailable] = useState<boolean | null>(null)
  const [question, setQuestion] = useState('')
  const [querying, setQuerying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const feedEndRef = useRef<HTMLDivElement>(null)

  const pushEntry = useCallback((entry: VLMEntry) => {
    setVlmAvailable(entry.vlm_available)
    setFeed(prev => {
      // Deduplicate: skip if latest description is identical
      if (prev.length > 0 && prev[prev.length - 1].entry.description === entry.description) {
        return prev
      }
      const next = [...prev, { id: ++_feedIdCounter, entry }]
      return next.length > 20 ? next.slice(-20) : next
    })
  }, [])

  // ── Auto-poll ─────────────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/vlm/latest`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data: VLMEntry = await res.json()
        if (!cancelled) pushEntry(data)
        setError(null)
      } catch (err) {
        if (!cancelled) setError('VLM endpoint unreachable — backend may be offline')
      }
    }

    poll()
    const timer = setInterval(poll, VLM_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [pushEntry])

  // ── Auto-scroll to latest ─────────────────────────────────────────────────
  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [feed])

  // ── Custom question ───────────────────────────────────────────────────────
  const handleQuery = async () => {
    const q = question.trim()
    if (!q || querying) return
    setQuerying(true)
    try {
      const res = await fetch(`${API_BASE}/vlm/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, force_refresh: true }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: VLMEntry = await res.json()
      pushEntry(data)
      setQuestion('')
      setError(null)
    } catch {
      setError('Query failed — check backend connection')
    } finally {
      setQuerying(false)
    }
  }

  const formatTime = (ts: number) =>
    new Date(ts * 1000).toLocaleTimeString('en-US', {
      hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
    })

  const sourceLabel = (src: string) =>
    isOpticalVlmSource(src) ? 'VLM' : 'RULE-BASED'

  // ── Risk keyword highlighting ─────────────────────────────────────────────
  const highlightText = (text: string) => {
    const patterns: { re: RegExp; cls: string }[] = [
      { re: /\b(critical|violation|missing|risk|hazard|danger)\b/gi, cls: 'vlm-highlight--critical' },
      { re: /\b(warning|elevated|partial|impaired)\b/gi,             cls: 'vlm-highlight--warn' },
      { re: /\b(compliant|safe|clear|full|all workers)\b/gi,         cls: 'vlm-highlight--ok' },
    ]
    let result = text
    // Simple approach: wrap matched words in spans
    patterns.forEach(({ re, cls }) => {
      result = result.replace(re, m => `<span class="${cls}">${m}</span>`)
    })
    return result
  }

  return (
    <div className="vlm-feed">
      {/* Header */}
      <div className="vlm-feed__header">
        <div className="vlm-feed__title">
          <span className="vlm-feed__title-text">VLM ACTIVITY FEED</span>
          <span className={`vlm-feed__badge ${vlmAvailable === true ? 'vlm-feed__badge--live' : vlmAvailable === false ? 'vlm-feed__badge--fallback' : 'vlm-feed__badge--loading'}`}>
            {vlmAvailable === true ? 'MOONDREAM2' : vlmAvailable === false ? 'FALLBACK' : 'LOADING'}
          </span>
        </div>
        <span className="vlm-feed__subtitle">
          {vlmAvailable === false
            ? 'Rule-based descriptions — install transformers + torch to enable VLM'
            : 'AI-generated site narration · updates every 10s'}
        </span>
      </div>

      {/* Error banner */}
      {error && (
        <div className="vlm-feed__error">{error}</div>
      )}

      {/* Feed */}
      <div className="vlm-feed__scroll">
        {feed.length === 0 && !error && (
          <div className="vlm-feed__empty">
            <div className="vlm-feed__spinner" />
            <span>Waiting for first description…</span>
          </div>
        )}
        {feed.map(item => (
          <div key={item.id} className="vlm-entry">
            <div className="vlm-entry__meta">
              <span className="vlm-entry__time">{formatTime(item.entry.timestamp)}</span>
              <span className={`vlm-entry__source ${isOpticalVlmSource(item.entry.source) ? 'vlm-entry__source--vlm' : ''}`}>
                {sourceLabel(item.entry.source)}
              </span>
            </div>
            <p
              className="vlm-entry__text"
              dangerouslySetInnerHTML={{ __html: highlightText(item.entry.description) }}
            />
            {item.entry.question && item.entry.question !== '' && (
              <div className="vlm-entry__question">Q: {item.entry.question}</div>
            )}
          </div>
        ))}
        <div ref={feedEndRef} />
      </div>

      {/* Query input */}
      <div className="vlm-feed__input-row">
        <input
          className="vlm-feed__input"
          type="text"
          placeholder="Ask about the site…"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleQuery()}
          disabled={querying}
        />
        <button
          className="vlm-feed__send"
          onClick={handleQuery}
          disabled={querying || !question.trim()}
        >
          {querying ? '…' : 'ASK'}
        </button>
      </div>
    </div>
  )
}
