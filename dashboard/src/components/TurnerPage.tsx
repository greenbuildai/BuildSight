import React, { useState } from 'react'
import { TurnerAssistant } from './TurnerAssistant'
import { TurnerVoiceMode } from './TurnerVoiceMode'

type TurnerTab = 'chat' | 'voice'

interface TurnerPageProps {
  onBack: () => void
  onOpenSettings?: () => void
}

export const TurnerPage: React.FC<TurnerPageProps> = ({ onBack, onOpenSettings }) => {
  const [tab, setTab] = useState<TurnerTab>('chat')

  return (
    <main className="dashboard dashboard--turner">
      <header className="topbar">
        <div className="topbar__brand">
          {tab !== 'voice' && (
            <>
              <p className="eyebrow">AI Supervisor</p>
              <h2>Turner Command Center</h2>
            </>
          )}
        </div>
        <div className="topbar__actions">
          <div className="turner-tab-switch">
            <button
              type="button"
              className={`turner-tab-btn ${tab === 'chat' ? 'turner-tab-btn--active' : ''}`}
              onClick={() => setTab('chat')}
            >
              <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true">
                <path d="M20 2H4a2 2 0 0 0-2 2v18l4-4h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2Zm-2 10H6v-2h12v2Zm0-4H6V6h12v2Z" />
              </svg>
              Chat Mode
            </button>
            <button
              type="button"
              className={`turner-tab-btn ${tab === 'voice' ? 'turner-tab-btn--active' : ''}`}
              onClick={() => setTab('voice')}
            >
              <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true">
                <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4Zm0 14.5a6.5 6.5 0 0 0 6.5-6.5.5.5 0 0 1 1 0 7.5 7.5 0 0 1-7 7.48V19h2a.5.5 0 0 1 0 1h-5a.5.5 0 0 1 0-1h2v-2.02A7.5 7.5 0 0 1 4.5 9a.5.5 0 0 1 1 0 6.5 6.5 0 0 0 6.5 6.5Z" />
              </svg>
              Voice Mode
            </button>
          </div>
          <button type="button" className="stg-back-btn" onClick={onBack}>
            <span className="stg-back-btn__icon">←</span>
            <span>Back to Dashboard</span>
          </button>
        </div>
      </header>

      <div className="turner-page__body">
        {tab === 'chat' && (
          <div className="turner-page__chat">
            <TurnerAssistant isHero onOpenSettings={onOpenSettings} />
          </div>
        )}
        {tab === 'voice' && (
          <div className="turner-page__voice">
            <TurnerVoiceMode />
          </div>
        )}
      </div>
    </main>
  )
}
