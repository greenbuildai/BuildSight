import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'

/* ═══════════════════════════════════════════════════════════════════════════════
   BuildSight Settings — Context & Persistence
   All settings are saved to localStorage and restored on load.
   ═══════════════════════════════════════════════════════════════════════════════ */

// ── Shape ────────────────────────────────────────────────────────────────────

export interface Settings {
  // ── Per-class confidence thresholds ────────────────────────────────────────
  // These replace the single confidenceThreshold for fine-grained control.
  workerConf:  number   // 0.20 — workers in low-contrast / dusty scenes score low
  helmetConf:  number   // 0.30 — raised to suppress hair/shoulder false positives
  vestConf:    number   // 0.18 — low so partially-occluded vests still pass

  // ── Per-class NMS IoU thresholds ───────────────────────────────────────────
  // Higher = less NMS suppression = adjacent workers both survive
  workerNmsIou:  number // 0.60 — prevents adjacent-worker suppression
  helmetNmsIou:  number // 0.55
  vestNmsIou:    number // 0.55

  // ── WBF fusion IoU per class ───────────────────────────────────────────────
  // Higher = only near-identical boxes fuse = different workers stay separate
  wbfWorkerIou:  number // 0.65
  wbfHelmetIou:  number // 0.45
  wbfVestIou:    number // 0.50

  // ── Image enhancement ──────────────────────────────────────────────────────
  enableClahe:  boolean // CLAHE contrast enhancement before inference
  claheClip:    number  // CLAHE clip limit (1.0–4.0, default 2.0)

  // Legacy single-slider (kept for backward compat — not used in inference)
  confidenceThreshold: number
  iouThreshold:        number
  wbfThreshold:        number

  detectWorker:   boolean
  detectHelmet:   boolean
  detectVest:     boolean
  restrictedZone: boolean
  showConfScores: boolean
  realtimeAlerts: boolean

  // Camera & Video
  cameraSource: 'default' | 'external' | 'cctv'
  cctvStreamUrl: string
  uploadQuality: 'low' | 'medium' | 'high'
  frameRate: number
  playbackSpeed: number
  autoStartDetection: boolean

  // Appearance
  theme: 'dark' | 'light'
  accentColor: string
  uiFontSize: 'small' | 'default' | 'large'
  compactView: boolean
  showSidebarLabels: boolean
  enableAnimations: boolean

  // Notifications
  soundAlerts: boolean
  criticalHighlight: boolean
  emailNotifications: boolean
  smsNotifications: boolean
  alertSensitivity: 'low' | 'medium' | 'high'
  escalationMinutes: number

  // System
  processingMode: 'gpu' | 'cpu' | 'auto'
  detectionQuality: 'fast' | 'balanced' | 'accurate'
  memoryOptimization: boolean
  debugMode: boolean
  selectedModel: string

  // User
  profileName: string
  language: string
  dateFormat: string
  timeFormat: '12h' | '24h'

  // Site Metadata
  siteCondition: string
}

// ── Defaults ─────────────────────────────────────────────────────────────────

export const DEFAULT_SETTINGS: Settings = {
  // Per-class conf
  workerConf:  0.20,
  helmetConf:  0.30,
  vestConf:    0.18,

  // Per-class NMS IoU
  workerNmsIou:  0.60,
  helmetNmsIou:  0.55,
  vestNmsIou:    0.55,

  // WBF IoU
  wbfWorkerIou:  0.65,
  wbfHelmetIou:  0.45,
  wbfVestIou:    0.50,

  // Enhancement
  enableClahe: true,
  claheClip:   2.0,

  // Legacy
  confidenceThreshold: 0.25,
  iouThreshold:        0.45,
  wbfThreshold:        0.55,

  detectWorker:   true,
  detectHelmet:   true,
  detectVest:     true,
  restrictedZone: true,
  showConfScores: true,
  realtimeAlerts: true,

  cameraSource: 'default',
  cctvStreamUrl: '',
  uploadQuality: 'high',
  frameRate: 15,
  playbackSpeed: 1.0,
  autoStartDetection: false,

  theme: 'dark',
  accentColor: '#ff4b00',
  uiFontSize: 'default',
  compactView: false,
  showSidebarLabels: true,
  enableAnimations: true,

  soundAlerts: false,
  criticalHighlight: true,
  emailNotifications: false,
  smsNotifications: false,
  alertSensitivity: 'medium',
  escalationMinutes: 5,

  processingMode: 'auto',
  detectionQuality: 'balanced',
  memoryOptimization: false,
  debugMode: false,
  selectedModel: 'ensemble-wbf',

  profileName: 'Site Supervisor',
  language: 'en',
  dateFormat: 'DD/MM/YYYY',
  timeFormat: '24h',
  siteCondition: 'Normal operational load; clear visibility; scheduled maintenance in Zone B.',
}

const STORAGE_KEY = 'buildsight_settings'

// ── Context ──────────────────────────────────────────────────────────────────

interface SettingsCtx {
  settings: Settings
  update: <K extends keyof Settings>(key: K, value: Settings[K]) => void
  resetAll: () => void
  resetCategory: (keys: (keyof Settings)[]) => void
}

const Ctx = createContext<SettingsCtx | null>(null)

export function useSettings() {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useSettings must be used inside SettingsProvider')
  return ctx
}

// ── Provider ─────────────────────────────────────────────────────────────────

function load(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) }
  } catch { /* corrupt data — use defaults */ }
  return { ...DEFAULT_SETTINGS }
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(load)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
  }, [settings])

  const update = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings(prev => ({ ...prev, [key]: value }))
  }

  const resetAll = () => setSettings({ ...DEFAULT_SETTINGS })

  const resetCategory = (keys: (keyof Settings)[]) => {
    setSettings(prev => {
      const next = { ...prev }
      keys.forEach(k => { (next as Record<string, unknown>)[k] = DEFAULT_SETTINGS[k] })
      return next
    })
  }

  return <Ctx.Provider value={{ settings, update, resetAll, resetCategory }}>{children}</Ctx.Provider>
}
