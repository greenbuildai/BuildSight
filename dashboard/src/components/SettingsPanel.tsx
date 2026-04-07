import { useState } from 'react'
import { useSettings, type Settings } from '../SettingsContext'

/* ═══════════════════════════════════════════════════════════════════════════════
   BuildSight Settings Panel
   6 tabbed categories — fully controlled via SettingsContext
   ═══════════════════════════════════════════════════════════════════════════════ */

const TABS = [
  { key: 'detection',     label: 'Detection' },
  { key: 'camera',        label: 'Camera & Video' },
  { key: 'appearance',    label: 'Appearance' },
  { key: 'notifications', label: 'Notifications' },
  { key: 'system',        label: 'System' },
  { key: 'user',          label: 'User' },
] as const

type TabKey = typeof TABS[number]['key']

/* ── Reusable control atoms ───────────────────────────────────────────────── */

function Toggle({ label, hint, value, onChange }: {
  label: string; hint?: string; value: boolean; onChange: (v: boolean) => void
}) {
  return (
    <label className="stg-toggle">
      <div className="stg-toggle__text">
        <span>{label}</span>
        {hint && <span className="stg-hint">{hint}</span>}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={value}
        className={`stg-switch ${value ? 'stg-switch--on' : ''}`}
        onClick={() => onChange(!value)}
      >
        <span className="stg-switch__thumb" />
      </button>
    </label>
  )
}

function Slider({ label, hint, value, min, max, step, unit, onChange }: {
  label: string; hint?: string; value: number; min: number; max: number;
  step: number; unit?: string; onChange: (v: number) => void
}) {
  return (
    <div className="stg-slider">
      <div className="stg-slider__head">
        <div className="stg-toggle__text">
          <span>{label}</span>
          {hint && <span className="stg-hint">{hint}</span>}
        </div>
        <strong>{typeof value === 'number' && value < 1 && unit === '%'
          ? `${(value * 100).toFixed(0)}%`
          : `${value}${unit ?? ''}`}</strong>
      </div>
      <input
        type="range" className="conf-slider__input"
        min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
      />
    </div>
  )
}

function Select<T extends string>({ label, hint, value, options, onChange }: {
  label: string; hint?: string; value: T;
  options: { value: T; label: string }[]; onChange: (v: T) => void
}) {
  return (
    <div className="stg-select">
      <div className="stg-toggle__text">
        <span>{label}</span>
        {hint && <span className="stg-hint">{hint}</span>}
      </div>
      <select
        className="stg-select__input"
        value={value}
        onChange={e => onChange(e.target.value as T)}
      >
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  )
}

function TextInput({ label, hint, value, placeholder, onChange }: {
  label: string; hint?: string; value: string; placeholder?: string;
  onChange: (v: string) => void
}) {
  return (
    <div className="stg-text-input">
      <div className="stg-toggle__text">
        <span>{label}</span>
        {hint && <span className="stg-hint">{hint}</span>}
      </div>
      <input
        type="text" className="stg-text-input__field"
        value={value} placeholder={placeholder}
        onChange={e => onChange(e.target.value)}
      />
    </div>
  )
}

function SectionTitle({ title }: { title: string }) {
  return <h4 className="stg-section-title">{title}</h4>
}

function ActionBtn({ label, variant, onClick }: {
  label: string; variant?: 'danger' | 'primary'; onClick: () => void
}) {
  return (
    <button
      className={`stg-action-btn ${variant ? `stg-action-btn--${variant}` : ''}`}
      onClick={onClick}
    >
      {label}
    </button>
  )
}

/* ── Category panels ──────────────────────────────────────────────────────── */

const DETECTION_KEYS: (keyof Settings)[] = [
  'workerConf', 'helmetConf', 'vestConf',
  'workerNmsIou', 'helmetNmsIou', 'vestNmsIou',
  'wbfWorkerIou', 'wbfHelmetIou', 'wbfVestIou',
  'enableClahe', 'claheClip',
  'detectWorker', 'detectHelmet', 'detectVest',
  'restrictedZone', 'showConfScores', 'realtimeAlerts',
]

function DetectionTab() {
  const { settings: s, update, resetCategory } = useSettings()
  return (
    <div className="stg-tab-body">

      <SectionTitle title="Detection Classes" />
      <Toggle label="Worker Detection"      hint="Detect and track worker bounding boxes" value={s.detectWorker} onChange={v => update('detectWorker', v)} />
      <Toggle label="Helmet Detection"      hint="Detect hard-hat presence on workers"    value={s.detectHelmet} onChange={v => update('detectHelmet', v)} />
      <Toggle label="Safety Vest Detection" hint="Detect high-vis vest compliance"        value={s.detectVest}   onChange={v => update('detectVest',   v)} />

      <SectionTitle title="Confidence Thresholds (per class)" />
      <p className="stg-hint" style={{ marginBottom: '0.5rem' }}>
        Minimum score for a detection to pass after WBF fusion. Changes apply on the next inference frame — no restart needed.
      </p>
      <Slider label="Worker Confidence" hint="Lower catches dusty / low-contrast workers"
        value={s.workerConf} min={0.05} max={0.90} step={0.05} unit="%"
        onChange={v => update('workerConf', v)} />
      <Slider label="Helmet Confidence" hint="Raise to suppress hair / shoulder false positives"
        value={s.helmetConf} min={0.05} max={0.90} step={0.05} unit="%"
        onChange={v => update('helmetConf', v)} />
      <Slider label="Safety Vest Confidence" hint="Lower to catch partially-occluded vests"
        value={s.vestConf} min={0.05} max={0.90} step={0.05} unit="%"
        onChange={v => update('vestConf', v)} />

      <SectionTitle title="NMS IoU Thresholds (per class)" />
      <p className="stg-hint" style={{ marginBottom: '0.5rem' }}>
        Non-maximum suppression inside each YOLO model. Higher = less suppression = more boxes survive (important for crowded scenes).
      </p>
      <Slider label="Worker NMS IoU" hint="Raise so adjacent workers are not suppressed"
        value={s.workerNmsIou} min={0.10} max={0.90} step={0.05} unit="%"
        onChange={v => update('workerNmsIou', v)} />
      <Slider label="Helmet NMS IoU"
        value={s.helmetNmsIou} min={0.10} max={0.90} step={0.05} unit="%"
        onChange={v => update('helmetNmsIou', v)} />
      <Slider label="Safety Vest NMS IoU"
        value={s.vestNmsIou} min={0.10} max={0.90} step={0.05} unit="%"
        onChange={v => update('vestNmsIou', v)} />

      <SectionTitle title="WBF Fusion IoU (per class)" />
      <p className="stg-hint" style={{ marginBottom: '0.5rem' }}>
        Weighted Box Fusion threshold. Higher = only near-identical boxes from both models fuse — keeps separate workers separate.
      </p>
      <Slider label="Worker WBF IoU" hint="Raise to prevent adjacent workers being merged"
        value={s.wbfWorkerIou} min={0.10} max={0.95} step={0.05} unit="%"
        onChange={v => update('wbfWorkerIou', v)} />
      <Slider label="Helmet WBF IoU"
        value={s.wbfHelmetIou} min={0.10} max={0.95} step={0.05} unit="%"
        onChange={v => update('wbfHelmetIou', v)} />
      <Slider label="Safety Vest WBF IoU"
        value={s.wbfVestIou} min={0.10} max={0.95} step={0.05} unit="%"
        onChange={v => update('wbfVestIou', v)} />

      <SectionTitle title="Image Enhancement" />
      <Toggle label="CLAHE Contrast Enhancement"
        hint="Adaptive histogram equalisation — improves detection in dusty / cement-coloured scenes"
        value={s.enableClahe} onChange={v => update('enableClahe', v)} />
      <Slider label="CLAHE Clip Limit" hint="Higher = more aggressive contrast boost (1.0–4.0)"
        value={s.claheClip} min={1.0} max={4.0} step={0.5} unit=""
        onChange={v => update('claheClip', v)} />

      <SectionTitle title="Overlay & Alerts" />
      <Toggle label="Restricted Zone Detection" hint="Highlight and monitor exclusion zones" value={s.restrictedZone} onChange={v => update('restrictedZone', v)} />
      <Toggle label="Show Confidence Scores"    hint="Display confidence % on bounding box labels" value={s.showConfScores} onChange={v => update('showConfScores', v)} />
      <Toggle label="Real-time Alerts"          hint="Trigger alert events during live detection" value={s.realtimeAlerts} onChange={v => update('realtimeAlerts', v)} />

      <div className="stg-tab-footer">
        <ActionBtn label="Reset Detection Settings" variant="danger" onClick={() => resetCategory(DETECTION_KEYS)} />
      </div>
    </div>
  )
}

const CAMERA_KEYS: (keyof Settings)[] = [
  'cameraSource', 'cctvStreamUrl', 'uploadQuality',
  'frameRate', 'playbackSpeed', 'autoStartDetection',
]

function CameraTab() {
  const { settings: s, update, resetCategory } = useSettings()
  return (
    <div className="stg-tab-body">
      <SectionTitle title="Camera Source" />
      <Select label="Camera Input" hint="Select the primary camera source for live detection"
        value={s.cameraSource}
        options={[
          { value: 'default', label: 'System Default Camera' },
          { value: 'external', label: 'External Webcam' },
          { value: 'cctv', label: 'CCTV / IP Stream' },
        ]}
        onChange={v => update('cameraSource', v)} />
      <TextInput
        label="CCTV Stream URL" hint="RTSP or HTTP stream endpoint"
        value={s.cctvStreamUrl} placeholder="rtsp://192.168.1.100:554/stream"
        onChange={v => update('cctvStreamUrl', v)} />

      <SectionTitle title="Video Processing" />
      <Select label="Upload Quality" hint="Compression level for uploaded video"
        value={s.uploadQuality}
        options={[
          { value: 'low', label: 'Low (Faster Upload)' },
          { value: 'medium', label: 'Medium (Balanced)' },
          { value: 'high', label: 'High (Best Quality)' },
        ]}
        onChange={v => update('uploadQuality', v)} />
      <Slider label="Frame Processing Rate" hint="Frames per second sent for inference"
        value={s.frameRate} min={1} max={30} step={1} unit=" fps"
        onChange={v => update('frameRate', v)} />
      <Slider label="Playback Speed" hint="Speed multiplier for uploaded video playback"
        value={s.playbackSpeed} min={0.25} max={4} step={0.25} unit="×"
        onChange={v => update('playbackSpeed', v)} />

      <SectionTitle title="Behavior" />
      <Toggle label="Auto-start Detection" hint="Begin inference immediately when feed loads"
        value={s.autoStartDetection} onChange={v => update('autoStartDetection', v)} />

      <div className="stg-tab-footer">
        <ActionBtn label="Reset Camera Settings" variant="danger" onClick={() => resetCategory(CAMERA_KEYS)} />
      </div>
    </div>
  )
}

const APPEARANCE_KEYS: (keyof Settings)[] = [
  'theme', 'accentColor', 'uiFontSize',
  'compactView', 'showSidebarLabels', 'enableAnimations',
]

const ACCENT_PRESETS = [
  { hex: '#ff4b00', name: 'Industrial Orange' },
  { hex: '#4d9cff', name: 'Blueprint Blue' },
  { hex: '#00e87a', name: 'Safety Green' },
  { hex: '#ff3b5c', name: 'Alert Red' },
  { hex: '#ffcc00', name: 'Hazard Yellow' },
  { hex: '#a855f7', name: 'Violet' },
]

function AppearanceTab() {
  const { settings: s, update, resetCategory } = useSettings()
  return (
    <div className="stg-tab-body">
      <SectionTitle title="Theme" />
      <Select label="Color Mode" hint="Dashboard color scheme"
        value={s.theme}
        options={[
          { value: 'dark', label: 'Dark Mode (Default)' },
          { value: 'light', label: 'Light Mode' },
        ]}
        onChange={v => update('theme', v)} />

      <div className="stg-color-picker">
        <div className="stg-toggle__text">
          <span>Accent Color</span>
          <span className="stg-hint">Primary UI highlight color</span>
        </div>
        <div className="stg-color-picker__swatches">
          {ACCENT_PRESETS.map(c => (
            <button key={c.hex}
              className={`stg-color-swatch ${s.accentColor === c.hex ? 'stg-color-swatch--active' : ''}`}
              style={{ background: c.hex }}
              title={c.name}
              onClick={() => update('accentColor', c.hex)}
            />
          ))}
        </div>
      </div>

      <SectionTitle title="Layout" />
      <Select label="Font Size" hint="Adjust dashboard text size"
        value={s.uiFontSize}
        options={[
          { value: 'small', label: 'Small' },
          { value: 'default', label: 'Default' },
          { value: 'large', label: 'Large' },
        ]}
        onChange={v => update('uiFontSize', v)} />
      <Toggle label="Compact View" hint="Reduce padding and spacing for denser layout"
        value={s.compactView} onChange={v => update('compactView', v)} />
      <Toggle label="Show Sidebar Labels" hint="Display text labels for nav items"
        value={s.showSidebarLabels} onChange={v => update('showSidebarLabels', v)} />
      <Toggle label="Enable Animations" hint="Scanlines, pulse effects, and transitions"
        value={s.enableAnimations} onChange={v => update('enableAnimations', v)} />

      <div className="stg-tab-footer">
        <ActionBtn label="Reset Appearance" variant="danger" onClick={() => resetCategory(APPEARANCE_KEYS)} />
      </div>
    </div>
  )
}

const NOTIFICATION_KEYS: (keyof Settings)[] = [
  'soundAlerts', 'criticalHighlight', 'emailNotifications',
  'smsNotifications', 'alertSensitivity', 'escalationMinutes',
]

function NotificationsTab() {
  const { settings: s, update, resetCategory } = useSettings()
  return (
    <div className="stg-tab-body">
      <SectionTitle title="Alert Delivery" />
      <Toggle label="Sound Alerts" hint="Play an audio tone on new detection events"
        value={s.soundAlerts} onChange={v => update('soundAlerts', v)} />
      <Toggle label="Critical Alert Highlighting" hint="Flash and emphasize critical-severity events"
        value={s.criticalHighlight} onChange={v => update('criticalHighlight', v)} />
      <Toggle label="Email Notifications" hint="Send email digests for escalated alerts"
        value={s.emailNotifications} onChange={v => update('emailNotifications', v)} />
      <Toggle label="SMS Notifications" hint="Send SMS for critical-severity events"
        value={s.smsNotifications} onChange={v => update('smsNotifications', v)} />

      <SectionTitle title="Sensitivity" />
      <Select label="Alert Sensitivity" hint="How aggressive the alert trigger should be"
        value={s.alertSensitivity}
        options={[
          { value: 'low', label: 'Low — Major violations only' },
          { value: 'medium', label: 'Medium — Balanced (Default)' },
          { value: 'high', label: 'High — All detections' },
        ]}
        onChange={v => update('alertSensitivity', v)} />
      <Slider label="Escalation Timeout" hint="Minutes before unacknowledged alerts escalate"
        value={s.escalationMinutes} min={1} max={60} step={1} unit=" min"
        onChange={v => update('escalationMinutes', v)} />

      <div className="stg-tab-footer">
        <ActionBtn label="Reset Notification Settings" variant="danger" onClick={() => resetCategory(NOTIFICATION_KEYS)} />
      </div>
    </div>
  )
}

const SYSTEM_KEYS: (keyof Settings)[] = [
  'processingMode', 'detectionQuality', 'memoryOptimization',
  'debugMode', 'selectedModel',
]

function SystemTab() {
  const { settings: s, update, resetCategory } = useSettings()
  const [cacheCleared, setCacheCleared] = useState(false)

  const clearCache = () => {
    // Clear any cached image data / detection results
    if ('caches' in window) {
      caches.keys().then(names => names.forEach(n => caches.delete(n)))
    }
    setCacheCleared(true)
    setTimeout(() => setCacheCleared(false), 3000)
  }

  return (
    <div className="stg-tab-body">
      <SectionTitle title="Processing" />
      <Select label="Processing Mode" hint="Hardware preference for inference"
        value={s.processingMode}
        options={[
          { value: 'auto', label: 'Auto — System decides (Default)' },
          { value: 'gpu', label: 'GPU — CUDA / WebGPU acceleration' },
          { value: 'cpu', label: 'CPU — Fallback mode' },
        ]}
        onChange={v => update('processingMode', v)} />
      <Select label="Detection Quality" hint="Balance between speed and accuracy"
        value={s.detectionQuality}
        options={[
          { value: 'fast', label: 'Fast — Lower resolution, higher FPS' },
          { value: 'balanced', label: 'Balanced (Default)' },
          { value: 'accurate', label: 'Accurate — Full resolution, lower FPS' },
        ]}
        onChange={v => update('detectionQuality', v)} />
      <Select label="Ensemble Model" hint="Model configuration for detection"
        value={s.selectedModel}
        options={[
          { value: 'ensemble-wbf', label: 'Ensemble WBF (YOLOv11 + YOLOv26)' },
          { value: 'yolov11-solo', label: 'YOLOv11 Solo' },
          { value: 'yolov26-solo', label: 'YOLOv26 Solo' },
          { value: 'yolov8-legacy', label: 'YOLOv8 Legacy' },
        ]}
        onChange={v => update('selectedModel', v)} />

      <SectionTitle title="Optimization" />
      <Toggle label="Memory Optimization" hint="Reduce memory usage — may lower throughput"
        value={s.memoryOptimization} onChange={v => update('memoryOptimization', v)} />
      <Toggle label="Debug Mode" hint="Show verbose logging in browser console"
        value={s.debugMode} onChange={v => update('debugMode', v)} />

      <SectionTitle title="Cache" />
      <div className="stg-cache-row">
        <ActionBtn
          label={cacheCleared ? '✓ CACHE CLEARED' : 'CLEAR CACHE'}
          onClick={clearCache}
        />
        <span className="stg-hint">Remove cached detection results and frame data</span>
      </div>

      <div className="stg-tab-footer">
        <ActionBtn label="Reset System Settings" variant="danger" onClick={() => resetCategory(SYSTEM_KEYS)} />
      </div>
    </div>
  )
}

const USER_KEYS: (keyof Settings)[] = [
  'profileName', 'language', 'dateFormat', 'timeFormat',
]

function UserTab() {
  const { settings: s, update, resetAll, resetCategory } = useSettings()
  return (
    <div className="stg-tab-body">
      <SectionTitle title="Profile" />
      <TextInput label="Display Name" hint="Your operational identity in the system"
        value={s.profileName} placeholder="Site Supervisor"
        onChange={v => update('profileName', v)} />

      <SectionTitle title="Regional" />
      <Select label="Language" hint="Interface language"
        value={s.language}
        options={[
          { value: 'en', label: 'English' },
          { value: 'hi', label: 'Hindi' },
          { value: 'ta', label: 'Tamil' },
          { value: 'es', label: 'Spanish' },
          { value: 'fr', label: 'French' },
          { value: 'zh', label: 'Chinese (Simplified)' },
        ]}
        onChange={v => update('language', v)} />
      <Select label="Date Format" hint="How dates are rendered across the dashboard"
        value={s.dateFormat}
        options={[
          { value: 'DD/MM/YYYY', label: 'DD/MM/YYYY' },
          { value: 'MM/DD/YYYY', label: 'MM/DD/YYYY' },
          { value: 'YYYY-MM-DD', label: 'YYYY-MM-DD (ISO)' },
        ]}
        onChange={v => update('dateFormat', v)} />
      <Select label="Time Format" value={s.timeFormat}
        options={[
          { value: '24h', label: '24-hour (14:30)' },
          { value: '12h', label: '12-hour (2:30 PM)' },
        ]}
        onChange={v => update('timeFormat', v)} />

      <SectionTitle title="Layout" />
      <div className="stg-info-card">
        <span className="stg-hint">
          Your current dashboard layout and all settings are automatically saved to this browser.
          Use the button below to reset everything to factory defaults.
        </span>
      </div>

      <div className="stg-tab-footer">
        <ActionBtn label="Reset User Preferences" variant="danger" onClick={() => resetCategory(USER_KEYS)} />
        <ActionBtn label="Reset All Settings to Default" variant="danger" onClick={() => {
          if (window.confirm('Reset ALL settings to factory defaults? This cannot be undone.')) resetAll()
        }} />
      </div>
    </div>
  )
}

/* ── Tab content dispatcher ───────────────────────────────────────────────── */

const TAB_PANELS: Record<TabKey, () => React.JSX.Element> = {
  detection: DetectionTab,
  camera: CameraTab,
  appearance: AppearanceTab,
  notifications: NotificationsTab,
  system: SystemTab,
  user: UserTab,
}

/* ── Main export ──────────────────────────────────────────────────────────── */

export function SettingsPanel() {
  const [tab, setTab] = useState<TabKey>('detection')
  const ActivePanel = TAB_PANELS[tab]

  return (
    <div className="stg-panel panel">
      <div className="stg-layout">
        <nav className="stg-nav" aria-label="Settings categories">
          {TABS.map(t => (
            <button key={t.key}
              className={`stg-nav__item ${tab === t.key ? 'stg-nav__item--active' : ''}`}
              onClick={() => setTab(t.key)}
            >
              {t.label}
            </button>
          ))}
        </nav>

        <div className="stg-content">
          <ActivePanel />
        </div>
      </div>
    </div>
  )
}
