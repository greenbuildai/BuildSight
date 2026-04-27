# Jovi — Handoff from Toni
**Date:** 2026-04-11  
**From:** Toni (Claude Sonnet 4.6)  
**Session context:** BuildSight PPE detection pipeline — post auto-scene-mode FPS/overlap fix

---

## SITUATION SUMMARY

Toni hit context limits mid-session. The previous session completed:

- ✅ Auto scene classification with hysteresis (`SceneConditionTracker`, `classify_scene_fast`)
- ✅ Scene caching (`SCENE_CACHE_FRAMES = 12`)
- ✅ S4 WBF worker IoU fixed (0.72 → 0.55, eliminates same-person duplicate boxes)
- ✅ `_final_worker_nms(iou_thresh=0.45)` — post-merge hard NMS
- ✅ `_merge_full_and_tile` dedup IoU raised (0.35 → 0.45)
- ✅ Conditional tile inference (only when full-frame < 3 workers)
- ✅ S4 single-model gate raised (0.35 → 0.40)
- ✅ S4 post-WBF worker gate raised (0.15 → 0.20)
- ✅ S4 portrait check tightened (aspect > 1.60, score < 0.62)
- ✅ `_soft_nms` class in `cross_class_nms` (worker IoU 0.45→0.38, centroid-distance guard)
- ✅ Recovery IoU gate raised (0.55 → 0.65)
- ✅ Recovery score gates tightened
- ✅ Per-region recovery cooldown (`RECOVERY_COOLDOWN_FRAMES = 4`)
- ✅ Ghost suppression (hits < 2 before emitting)
- ✅ Static motion variance tightened (3.5 → 3.0 px²)
- ✅ Gemini rate-limiter (`GEMINI_RATE_LIMIT_PER_N = 10`)
- ✅ Frontend: `SceneAutoIndicator`, `CONDITION_COLORS`, 10-frame rolling FPS
- ✅ Frontend: throttled DOM updates (1 Hz React state flush)

The user then submitted a new RACE prompt (shown in full at end of this file). Toni acknowledged it, set up the todo list, created the backup folder, and reached context limits.

---

## YOUR IMMEDIATE TASK — Complete the Pending RACE Prompt

**Work in this exact order:**

---

### Step 1 — Create Backup (FIRST — do not skip)

Create folder:
```
dashboard/backend/backups/2026-04-11_scene_classification_fp_fix/
```

Copy these files exactly as they currently exist:
```
dashboard/backend/server.py
scripts/site_aware_ensemble.py
scripts/adaptive_postprocess.py
scripts/ensemble_video.py
scripts/gemini_auditor.py
dashboard/src/components/DetectionPanel.tsx
dashboard/src/components/DetectionPanel.css
```

Write `BACKUP_NOTES.md` documenting:
- Current auto scene classification logic and thresholds
- Current S1/S2/S3/S4 routing rules
- Current NMS settings after all previous fixes
- Current FPS performance achieved (S1: ~25 FPS, S4: ~15 FPS after caching + conditional tiling)
- All FPS optimizations already in place (scene caching, conditional tiling, Gemini rate-limit, 1 Hz React flush)
- Known remaining issues: floating PPE FPs, clutter still in certain zones, no trigger reason visible in UI

---

### Step 2 — `classify_scene_fast` → returns `(condition, reason)` tuple

**File:** `dashboard/backend/server.py`

Change `classify_scene_fast()` signature from returning `str` to returning `Tuple[str, str]`:

```python
def classify_scene_fast(img_bgr, model, device, half) -> tuple[str, str]:
    # ... existing logic unchanged ...
    
    # Change every return to return a (condition, reason) tuple:
    if brightness < AUTO_LOW_LIGHT_THRESH:
        return "S3_low_light", f"brightness {brightness:.0f}"
    if contrast < AUTO_DUSTY_STD_THRESH and saturation < AUTO_DUSTY_SAT_THRESH:
        return "S2_dusty", f"haze σ={contrast:.0f}"
    
    # After model pass for worker count:
    if n_workers >= AUTO_CROWD_WORKER_THRESH:
        return "S4_crowded", f"{n_workers} workers"
    if n_workers >= 3 and (cluster overlap check):
        return "S4_crowded", "3+ clustered"
    
    return "S1_normal", "normal"
```

**Also add `_last_reason: str = "normal"` to `SceneConditionTracker.__init__` and `reset()`.**

**In the auto-mode caching block inside `run_inference`** (around line 1365), update all places that call `classify_scene_fast` to unpack the tuple:
```python
# Image-stats-only paths (already handled inline before calling classify_scene_fast)
# For S3: reason = f"brightness {brightness_quick:.0f}"
# For S2: reason = f"haze σ={contrast_quick:.0f}"
# For model-pass path:
raw_cond, reason = classify_scene_fast(img_bgr, model_v11, DEVICE, USE_HALF)
_scene_tracker._last_raw_cond = raw_cond
_scene_tracker._last_reason = reason
# For cached path:
reason = _scene_tracker._last_reason
```

Store `reason` in `_scene_tracker._last_reason` so the endpoint can read it without re-running classification.

---

### Step 3 — `SceneConditionTracker` Transition Logging

**File:** `dashboard/backend/server.py`

Add to `SceneConditionTracker.__init__`:
```python
self._transitions: list[dict] = []   # last 10 mode transitions
self._last_reason: str = "normal"
```

Add to `SceneConditionTracker.update()` — after `self.current` is set, detect a change:
```python
# At the end of update(), after assigning self.current:
if self.current != prev_condition:   # prev_condition = self.current captured before changes
    self._transitions.append({
        "to": self.current,
        "reason": self._last_reason,
        "frame": self._cache_frame_count,
    })
    if len(self._transitions) > 10:
        self._transitions.pop(0)
```

Add to `reset()`: `self._transitions = []`

Add method:
```python
def recent_transitions(self, n: int = 5) -> list[dict]:
    return list(self._transitions[-n:])
```

---

### Step 4 — `StaticClutterMask` class

**File:** `dashboard/backend/server.py`

Add before `_final_worker_nms`:

```python
class StaticClutterMask:
    """
    Identifies spatial regions of the frame that consistently contain static
    construction materials (no motion + material-like color).
    
    Built from the first BUILD_FRAMES frames by comparing frame diffs.
    Updated every UPDATE_INTERVAL_S seconds.
    Applied as a confidence penalty: detections with centroid in a high-clutter
    region get score penalised unless their score is already very high.
    
    Query cost: O(1) — single pixel lookup into a pre-built mask.
    """
    BUILD_FRAMES = 8
    UPDATE_INTERVAL_S = 60.0
    PENALTY = 0.18          # score reduction for detections in clutter zones
    PENALTY_GATE = 0.70     # don't penalise boxes already confident enough

    def __init__(self):
        self._frames: list[np.ndarray] = []
        self._mask: np.ndarray | None = None   # uint8 H×W, 255=clutter 0=clear
        self._last_update: float = 0.0
        self._built: bool = False

    def feed(self, img_bgr: np.ndarray) -> None:
        """Call once per inference frame while building the mask."""
        if self._built:
            return
        small = cv2.resize(img_bgr, (160, 90))
        self._frames.append(small)
        if len(self._frames) >= self.BUILD_FRAMES:
            self._build()

    def _build(self) -> None:
        frames = self._frames[:self.BUILD_FRAMES]
        # Motion mask: std across frames per pixel
        stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
        motion_std = np.std(stack, axis=0).mean(axis=2)   # H×W
        static_mask = (motion_std < 8.0).astype(np.uint8) * 255

        # Color mask: brick red, cement grey, yellow tarpaulin
        combined_color = np.zeros(frames[0].shape[:2], dtype=np.uint8)
        for f in frames:
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            # Brick red: H 0–15 or 165–180, S>60, V>60
            red1 = cv2.inRange(hsv, (0, 60, 60), (15, 255, 255))
            red2 = cv2.inRange(hsv, (165, 60, 60), (180, 255, 255))
            # Cement grey: S<40, V 60–180
            grey = cv2.inRange(hsv, (0, 0, 60), (180, 40, 180))
            # Yellow tarpaulin: H 20–35, S>80
            yellow = cv2.inRange(hsv, (20, 80, 80), (35, 255, 255))
            combined_color = cv2.bitwise_or(
                combined_color,
                cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(grey, yellow))
            )

        # Clutter = static AND material-colored
        clutter = cv2.bitwise_and(static_mask, combined_color)
        # Dilate slightly to cover full box regions
        clutter = cv2.dilate(clutter, np.ones((5, 5), np.uint8), iterations=2)
        self._mask = clutter
        self._built = True
        self._last_update = time.time()
        self._frames.clear()

    def should_rebuild(self) -> bool:
        return self._built and (time.time() - self._last_update) > self.UPDATE_INTERVAL_S

    def reset(self) -> None:
        self.__init__()

    def query_box(self, box: list, img_w: int, img_h: int) -> float:
        """
        Returns clutter fraction [0,1] for the given box.
        Scales box to the 160×90 mask resolution.
        Returns 0.0 if mask not built yet.
        """
        if self._mask is None:
            return 0.0
        mh, mw = self._mask.shape
        sx = mw / max(img_w, 1)
        sy = mh / max(img_h, 1)
        x1 = max(0, int(box[0] * sx))
        y1 = max(0, int(box[1] * sy))
        x2 = min(mw, int(box[2] * sx) + 1)
        y2 = min(mh, int(box[3] * sy) + 1)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        region = self._mask[y1:y2, x1:x2]
        return float(np.mean(region > 0))


_static_clutter_mask = StaticClutterMask()
```

**Wire into `run_inference`** after the existing `is_valid_worker` filter:
```python
# ── Static clutter mask confidence penalty ────────────────────────────────
# Feed the current frame to the mask builder during the warm-up period.
# After the mask is built, apply a confidence penalty to worker detections
# whose centroid lands in a high-clutter zone.
_static_clutter_mask.feed(img_bgr)
if _static_clutter_mask._built:
    penalised = []
    for det in raw:
        if det["cls"] in worker_cls_ids:
            cf = _static_clutter_mask.query_box(det["box"], w, h)
            if cf > 0.55 and det["score"] < StaticClutterMask.PENALTY_GATE:
                det = dict(det)  # shallow copy before mutating
                det["score"] = max(0.0, det["score"] - StaticClutterMask.PENALTY)
        penalised.append(det)
    raw = penalised
    # Re-apply is_valid_worker after penalty so penalised boxes below post-gate are dropped
    raw = [
        d for d in raw
        if d["cls"] not in worker_cls_ids
        or is_valid_worker(d["box"], w, h, d["score"],
                           model_count=d.get("n_models", 1),
                           condition=condition,
                           nearby_ppe=_has_nearby_ppe(d["box"], raw))
    ]
```

**Also reset on `reset_tracker`** — in the `/api/detect/frame` endpoint where `_scene_tracker.reset()` is called, also call `_static_clutter_mask.reset()`.

---

### Step 5 — Replace Hard NMS with Soft-NMS

**File:** `dashboard/backend/server.py`

Replace `_final_worker_nms` entirely:

```python
def _final_worker_nms(raw: list, sigma: float = 0.5, score_thresh: float = 0.18) -> list:
    """
    Gaussian Soft-NMS applied to worker boxes after all merging stages.
    
    Instead of hard-suppressing overlapping boxes, decays their scores by
    a Gaussian function of IoU overlap with the highest-scoring kept box.
    This is more graceful than hard NMS: genuinely close-but-separate workers
    survive with slightly lower scores rather than being cut entirely.
    
    sigma: controls decay speed — lower = more aggressive suppression.
    score_thresh: boxes whose score decays below this are dropped.
    """
    worker_ids = _worker_ids()
    workers = sorted(
        [d for d in raw if d["cls"] in worker_ids],
        key=lambda x: -x["score"],
    )
    others = [d for d in raw if d["cls"] not in worker_ids]

    import math
    scores = [w["score"] for w in workers]
    n = len(workers)
    kept_flags = [True] * n

    for i in range(n):
        if not kept_flags[i]:
            continue
        for j in range(i + 1, n):
            if not kept_flags[j]:
                continue
            ov = iou_box(workers[i]["box"], workers[j]["box"])
            if ov > 0:
                # Gaussian decay
                decay = math.exp(-(ov * ov) / sigma)
                scores[j] *= decay
                if scores[j] < score_thresh:
                    kept_flags[j] = False

    kept = []
    for i, w in enumerate(workers):
        if kept_flags[i]:
            w = dict(w)
            w["score"] = scores[i]
            kept.append(w)

    return others + kept
```

---

### Step 6 — `suppress_floating_ppe` function

**File:** `dashboard/backend/server.py`

Add after `_final_worker_nms`:

```python
def suppress_floating_ppe(raw: list) -> list:
    """
    Remove helmet and safety-vest detections that have no co-located worker
    body box within a reasonable proximity.
    
    Floating PPE (helmet/vest without a nearby worker box) is almost always:
      - A hat or fluorescent sign on a material pile
      - A vest-coloured tarpaulin or bag
      - A stray detection on the background
    
    High-confidence PPE (score ≥ 0.82) is always kept — a clear helmet detection
    is strong evidence of a worker even if the body is occluded.
    """
    worker_ids = _worker_ids()
    helmet_ids = _helmet_ids()
    vest_ids = _vest_ids()

    workers = [d for d in raw if d["cls"] in worker_ids]
    result = []

    for det in raw:
        if det["cls"] not in helmet_ids and det["cls"] not in vest_ids:
            result.append(det)
            continue

        # High-confidence PPE: keep unconditionally
        if det.get("score", 0) >= 0.82:
            result.append(det)
            continue

        # Check if any worker box contains or is near this PPE centroid
        cx = (det["box"][0] + det["box"][2]) / 2.0
        cy = (det["box"][1] + det["box"][3]) / 2.0
        anchored = False
        for w in workers:
            wx1, wy1, wx2, wy2 = w["box"]
            pad_x = (wx2 - wx1) * 0.35
            pad_y = (wy2 - wy1) * 0.35
            if (wx1 - pad_x <= cx <= wx2 + pad_x and
                    wy1 - pad_y <= cy <= wy2 + pad_y):
                anchored = True
                break
        if anchored:
            result.append(det)
        # else: floating PPE suppressed

    return result
```

**Wire into `run_inference`** after `_final_worker_nms` and before `apply_gemini_audit`:
```python
raw = _final_worker_nms(raw, sigma=0.5, score_thresh=0.18)
raw = suppress_floating_ppe(raw)   # ← ADD THIS LINE
_gemini_enabled = gemini_audit and _should_run_gemini()
raw, audit_meta = apply_gemini_audit(...)
```

---

### Step 7 — Update `run_inference` to return 5-tuple

**File:** `dashboard/backend/server.py`

Change the last return line:
```python
# WAS:
return raw, used_ensemble, audit_meta, condition

# NOW:
reason = _scene_tracker._last_reason
return raw, used_ensemble, audit_meta, condition, reason
```

**Update all 3 callers** (search for `= await run_in_threadpool(run_inference` and `= run_inference(`):

1. `/api/detect/image` endpoint:
```python
detections, used_ensemble, audit_meta, resolved_condition, scene_reason = await run_in_threadpool(run_inference, ...)
```

2. `/api/detect/frame` endpoint:
```python
detections, used_ensemble, audit_meta, resolved_condition, scene_reason = await run_in_threadpool(run_inference, ...)
# Add to return dict:
return {
    ...
    "condition": resolved_condition,
    "scene_reason": scene_reason,           # ← ADD
    "mode_transitions": _scene_tracker.recent_transitions(5),  # ← ADD
    ...
}
```

3. Video batch loop:
```python
raw, used, _, _cond, _reason = run_inference(...)
```

---

### Step 8 — Recovery Every 3rd Frame

**File:** `scripts/site_aware_ensemble.py`

Add at module level (near the other constants):
```python
RECOVERY_EVERY_N_FRAMES = 3   # run recovery pass only on every Nth inference cycle
_recovery_schedule_counter: int = 0
```

In `wbf_fuse_condition`, replace:
```python
if condition == "S4_crowded":
    advance_recovery_frame()
    fused = recover_crowded_workers(fused, all_preds, profile["recover_worker"])
```

With:
```python
if condition == "S4_crowded":
    global _recovery_schedule_counter
    _recovery_schedule_counter += 1
    advance_recovery_frame()
    if (_recovery_schedule_counter % RECOVERY_EVERY_N_FRAMES) == 1:
        fused = recover_crowded_workers(fused, all_preds, profile["recover_worker"])
```

---

### Step 9 — Frontend Updates

**File:** `dashboard/src/components/DetectionPanel.tsx`

#### 9a — Add `sceneTrigger` and `modeLog` states

In `VideoUploadMode`, after the existing `detectedCondition` state:
```typescript
const [sceneTrigger, setSceneTrigger] = useState<string>('')
const [modeLog, setModeLog] = useState<Array<{frame: number; to: string; reason: string}>>([])
const [showModeLog, setShowModeLog] = useState(false)
```

Add pending refs for throttling:
```typescript
const pendingTriggerRef = useRef<string>('')
const pendingModeLogRef = useRef<Array<{frame: number; to: string; reason: string}>>([])
```

#### 9b — Consume `data.scene_reason` and `data.mode_transitions` in inference loop

In the inference response handler, inside the `if (now - lastLabelUpdateRef.current >= 1000)` block, add:
```typescript
if (data.scene_reason) pendingTriggerRef.current = data.scene_reason
if (data.mode_transitions?.length) pendingModeLogRef.current = data.mode_transitions

// Inside the 1Hz flush:
setSceneTrigger(pendingTriggerRef.current)
if (pendingModeLogRef.current.length) setModeLog([...pendingModeLogRef.current])
```

#### 9c — Update `SceneAutoIndicator` to show trigger reason

Change the badge section in `SceneAutoIndicator` to show `trigger` as a subtitle:

```tsx
function SceneAutoIndicator({
  condition, autoMode, trigger, onToggleAuto, onManualChange,
}: {
  condition: string
  autoMode: boolean
  trigger?: string
  onToggleAuto: () => void
  onManualChange: (c: Condition) => void
}) {
  // ... existing code ...
  return (
    <div className="det-scene-indicator">
      <div className="det-scene-indicator__header">...</div>
      {autoMode ? (
        <div>
          <div className="det-scene-indicator__badge" style={{ borderColor: color, color }}>
            <span className="det-scene-indicator__dot" style={{ background: color }} />
            {label}
          </div>
          {trigger && (
            <div className="det-scene-indicator__trigger">{trigger.toUpperCase()}</div>
          )}
        </div>
      ) : (
        <ConditionPicker value={condition as Condition} onChange={onManualChange} />
      )}
    </div>
  )
}
```

Pass `trigger={sceneTrigger}` when calling `SceneAutoIndicator` in `VideoUploadMode`.

#### 9d — Add collapsible `ModeTransitionLog` component

```tsx
function ModeTransitionLog({
  transitions, visible, onToggle,
}: {
  transitions: Array<{frame: number; to: string; reason: string}>
  visible: boolean
  onToggle: () => void
}) {
  if (!transitions.length) return null
  return (
    <div className="det-mode-log">
      <button className="det-mode-log__toggle" onClick={onToggle}>
        MODE LOG {visible ? '▲' : '▼'}
      </button>
      {visible && (
        <div className="det-mode-log__entries">
          {[...transitions].reverse().map((t, i) => (
            <div key={i} className="det-mode-log__entry">
              <span className="det-mode-log__entry-to"
                    style={{ color: CONDITION_COLORS[t.to] ?? '#aaa' }}>
                {t.to.replace(/_/g, ' ').toUpperCase()}
              </span>
              <span className="det-mode-log__entry-reason">{t.reason}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

Render it inside `renderControls` below `SceneAutoIndicator`:
```tsx
<ModeTransitionLog
  transitions={modeLog}
  visible={showModeLog}
  onToggle={() => setShowModeLog(v => !v)}
/>
```

#### 9e — CSS additions

Add to `DetectionPanel.css`:
```css
.det-scene-indicator__trigger {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--color-text-muted);
  letter-spacing: 0.06em;
  margin-top: 0.25rem;
  padding-left: 0.2rem;
}

.det-mode-log {
  margin-top: 0.5rem;
}

.det-mode-log__toggle {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--color-text-muted);
  background: transparent;
  border: 1px solid var(--color-border);
  padding: 0.2rem 0.5rem;
  cursor: pointer;
  width: 100%;
  text-align: left;
  letter-spacing: 0.08em;
}

.det-mode-log__entries {
  border: 1px solid var(--color-border);
  border-top: none;
  max-height: 120px;
  overflow-y: auto;
}

.det-mode-log__entry {
  display: flex;
  gap: 0.5rem;
  padding: 0.2rem 0.4rem;
  font-family: var(--font-mono);
  font-size: 0.62rem;
  border-bottom: 1px solid var(--color-border);
}

.det-mode-log__entry:last-child { border-bottom: none; }

.det-mode-log__entry-to {
  font-weight: 700;
  flex-shrink: 0;
}

.det-mode-log__entry-reason {
  color: var(--color-text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
```

---

### Step 10 — TypeScript Check

After all frontend changes, run:
```bash
cd dashboard && npx tsc --noEmit
```

Fix any type errors before finalising.

---

## FILE MAP — What to Touch

| File | Steps |
|------|-------|
| `dashboard/backend/server.py` | 2, 3, 4, 5, 6, 7 |
| `scripts/site_aware_ensemble.py` | 8 |
| `dashboard/src/components/DetectionPanel.tsx` | 9a–9d |
| `dashboard/src/components/DetectionPanel.css` | 9e |

---

## CRITICAL CONSTRAINTS (do not violate)

1. **Never break FPS targets**: S1 ≥ 25 FPS, S4 ≥ 15 FPS, latency < 300ms
2. `StaticClutterMask.feed()` and `query_box()` must be < 1ms each
3. `suppress_floating_ppe` is O(n_ppe × n_workers) — both are small (< 20), fine
4. `_final_worker_nms` Soft-NMS is O(n²) over workers only — fine for < 20 workers
5. All new React state uses existing 1 Hz throttle — no new per-frame `setState` calls
6. The 5-tuple return from `run_inference` must be updated in **all 3 callers** or the server will crash on startup
7. `_static_clutter_mask` is module-level; `reset()` must be called alongside `_scene_tracker.reset()` in the `/api/detect/frame` endpoint handler when `reset_tracker=1`

---

## WHAT NOT TO DO

- Do NOT re-implement features already done (listed in "Situation Summary" above)
- Do NOT change `SCENE_CACHE_FRAMES`, `RECOVERY_COOLDOWN_FRAMES`, or the hysteresis constants — they are tuned
- Do NOT add optical flow, HOG, or pose estimation — too slow for real-time
- Do NOT change the `wbf_fuse` or `wbf_fuse_condition` WBF logic — it was already fixed
- Do NOT add new Python threads or async tasks — FastAPI threadpool handles this
- Do NOT touch the frontend canvas RAF loop or `drawOverlay` — already decoupled

---

## ORIGINAL RACE PROMPT (for full context)

The user submitted this prompt with two construction site CCTV screenshots:

**Problem 1:** Auto scene classification not accurate enough — needs deterministic thresholds, must never flicker, S4 must trigger on 4+ workers or 3+ grouped workers.

**Problem 2:** Crowded-scene logic too aggressive — FPs for cement bags, buckets, scaffolding, planks, bricks, material piles. Multiple overlapping boxes per worker (e.g., worker 31%, worker 48%, worker 50% stacked on one person). Floating helmet/vest on clutter. Large FP clusters during S4.

Full deliverable list from the RACE prompt:
- ✅ Backup + BACKUP_NOTES.md
- Static clutter mask (pre-computed, pixel lookup)
- `classify_scene_fast` returning (condition, reason)
- Soft-NMS replacing hard NMS
- `suppress_floating_ppe`
- Recovery every 3rd frame
- Scene trigger reason in SceneAutoIndicator badge
- Collapsible mode transition log in sidebar
- `scene_reason` and `mode_transitions` in `/api/detect/frame` response
- TypeScript clean (zero errors)
- No FPS regression from current baselines

---

*Handoff written by Toni — 2026-04-11*
