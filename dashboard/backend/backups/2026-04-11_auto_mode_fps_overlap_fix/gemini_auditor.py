"""
gemini_auditor.py
=================
BuildSight AI — Gemini 2.5 Flash post-processing auditor.

Two roles:
  1. VALIDATE  — reject false-positive detections from GroundingDINO
  2. SUPPLEMENT — add missed helmets / vests that DINO scored below threshold

Uses google-genai SDK v1.68.0 (new client-based API).
"""

import io
import cv2
import json
import time
import numpy as np
from PIL import Image

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_CLASS_NAMES = {0: "helmet", 1: "safety_vest", 2: "worker"}

def _iou(a, b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _draw_annotated(img_bgr, dets):
    """
    Draw detection boxes onto a copy of img_bgr.
    Workers = blue,  helmets = yellow,  vests = green.
    Each box gets a short ID label (W0, H0, V0 …).
    Returns (annotated_image, id_to_index_map).
    """
    out = img_bgr.copy()
    colours = {2: (255, 80, 0), 0: (0, 220, 220), 1: (0, 200, 0)}
    counts  = {2: 0, 0: 0, 1: 0}
    prefixes = {2: "W", 0: "H", 1: "V"}
    id_map   = {}   # "W0" -> index in dets

    for idx, det in enumerate(dets):
        cid  = det["class_id"]
        if cid not in counts:
            continue
        uid  = f"{prefixes[cid]}{counts[cid]}"
        counts[cid] += 1
        id_map[uid] = idx

        x1, y1, x2, y2 = map(int, det["xyxy"])
        col = colours[cid]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, uid, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

    return out, id_map


def _pil_to_bytes(pil_img, quality=85):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# GeminiAuditor
# ──────────────────────────────────────────────────────────────────────────────

class GeminiAuditor:
    """
    Post-processing auditor using Gemini 2.5 Flash.

    Call:
        new_dets, kept_dets = auditor.audit(img_bgr, all_dets, condition)

    Parameters
    ----------
    img_bgr   : np.ndarray  — full-resolution BGR image (original, NOT preprocessed)
    all_dets  : list[dict]  — list of dicts with keys: class_id, class_name, xyxy, score
    condition : str         — "normal" | "low_light" | "dusty" | "crowded"

    Returns
    -------
    new_dets  : list[dict]  — extra detections from Gemini (source = "gemini")
    kept_dets : list[dict]  — original dets minus Gemini-rejected FPs
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        if not _GENAI_AVAILABLE:
            raise ImportError("google-genai is not installed. Run: pip install google-genai>=1.68.0")
        self.client     = genai.Client(api_key=api_key)
        self.model_name = model_name

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    # Confidence band for ambiguous detections that trigger a Gemini call.
    # Detections outside [AMBIGUOUS_LOW, AMBIGUOUS_HIGH] are passed through
    # without a Gemini call: high-confidence dets are kept, very-low-conf are
    # already filtered by post-gates before reaching the auditor.
    AMBIGUOUS_LOW: float = 0.30
    AMBIGUOUS_HIGH: float = 0.68

    def audit(self, img_bgr: np.ndarray, all_dets: list, condition: str = "normal",
              supplement: bool = False):
        """
        Validator-only Gemini audit.  Gemini rejects false positives and,
        optionally, adds missed PPE (supplement=True).  Default is False —
        Gemini should only reject, never create new boxes, to avoid phantom
        worker/PPE detections on clutter.

        Only detections within the ambiguous confidence band
        [AMBIGUOUS_LOW, AMBIGUOUS_HIGH] are sent to Gemini; high-confidence
        detections are always kept, sub-threshold detections are passed through
        unchanged (they should have been removed by earlier post-gates).

        Returns (new_dets, kept_dets).
          new_dets  — [] unless supplement=True.
          kept_dets — all_dets minus Gemini-rejected false positives.
        """
        if not all_dets:
            return [], []

        # Confident detections pass through without a Gemini call.
        confident_dets = [d for d in all_dets if d["score"] > self.AMBIGUOUS_HIGH]
        # Ambiguous detections are sent to Gemini for validation.
        ambiguous_dets = [d for d in all_dets
                          if self.AMBIGUOUS_LOW <= d["score"] <= self.AMBIGUOUS_HIGH]
        # Very-low-confidence detections are passed through unchanged —
        # they should have been removed by upstream post-gates already.
        low_dets = [d for d in all_dets if d["score"] < self.AMBIGUOUS_LOW]

        if not ambiguous_dets:
            # Nothing to review — skip Gemini entirely, preserve all detections.
            return [], list(all_dets)

        h_img, w_img = img_bgr.shape[:2]

        # Draw annotated overlay for Gemini (only the ambiguous subset)
        annotated, id_map = _draw_annotated(img_bgr, ambiguous_dets)

        # Build prompt (validator role — no supplement instructions by default)
        prompt = self._build_prompt(ambiguous_dets, id_map, condition, w_img, h_img)

        # Call Gemini
        response_text = self._call_gemini(annotated, prompt)
        if response_text is None:
            # On API failure: keep everything, add nothing.
            return [], list(all_dets)

        # Parse response — Gemini validates/rejects the ambiguous subset
        gemini_new, ambiguous_kept = self._parse_response(
            response_text, ambiguous_dets, id_map, w_img, h_img)

        # Rebuild the full detection list:
        #   confident (always kept) + Gemini-validated ambiguous + low (pass-through)
        kept_dets = confident_dets + ambiguous_kept + low_dets

        # Discard Gemini-added boxes unless supplement mode is explicitly enabled.
        new_dets = gemini_new if supplement else []

        return new_dets, kept_dets

    # Backwards-compatible shim for old callers that pass boxes/labels/scores separately
    def audit_image(self, img_bgr, boxes, labels, scores):
        """
        Legacy interface used by the old placeholder in annotate_indian_dataset.py.
        Returns list[bool] approvals (True = keep).
        """
        # Build det dicts from parallel lists
        name_to_id = {"helmet": 0, "safety_vest": 1, "worker": 2}
        dets = []
        for box, label, score in zip(boxes, labels, scores):
            cid = name_to_id.get(label, -1)
            dets.append({"class_id": cid, "class_name": label, "xyxy": box, "score": score})

        _, kept = self.audit(img_bgr, dets)
        kept_set = {id(d) for d in kept}
        return [id(d) in kept_set for d in dets]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, dets, id_map, condition, w_img, h_img):
        # Reverse id_map: index -> uid
        idx_to_uid = {v: k for k, v in id_map.items()}

        lines: list[str] = []
        for idx, det in enumerate(dets):
            uid  = idx_to_uid.get(idx, f"X{idx}")
            x1, y1, x2, y2 = [round(v) for v in det["xyxy"]]
            lines.append(f"  {uid}: {det['class_name']}  box=[{x1},{y1},{x2},{y2}]  conf={det['score']:.2f}")

        det_list = "\n".join(lines) if lines else "  (none)"

        condition_note = ""
        if condition in ("low_light", "low_light_condition"):
            condition_note = (
                "NOTE: This image was captured at night or in very low light. "
                "Helmets and vests may appear darker than usual — look for shape and reflective strips, not just colour."
            )
        elif condition in ("dusty", "dusty_condition"):
            condition_note = (
                "NOTE: This is a dusty construction site. Workers and PPE may be partially obscured by dust haze."
            )

        return f"""You are a construction-site PPE safety auditor analysing a real image from an Indian construction site.

Image dimensions: {w_img}×{h_img} pixels.
{condition_note}

CURRENT DETECTIONS (drawn on the image in colour):
  Blue boxes  = Workers (W0, W1, …)
  Yellow boxes = Helmets (H0, H1, …)
  Green boxes  = Safety vests (V0, V1, …)

{det_list}

YOUR TASK — VALIDATION ONLY:
Reject false-positive detections. These are medium-confidence boxes that the
YOLO ensemble flagged but may actually be site clutter.

For each labelled box decide if it is a TRUE detection:
  - A Worker box is TRUE only if it contains a visible human being.
    REJECT if it is: a cement bag, sand bag, blue bucket, scaffolding pole,
    machinery part, tarpaulin, material pile, shadow, sign, or static object.
  - A Helmet box is TRUE only if it is a hard hat / safety helmet worn on a
    human head. REJECT if it is a round object, lamp, ball, or pipe fitting.
  - A Vest box is TRUE only if it is a reflective hi-vis safety vest worn on a
    human torso. REJECT if it is blue sheeting, tarpaulin, coloured sack, or
    any item not worn by a person.

NOTE: When in doubt, REJECT the detection. It is better to miss one marginal
detection than to count scaffolding or a cement bag as a worker.

SUPPLEMENTS: Leave "supplements" as an empty list []. Do not add new boxes.

RESPONSE FORMAT — strict JSON, no markdown:
{{
  "validations": {{
    "<id>": true | false,   // true = correct detection, false = false positive
    ...
  }},
  "supplements": [
    {{
      "worker_id": "<W0>",         // which worker this PPE belongs to
      "class": "helmet" | "safety_vest",
      "box": [x1, y1, x2, y2]     // pixel coords in the original image
    }},
    ...
  ]
}}

IMPORTANT:
- Use integer pixel coordinates (not normalised floats).
- Only include IDs that appear in the CURRENT DETECTIONS list above.
- If no supplements needed, use an empty list [].
- Respond with ONLY the JSON object — no explanation, no markdown fences.
"""

    def _call_gemini(self, annotated_bgr, prompt, max_retries=3):
        """Send image + prompt to Gemini, return raw response text or None."""
        # If daily quota was already exhausted this session, skip immediately
        if getattr(self, "_daily_quota_exhausted", False):
            return None

        img_rgb  = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(img_rgb)
        img_bytes = _pil_to_bytes(pil_img)
        img_part  = genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, img_part],
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                return response.text
            except Exception as exc:
                exc_str = str(exc)
                # Detect daily quota exhaustion — no point retrying until tomorrow
                if "PerDay" in exc_str or ("limit: 20" in exc_str and "FreeTier" in exc_str):
                    print("  [GeminiAuditor] Daily free-tier quota (20 req/day) exhausted — "
                          "disabling Gemini for this session. DINO detections will be kept as-is.")
                    self._daily_quota_exhausted = True
                    return None
                wait = 5 * (attempt + 1)
                if attempt < max_retries - 1:
                    print(f"  [GeminiAuditor] API error (attempt {attempt+1}): {exc} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"  [GeminiAuditor] All retries exhausted: {exc}")
                    return None

    def _parse_response(self, text, all_dets, id_map, w_img, h_img):
        """
        Parse Gemini JSON response.
        Returns (new_dets, kept_dets).
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from text if model wrapped it
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    print("  [GeminiAuditor] Could not parse response JSON — keeping all dets")
                    return [], list(all_dets)
            else:
                print("  [GeminiAuditor] Could not parse response JSON — keeping all dets")
                return [], list(all_dets)

        # ── 1. Validation pass ──────────────────────────────────────────
        validations = data.get("validations", {})
        rejected_indices = set()
        for uid, is_valid in validations.items():
            if uid in id_map and is_valid is False:
                rejected_indices.add(id_map[uid])
                print(f"  [GeminiAuditor] REJECTED {uid} ({all_dets[id_map[uid]]['class_name']} "
                      f"conf={all_dets[id_map[uid]]['score']:.2f})")

        kept_dets = [d for i, d in enumerate(all_dets) if i not in rejected_indices]

        # ── 2. Supplementation pass ─────────────────────────────────────
        supplements = data.get("supplements", [])
        idx_to_uid  = {v: k for k, v in id_map.items()}

        # Build worker lookup by uid
        worker_by_uid = {}
        for i, det in enumerate(all_dets):
            uid = idx_to_uid.get(i)
            if uid and uid.startswith("W") and det["class_id"] == 2:
                worker_by_uid[uid] = det

        new_dets = []
        name_to_id = {"helmet": 0, "safety_vest": 1}

        for sup in supplements:
            wuid  = sup.get("worker_id", "")
            cls   = sup.get("class", "")
            box   = sup.get("box", [])

            if cls not in name_to_id:
                continue
            if len(box) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            # Clamp to image bounds
            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            x2 = max(0, min(x2, w_img))
            y2 = max(0, min(y2, h_img))

            bw = x2 - x1; bh = y2 - y1
            if bw < 8 or bh < 8:
                print(f"  [GeminiAuditor] Supplement {cls} from {wuid} too small ({bw}×{bh}) — skipped")
                continue

            # Dedup against existing kept detections (IoU > 0.20 = duplicate)
            xyxy = [x1, y1, x2, y2]
            cid  = name_to_id[cls]
            is_dup = any(
                d["class_id"] == cid and _iou(xyxy, d["xyxy"]) > 0.20
                for d in kept_dets + new_dets
            )
            if is_dup:
                continue

            # Sanity: PPE centre should be near the claimed worker box
            if wuid in worker_by_uid:
                w = worker_by_uid[wuid]
                wx1, wy1, wx2, wy2 = w["xyxy"]
                pcx = (x1 + x2) / 2; pcy = (y1 + y2) / 2
                ww  = wx2 - wx1;     wh  = wy2 - wy1
                margin_x = ww * 0.25
                margin_up = wh * 0.50
                margin_dn = wh * 0.25
                if not (wx1 - margin_x <= pcx <= wx2 + margin_x and
                        wy1 - margin_up <= pcy <= wy2 + margin_dn):
                    print(f"  [GeminiAuditor] Supplement {cls} centre ({pcx:.0f},{pcy:.0f}) "
                          f"outside {wuid} extended box — skipped")
                    continue

            new_det = {
                "class_id":   cid,
                "class_name": cls,
                "xyxy":       xyxy,
                "score":      0.70,   # synthetic confidence for Gemini-added detections
                "source":     "gemini",
            }
            new_dets.append(new_det)
            print(f"  [GeminiAuditor] ADDED {cls} for {wuid} at [{x1},{y1},{x2},{y2}]")

        return new_dets, kept_dets
