import time
from dataclasses import dataclass, field
from typing import Dict, List


PPE_PRESENT = "PRESENT"
PPE_ABSENT = "ABSENT"
PPE_UNCERTAIN = "UNCERTAIN"


@dataclass
class TrackState:
    helmet_missing: int = 0
    vest_missing: int = 0
    helmet_confirmed: str = PPE_UNCERTAIN
    vest_confirmed: str = PPE_UNCERTAIN
    last_alert_time: Dict[str, float] = field(default_factory=dict)


@dataclass
class Violation:
    worker_id: int
    violation: str


class ORNReasoner:
    def __init__(
        self,
        confirmation_frames: int = 3,
        occlusion_overlap: float = 0.35,
        alert_cooldown_sec: float = 3.0,
    ):
        self.confirmation_frames = confirmation_frames
        self.occlusion_overlap = occlusion_overlap
        self.alert_cooldown_sec = alert_cooldown_sec
        self._track_states: Dict[int, TrackState] = {}

    def reset(self) -> None:
        self._track_states = {}

    def apply(self, detected_objects: List) -> List[Violation]:
        occlusion_map = self._compute_occlusions(detected_objects)
        violations: List[Violation] = []

        for obj in detected_objects:
            track_id = obj.track_id
            state = self._track_states.setdefault(track_id, TrackState())
            occluded = occlusion_map.get(track_id, False)

            helmet_state = self._resolve_state(
                has_equipment=obj.compliance.helmet,
                occluded=occluded,
                missing_frames=state.helmet_missing,
                confirmed_state=state.helmet_confirmed,
            )
            vest_state = self._resolve_state(
                has_equipment=obj.compliance.vest,
                occluded=occluded,
                missing_frames=state.vest_missing,
                confirmed_state=state.vest_confirmed,
            )

            state.helmet_missing = helmet_state["missing_frames"]
            state.vest_missing = vest_state["missing_frames"]
            state.helmet_confirmed = helmet_state["confirmed_state"]
            state.vest_confirmed = vest_state["confirmed_state"]

            obj.occluded = occluded
            obj.compliance.helmet_state = helmet_state["state"]
            obj.compliance.vest_state = vest_state["state"]
            obj.compliance.overall_state = self._combine_states(
                helmet_state["state"], vest_state["state"]
            )

            now = time.time()
            if self._should_alert(helmet_state, state, now, "helmet"):
                violations.append(Violation(worker_id=track_id, violation="Helmet"))
            if self._should_alert(vest_state, state, now, "vest"):
                violations.append(Violation(worker_id=track_id, violation="Safety Vest"))

        return violations

    def _should_alert(self, ppe_state: Dict[str, int], state: TrackState, now: float, key: str) -> bool:
        if ppe_state["state"] != PPE_ABSENT:
            return False
        if ppe_state["missing_frames"] != self.confirmation_frames:
            return False
        last_time = state.last_alert_time.get(key, 0.0)
        if now - last_time < self.alert_cooldown_sec:
            return False
        state.last_alert_time[key] = now
        return True

    def _resolve_state(self, has_equipment: bool, occluded: bool, missing_frames: int, confirmed_state: str) -> Dict[str, int]:
        if has_equipment:
            return {
                "state": PPE_PRESENT,
                "missing_frames": 0,
                "confirmed_state": PPE_PRESENT,
            }

        if occluded:
            return {
                "state": PPE_UNCERTAIN,
                "missing_frames": missing_frames,
                "confirmed_state": confirmed_state,
            }

        missing_frames += 1
        if missing_frames >= self.confirmation_frames:
            return {
                "state": PPE_ABSENT,
                "missing_frames": missing_frames,
                "confirmed_state": PPE_ABSENT,
            }

        return {
            "state": PPE_UNCERTAIN,
            "missing_frames": missing_frames,
            "confirmed_state": confirmed_state,
        }

    def _combine_states(self, helmet_state: str, vest_state: str) -> str:
        if helmet_state == PPE_ABSENT or vest_state == PPE_ABSENT:
            return PPE_ABSENT
        if helmet_state == PPE_UNCERTAIN or vest_state == PPE_UNCERTAIN:
            return PPE_UNCERTAIN
        return PPE_PRESENT

    def _compute_occlusions(self, detected_objects: List) -> Dict[int, bool]:
        occlusion_map: Dict[int, bool] = {}
        for obj in detected_objects:
            occlusion_map[obj.track_id] = False

        for i, obj in enumerate(detected_objects):
            bbox_a = obj.bbox
            area_a = self._area(bbox_a)
            if area_a <= 0:
                continue
            for j, other in enumerate(detected_objects):
                if i == j:
                    continue
                overlap = self._overlap_ratio(bbox_a, other.bbox, area_a)
                if overlap >= self.occlusion_overlap:
                    occlusion_map[obj.track_id] = True
                    break

        return occlusion_map

    def _area(self, box: List[float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def _overlap_ratio(self, box_a: List[float], box_b: List[float], area_a: float) -> float:
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])
        inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
        if area_a <= 0:
            return 0.0
        return inter_area / area_a


orn_service = ORNReasoner()
