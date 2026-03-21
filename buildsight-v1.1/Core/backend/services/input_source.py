import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class InputState:
    mode: str
    source: Optional[str]


class InputSourceManager:
    def __init__(self, live_source: str):
        self._lock = threading.Lock()
        self._live_source = live_source
        self._video_source: Optional[str] = None
        self._mode = "video"
        self._source_changed = False

    def set_video_source(self, path: str) -> None:
        with self._lock:
            self._video_source = path
            self._mode = "video"
            self._source_changed = True

    def set_mode(self, mode: str) -> None:
        if mode not in ["video", "live"]:
            raise ValueError("mode must be 'video' or 'live'")
        with self._lock:
            self._mode = mode
            self._source_changed = True

    def consume_change(self) -> bool:
        with self._lock:
            changed = self._source_changed
            self._source_changed = False
            return changed

    def get_state(self) -> InputState:
        with self._lock:
            source = self._live_source if self._mode == "live" else self._video_source
            return InputState(mode=self._mode, source=source)

    def has_video(self) -> bool:
        with self._lock:
            return bool(self._video_source)
