import threading
import time


class SceneState:
    """Thread-safe shared scene data with exponential moving average consensus.

    Each (position, candy_type) pair maintains a score that rises when
    detected and decays when not detected.  A candy appears in the
    consensus when its score exceeds a threshold, which naturally
    filters brief false detections while retaining intermittent but
    sustained detections.
    """

    def __init__(self, alpha: float = 0.2, threshold: float = 0.2):
        self._lock = threading.Lock()
        self._alpha = alpha
        self._threshold = threshold
        # position -> {candy_type -> score}
        self._scores: dict[str, dict[str, float]] = {}
        self._all_positions: set[str] = set()
        self._scene: dict[str, str] = {}
        self._last_updated: float = 0.0

    def update(self, raw_scene: dict[str, str], all_positions: set[str] | None = None) -> None:
        with self._lock:
            if all_positions is not None:
                self._all_positions = all_positions
            alpha = self._alpha
            for pos in self._all_positions:
                if pos not in self._scores:
                    self._scores[pos] = {}
                detected_type = raw_scene.get(pos)
                # Decay all existing type scores for this position,
                # and boost the detected type
                for candy_type in list(self._scores[pos]):
                    if candy_type == detected_type:
                        self._scores[pos][candy_type] = self._scores[pos][candy_type] * (1 - alpha) + alpha
                    else:
                        self._scores[pos][candy_type] *= (1 - alpha)
                        if self._scores[pos][candy_type] < 0.01:
                            del self._scores[pos][candy_type]
                # New type not seen before at this position
                if detected_type and detected_type not in self._scores[pos]:
                    self._scores[pos][detected_type] = alpha
            self._scene = self._compute_consensus()
            self._last_updated = time.monotonic()

    def get_scene(self) -> dict[str, str]:
        with self._lock:
            return dict(self._scene)

    def _compute_consensus(self) -> dict[str, str]:
        result = {}
        for pos, types in self._scores.items():
            if types:
                best_type = max(types, key=lambda k: types[k])
                if types[best_type] >= self._threshold:
                    result[pos] = best_type
        return dict(sorted(result.items()))
