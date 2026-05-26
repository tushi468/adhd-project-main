from dataclasses import dataclass, field
from typing import Optional, List
import collections


@dataclass
class GazeState:
    """
    Stores current gaze estimation state for one video frame.

    Fixation logic follows Pupil Labs' dispersion-duration model:
      - dispersion  : spatial spread of recent gaze samples (pixels)
      - duration    : how long gaze has stayed within dispersion threshold (s)
      - is_fixating : True once both thresholds are satisfied
    """
    pred_x:         Optional[float] = None
    pred_y:         Optional[float] = None
    blink_detected: Optional[bool]  = None
    cursor_alpha:   float           = 0.0
    contours:       List            = field(default_factory=list)  # only for kde filter

    # True when gaze has been stable in one spot — used for fixation ring
    is_fixating:       bool  = False
    fixation_duration: float = 0.0   # seconds held in current fixation
    dispersion:        float = 0.0   # max pixel spread across history window

    # internal rolling buffer — NOT serialised
    # AngleBuffer pattern from alireza787b/Python-Gaze-Face-Tracker
    # (MIT, https://github.com/alireza787b/Python-Gaze-Face-Tracker)
    # Original uses deque(maxlen=40) for angle smoothing; we apply the same
    # pattern to (timestamp, x, y) gaze positions.
    _position_history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=40),
        repr=False, compare=False,
    )
    _fixation_start: Optional[float] = field(
        default=None, repr=False, compare=False
    )