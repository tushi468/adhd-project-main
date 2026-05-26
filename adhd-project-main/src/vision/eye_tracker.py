import collections
import time

import cv2
import numpy as np
import pygame

from eyetrax.calibration import run_9_point_calibration
from eyetrax.filters import KalmanSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from src.config import *
from src.vision.gaze_state import GazeState

# Created once at module level — expensive to initialise per frame
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def _preprocess(frame):
    """
    Bilateral filter + CLAHE on luminance channel.
    Improves iris detection for glasses wearers and small/squinted eyes.
    - Bilateral filter removes noise without blurring the iris edge
    - CLAHE only on L channel (brightness) — correct, not on colour
    Inspired by GazeTracking (github.com/antoinelame/GazeTracking)
    """
    frame = cv2.bilateralFilter(frame, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge([_CLAHE.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# I-DT constants (ported from Pupil Labs defaults)
# https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/fixation_detector.py
_IDT_MIN_DURATION_MS = 150   # ms — minimum fixation hold time
_IDT_MAX_DISPERSION  = 80    # px — max spatial spread across window

# Rolling buffer size — mirrors deque(maxlen=40) from alireza787b/AngleBuffer
# https://github.com/alireza787b/Python-Gaze-Face-Tracker/blob/main/AngleBuffer.py
_SMOOTH_WINDOW = 40


class _GazeBuffer:
    """Rolling mean smoother for (x, y) gaze coordinates."""
    def __init__(self, size: int = _SMOOTH_WINDOW):
        self.buffer: collections.deque = collections.deque(maxlen=size)

    def add(self, x: float, y: float):
        self.buffer.append((x, y))

    def get_average(self):
        return tuple(np.mean(self.buffer, axis=0))

    def clear(self):
        self.buffer.clear()


class EyeTracker:
    def __init__(self):
        self.gaze_estimator = GazeEstimator()
        self.current_state = GazeState()

        # rolling blink log → BPM over last 60 s
        self._blink_log = collections.deque()
        self._prev_blink = False

        # session-level stats for summary report
        self._session_start       = time.time()
        self._total_frames        = 0
        self._fixated_frames      = 0
        self._total_fixation_time = 0.0

        self._set_smoother()

    # -----------------------------
    # MODEL
    # -----------------------------

    def create_model(self, path):
        try:
            # 9-point gives better spatial coverage than 5-point
            run_9_point_calibration(self.gaze_estimator)
            self.gaze_estimator.save_model(path)
            self.gaze_estimator.load_model(path)
            print(f"[EyeTracker] Model created: {path}")
        except Exception as e:
            print(f"[EyeTracker] \033[91mError:\033[0m create_model failed: {e}")
            raise  # surface the failure — caller must know it failed

    def load_model(self, path):
        try:
            self.gaze_estimator.load_model(path)
            print(f"[EyeTracker] Loaded model: {path}")
        except Exception as e:
            print(f"[EyeTracker] \033[91mError:\033[0m load_model failed: {e}")
            raise

    # -----------------------------
    # FILTER
    # -----------------------------

    def _set_smoother(self):
        self.kalman    = make_kalman()
        self.smoother  = KalmanSmoother(self.kalman)
        self._gaze_buf = _GazeBuffer(size=_SMOOTH_WINDOW)

    # -----------------------------
    # RESET
    # -----------------------------

    def reset(self):
        """Clear per-trial state without losing the loaded model.
        Call this between sessions or game rounds."""
        self.current_state        = GazeState()
        self._blink_log.clear()
        self._prev_blink          = False
        self._session_start       = time.time()
        self._total_frames        = 0
        self._fixated_frames      = 0
        self._total_fixation_time = 0.0
        self._set_smoother()

    # -----------------------------
    # I-DT FIXATION DETECTION
    # -----------------------------

    def _idt_fixation(self, s: GazeState, x: int, y: int) -> GazeState:
        """
        Dispersion-threshold fixation classifier (I-DT).
        Ported from Pupil Labs' fixation_detector.py (GNU LGPL v3).
        MIN_DURATION=150ms, MAX_DISPERSION=80px.
        """
        now = time.time()
        s._position_history.append((now, x, y))

        if len(s._position_history) < 2:
            s.is_fixating       = False
            s.dispersion        = 0.0
            s.fixation_duration = 0.0
            return s

        window = list(s._position_history)
        xs = [p[1] for p in window]
        ys = [p[2] for p in window]

        s.dispersion = max(max(xs) - min(xs), max(ys) - min(ys))

        if s.dispersion <= _IDT_MAX_DISPERSION:
            if s._fixation_start is None:
                s._fixation_start = window[0][0]
            held_ms             = (now - s._fixation_start) * 1000
            s.is_fixating       = held_ms >= _IDT_MIN_DURATION_MS
            s.fixation_duration = (now - s._fixation_start) if s.is_fixating else 0.0
        else:
            # drop oldest sample and reset — dispersion exceeded
            s._position_history.popleft()
            s._fixation_start   = None
            s.is_fixating       = False
            s.fixation_duration = 0.0

        return s

    # -----------------------------
    # UPDATE
    # -----------------------------

    def update(self, frame) -> GazeState:
        if frame is None:
            return self.current_state

        # preprocess before feature extraction (glasses/small eyes support)
        processed = _preprocess(frame)

        s = self.current_state
        self._total_frames += 1

        try:
            features, blink_detected = self.gaze_estimator.extract_features(processed)
        except Exception:
            return s

        s.blink_detected = blink_detected

        # track blink onsets (rising edge only) for BPM
        # rising edge = only the moment blinking starts, not every blink frame
        if blink_detected and not self._prev_blink:
            self._blink_log.append(time.time())
        self._prev_blink = blink_detected

        # drop blinks older than 60 seconds from the rolling window
        now = time.time()
        while self._blink_log and self._blink_log[0] < now - 60:
            self._blink_log.popleft()

        if features is not None and not blink_detected:
            try:
                gaze_point       = self.gaze_estimator.predict(np.array([features]))[0]
                raw_x, raw_y     = map(int, gaze_point)

                # Kalman first, then rolling mean (two-stage smoothing)
                kx, ky           = self.smoother.step(raw_x, raw_y)
                self._gaze_buf.add(kx, ky)
                sx, sy           = self._gaze_buf.get_average()
                s.pred_x, s.pred_y = int(sx), int(sy)

                s.cursor_alpha   = min(s.cursor_alpha + CURSOR_STEP, 1.0)

                # I-DT fixation on the smoothed coordinate
                s = self._idt_fixation(s, s.pred_x, s.pred_y)

                if s.is_fixating:
                    self._fixated_frames      += 1
                    self._total_fixation_time += 1.0 / FPS

            except Exception:
                s.pred_x = s.pred_y = None
                s.cursor_alpha = max(s.cursor_alpha - CURSOR_STEP, 0.0)
                s.is_fixating  = False
        else:
            s.pred_x = s.pred_y = None
            s.cursor_alpha = max(s.cursor_alpha - CURSOR_STEP, 0.0)
            s.is_fixating  = False

        return s

    # -----------------------------
    # RENDER
    # -----------------------------

    def render(self, screen, frame, current_state, font):
        if frame is None:
            return

        # flip for display only — update() already saw the preprocessed frame
        frame = cv2.flip(frame, 1)

        # gaze cursor with alpha fade + fixation ring
        if (
            current_state.pred_x is not None
            and current_state.pred_y is not None
            and current_state.cursor_alpha > 0
        ):
            cx        = int(current_state.pred_x)
            cy        = int(current_state.pred_y)
            alpha_int = int(current_state.cursor_alpha * 255)

            if current_state.is_fixating:
                # SRCALPHA surface so the alpha fade actually works
                duration_bonus = min(current_state.fixation_duration / 2.0, 1.0) * 10
                radius = int(18 + duration_bonus)
                pad    = 14
                size   = (radius + pad) * 2
                surf   = pygame.Surface((size, size), pygame.SRCALPHA)
                center = (radius + pad, radius + pad)

                pygame.draw.circle(surf, (*COLORS[4][:3], alpha_int), center, radius)
                # outer ring: green = fixating
                pygame.draw.circle(surf, (60, 220, 120, alpha_int // 2), center, radius, 3)

                # halo when dispersion is very low (<10 px)
                if current_state.dispersion < 10:
                    halo_alpha = int(alpha_int * 0.3)
                    pygame.draw.circle(
                        surf, (60, 220, 120, halo_alpha), center, radius + 8, 1
                    )

                screen.blit(surf, (cx - radius - pad, cy - radius - pad))

            else:
                radius = 15
                surf   = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                center = (radius, radius)

                # SRCALPHA surface so the alpha fade actually works
                pygame.draw.circle(surf, (*COLORS[4][:3], alpha_int), center, radius)
                # outer ring: amber = moving
                pygame.draw.circle(surf, (220, 160, 40, alpha_int // 3), center, radius, 2)
                screen.blit(surf, (cx - radius, cy - radius))

        # camera thumbnail
        try:
            thumb = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
            screen.blit(
                pygame.surfarray.make_surface(thumb),
                (
                    screen.get_width()  - CAM_WIDTH  - MARGIN,
                    screen.get_height() - CAM_HEIGHT - MARGIN,
                ),
            )
        except Exception:
            pass

        # HUD: blink state + rolling BPM
        bpm = len(self._blink_log)
        if current_state.blink_detected:
            blink_txt = f"Blinking   BPM: {bpm}"
            blink_clr = (80, 80, 240)
        else:
            blink_txt = f"Eyes open  BPM: {bpm}"
            blink_clr = (60, 210, 100)

        screen.blit(font.render(blink_txt, True, blink_clr), (50, 100))

    # -----------------------------
    # SESSION SUMMARY
    # -----------------------------

    def get_session_summary(self) -> dict:
        """Return per-session analytics and plain-English suggestions."""
        session_duration = time.time() - self._session_start
        fix_ratio = (
            self._fixated_frames / self._total_frames
            if self._total_frames > 0 else 0.0
        )
        bpm = len(self._blink_log)

        suggestions = []

        if fix_ratio < 0.15:
            suggestions.append("Low fixation — try recalibrating or sit closer to screen.")
        elif fix_ratio < 0.40:
            suggestions.append("Moderate fixation — good for a webcam session!")
        elif fix_ratio > 0.70:
            suggestions.append("Excellent focus! Very stable gaze this session.")

        if bpm > 30:
            suggestions.append("High blink rate — check lighting or eye fatigue.")
        elif bpm < 6:
            suggestions.append("Very low blink rate — remember to blink regularly.")
        else:
            suggestions.append(f"Healthy blink rate at {bpm} BPM.")

        if session_duration < 30:
            suggestions.append("Short session — play longer for better accuracy.")

        return {
            "session_duration_s":    round(session_duration, 1),
            "fixation_ratio_pct":    round(fix_ratio * 100, 1),
            "total_fixation_time_s": round(self._total_fixation_time, 1),
            "blink_bpm":             bpm,
            "suggestions":           suggestions,
        }