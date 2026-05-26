import os
import sys
import time

import numpy as np
import pygame

from src.common.event_manager import EventManager
from src.config import *
from src.core.engine import GameEngine
from src.core.game_state import GameState
from src.core.profile_manager import ProfileManager
from src.core.session_recorder import SessionRecorder
from src.ui.ui_manager import UIManager
from src.vision.camera_threading import CameraThreading
from src.vision.eye_tracker import EyeTracker
from wokwi_simulate_server.external_app import ExternalApp


class GameManager:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen_width, self.screen_height = self.screen.get_size()
        pygame.display.set_caption("ADHD Tracker - Demo Version")

        self.font       = pygame.font.SysFont("Arial", 32)
        self.small_font = pygame.font.SysFont("Arial", 24)
        self.big_font   = pygame.font.SysFont("Arial", 64)
        self.clock      = pygame.time.Clock()

        self.is_running = True
        self.game_state = GameState.LOGIN
        self.has_camera = False

        self.eye_tracker      = None
        self.session_recorder = None
        self.camera           = None
        self.current_user     = None
        self.current_frame    = None
        self.current_state    = None

        # initialize event manager
        self.event_manager = EventManager()

        # initialize managers
        self.ui              = UIManager(self.event_manager)
        self.engine          = GameEngine(self.event_manager)
        self.wokwi_server    = ExternalApp(WOKWI_SERVER)
        self.profile_manager = ProfileManager()

        self.setup_event_subscribers()

    # ── eye tracker + session ───────────────────────────────────────────────

    # eye tracker + session recorder
    def setup_eye_tracker(self, current_user):
        self.eye_tracker      = EyeTracker()
        self.session_recorder = SessionRecorder(current_user)

    # camera
    def setup_camera(self):
        self.camera        = CameraThreading().start()
        self.current_frame = None
        self.current_state = None

    # update camera
    def update_camera(self):
        if not self.camera or not self.eye_tracker or not self.session_recorder:
            return
        self.current_frame = self.camera.read()
        if self.current_frame is not None:
            self.current_state = self.eye_tracker.update(self.current_frame)
            x, y = self.current_state.pred_x, self.current_state.pred_y
            self.session_recorder.record(x, y)

    def render_gaze_cursor(self):
        if self.eye_tracker and self.current_state:
            self.eye_tracker.render(
                self.screen, self.current_frame, self.current_state, self.font
            )

    # heatmap safe
    def generate_heatmap(self):
        if self.session_recorder is None:
            print("[GameManager] No session recorder → skip heatmap")
            return

        try:
            path = self.session_recorder.save_session()
        except Exception as e:
            print(f"[GameManager] save_session() failed: {e} → skip heatmap")
            return

        if path is None:
            print("[GameManager] No gaze data recorded → skip heatmap")
            return

        try:
            bg = np.uint8(np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2)))
            self.session_recorder.generate_heatmap(bg, path)
        except Exception as e:
            print(f"[GameManager] generate_heatmap() failed: {e}")

    # ── events ──────────────────────────────────────────────────────────────

    # event setup
    def setup_event_subscribers(self):
        self.event_manager.subscribe("LOGIN_REQUEST",        self.validate_login)
        self.event_manager.subscribe("START_CALIBRATION",    self.start_calibration)
        self.event_manager.subscribe("BACK_TO_MENU",         self.back_to_menu)
        self.event_manager.subscribe("QUIT_GAME",            self.quit_game)
        self.event_manager.subscribe("START_GAME",           self.start_game)
        self.event_manager.subscribe("BACK_TO_LOGIN",        self.back_to_login)
        self.event_manager.subscribe("MODEL_STATUS_CHANGED", self.on_model_status_changed)

    # login
    def validate_login(self, data):
        username = data["username"].strip()
        password = data["password"].strip()
        if not username:
            return
        user = self.profile_manager.find_user_by_name(username)
        if user:
            if user.password != password:
                return
            self.current_user = user
        else:
            self.current_user = self.profile_manager.create_new_user(username, password)

        self.game_state = GameState.MENU
        self.ui.switch_state(GameState.MENU)
        self.event_manager.emit("MODEL_STATUS_CHANGED", {
            "has_model":  os.path.exists(str(self.current_user.model_path)),
            "model_path": str(self.current_user.model_path),
        })

    # model check
    def on_model_status_changed(self, data):
        if data["has_model"]:
            self.setup_eye_tracker(self.current_user)
            if self.current_user.model_path:
                self.eye_tracker.load_model(self.current_user.model_path)
        else:
            print("[GameManager] no model → calibration required")

    # calibration
    def start_calibration(self):
        if not self.eye_tracker:
            self.setup_eye_tracker(self.current_user)
        self.eye_tracker.create_model(self.current_user.model_path)
        self.game_state = GameState.MENU
        self.ui.switch_state(self.game_state)
        self.event_manager.emit("MODEL_STATUS_CHANGED", {
            "has_model":  True,
            "model_path": str(self.current_user.model_path),
        })

    def back_to_menu(self):
        self.game_state = GameState.MENU
        self.ui.switch_state(GameState.MENU)

    def start_game(self):
        self.game_state = GameState.PLAYING
        self.ui.switch_state(GameState.PLAYING)
        self.engine.reset_game()

    def back_to_login(self):
        self.game_state = GameState.LOGIN
        self.ui.switch_state(GameState.LOGIN)

    def quit_game(self):
        self.is_running = False

    # ── quit summary screen ─────────────────────────────────────────────────

    def show_quit_summary(self):
        if self.eye_tracker is None:
            return

        summary = self.eye_tracker.get_session_summary()

        BG    = (15,  18,  30)
        TITLE = (180, 210, 255)
        STAT  = (200, 200, 200)
        GOOD  = (60,  220, 120)
        WARN  = (220, 160, 40)
        DIM   = (120, 120, 140)

        deadline = time.time() + 6   # auto-close after 6 s

        while time.time() < deadline:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN):
                        return

            self.screen.fill(BG)
            W, H = self.screen_width, self.screen_height
            cx   = W // 2

            # title
            t = self.big_font.render("Session Summary", True, TITLE)
            self.screen.blit(t, t.get_rect(centerx=cx, top=50))

            # stats block
            stats = [
                ("Duration",       f"{summary['session_duration_s']} s"),
                ("Fixation ratio", f"{summary['fixation_ratio_pct']} %"),
                ("Total fixation", f"{summary['total_fixation_time_s']} s"),
                ("Blink rate",     f"{summary['blink_bpm']} BPM"),
            ]
            y = 170
            for label, value in stats:
                self.screen.blit(
                    self.font.render(f"{label}:", True, DIM),  (cx - 300, y)
                )
                self.screen.blit(
                    self.font.render(value,       True, STAT), (cx + 30,  y)
                )
                y += 52

            # divider
            pygame.draw.line(self.screen, (50, 55, 80), (cx - 300, y + 8), (cx + 300, y + 8), 1)
            y += 36

            # suggestions
            hdr = self.small_font.render("Suggestions", True, TITLE)
            self.screen.blit(hdr, hdr.get_rect(centerx=cx, top=y))
            y += 38

            if summary["suggestions"]:
                for tip in summary["suggestions"]:
                    col = GOOD if ("Great" in tip or "stable" in tip) else WARN
                    s   = self.small_font.render(f"•  {tip}", True, col)
                    self.screen.blit(s, s.get_rect(centerx=cx, top=y))
                    y += 34
            else:
                ok = self.font.render("No issues detected.", True, GOOD)
                self.screen.blit(ok, ok.get_rect(centerx=cx, top=y))

            # footer countdown
            remaining = max(0, int(deadline - time.time()) + 1)
            footer = self.small_font.render(
                f"Closing in {remaining}s  —  SPACE to close now", True, DIM
            )
            self.screen.blit(footer, footer.get_rect(centerx=cx, bottom=H - 28))

            pygame.display.update()
            self.clock.tick(30)

    # ── main loop ───────────────────────────────────────────────────────────

    # main loop
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit_game()

            self.ui.handle_event(event)

            if self.game_state == GameState.PLAYING and not self.has_camera:
                self.setup_camera()
                self.has_camera = True

            self.engine._handle_events(event)

    def update(self, deltaTime):
        self.ui.update(deltaTime)
        if self.game_state == GameState.PLAYING:
            self.engine._update()
            self.update_camera()

    def render(self):
        self.screen.fill((0, 0, 0))
        if self.game_state == GameState.PLAYING:
            self.engine._draw(self.screen, self.font)
            self.render_gaze_cursor()
        self.ui.render(self.screen)
        pygame.display.update()

    def run(self):
        while self.is_running:
            deltaTime = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(deltaTime)
            self.render()

        if self.camera:
            self.camera.stop()

        self.wokwi_server.stop()
        self.engine.cleanup()
        self.generate_heatmap()

        # show summary screen before closing
        self.show_quit_summary()

        pygame.quit()
        sys.exit()