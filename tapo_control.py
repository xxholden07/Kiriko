import argparse
import json
import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import ephem

os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
from onvif import ONVIFCamera


CONFIG_FILE = Path(__file__).resolve().parent / "tapo_config.json"

WINDOW_NAME = "Tapo C200 - Controle"

# Haarcascade para deteccao facial
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Coordenadas de Araraquara
OBSERVER_LAT = "-21.7946"
OBSERVER_LON = "-48.1766"
OBSERVER_ELEV = 664
BRT_OFFSET = timedelta(hours=3)


class SunTracker:
    """Rastreamento solar em tempo real usando ephem."""

    def __init__(self) -> None:
        self._obs = ephem.Observer()
        self._obs.lat = OBSERVER_LAT
        self._obs.lon = OBSERVER_LON
        self._obs.elevation = OBSERVER_ELEV
        self._sun = ephem.Sun()
        self._cache: dict = {}
        self._last_update = 0.0

    def update(self) -> dict:
        now = time.time()
        if now - self._last_update < 10:
            return self._cache
        self._last_update = now

        utc_now = datetime.now(UTC)
        self._obs.date = utc_now
        self._sun.compute(self._obs)

        elevacao = math.degrees(self._sun.alt)
        azimute = math.degrees(self._sun.az)

        try:
            # Calcular nascer e por do dia de hoje (BRT)
            # Meia-noite BRT = 03:00 UTC
            today_start = utc_now.replace(hour=3, minute=0, second=0, microsecond=0)
            if utc_now.hour < 3:
                today_start -= timedelta(days=1)
            obs_today = ephem.Observer()
            obs_today.lat = OBSERVER_LAT
            obs_today.lon = OBSERVER_LON
            obs_today.elevation = OBSERVER_ELEV
            obs_today.date = today_start
            sun_today = ephem.Sun()
            nascer_utc = obs_today.next_rising(sun_today).datetime()
            por_utc = obs_today.next_setting(sun_today).datetime()
            nascer_brt = (nascer_utc - BRT_OFFSET).strftime("%H:%M")
            por_brt = (por_utc - BRT_OFFSET).strftime("%H:%M")
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            nascer_brt = "--:--"
            por_brt = "--:--"

        is_day = elevacao > 0

        self._cache = {
            "elevacao": elevacao,
            "azimute": azimute,
            "nascer": nascer_brt,
            "por": por_brt,
            "is_day": is_day,
        }
        return self._cache


def build_rtsp_url(ip: str, user: str, password: str, stream: str) -> str:
    safe_user = quote(user, safe="")
    safe_password = quote(password, safe="")
    return f"rtsp://{safe_user}:{safe_password}@{ip}:554/{stream}"


def load_config() -> dict[str, object]:
    config: dict[str, object] = {}
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    env_map = {
        "ip": os.getenv("TAPO_IP"),
        "user": os.getenv("TAPO_USER"),
        "password": os.getenv("TAPO_PASSWORD"),
        "stream": os.getenv("TAPO_STREAM"),
        "onvif_port": os.getenv("TAPO_ONVIF_PORT"),
        "speed": os.getenv("TAPO_SPEED"),
        "pulse_duration": os.getenv("TAPO_PULSE_DURATION"),
    }
    for key, value in env_map.items():
        if value not in (None, ""):
            config[key] = value

    return config


def get_wsdl_dir() -> str:
    package_dir = Path(__file__).resolve().parent / ".venv" / "lib"
    candidates = list(package_dir.glob("python*/site-packages/wsdl"))
    if candidates:
        return str(candidates[0])

    import onvif as _onvif

    site_packages = Path(_onvif.__file__).resolve().parent.parent
    return str(site_packages / "wsdl")


class TapoCameraController:
    """Controle PTZ + Imaging (visao noturna, brilho, contraste etc.)."""

    IR_MODES = ("AUTO", "ON", "OFF")

    def __init__(self, ip: str, user: str, password: str, port: int) -> None:
        self.camera = ONVIFCamera(ip, port, user, password, get_wsdl_dir())
        self.media_service = self.camera.create_media_service()
        self.ptz_service = self.camera.create_ptz_service()
        self.profile = self.media_service.GetProfiles()[0]
        self.profile_token = self.profile.token
        self.video_source_token = self.profile.VideoSourceConfiguration.SourceToken

        request = self.ptz_service.create_type("GetConfigurationOptions")
        request.ConfigurationToken = self.profile.PTZConfiguration.token
        configuration = self.ptz_service.GetConfigurationOptions(request)
        self.velocity_space = configuration.Spaces.ContinuousPanTiltVelocitySpace[0].URI

        # Imaging service
        self.imaging_service = self.camera.create_imaging_service()
        self._ir_mode_index = 0  # AUTO

    # ── PTZ ──────────────────────────────────────────────────────────────

    def move(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        request = self.ptz_service.create_type("ContinuousMove")
        request.ProfileToken = self.profile_token
        request.Velocity = {
            "PanTilt": {"x": pan, "y": tilt, "space": self.velocity_space},
            "Zoom": {"x": zoom},
        }
        self.ptz_service.ContinuousMove(request)

    def smooth_pulse(
        self,
        pan: float,
        tilt: float,
        zoom: float = 0.0,
        duration: float = 0.22,
    ) -> None:
        ramp = (0.35, 0.7, 1.0, 0.7, 0.35)
        step_sleep = max(duration / len(ramp), 0.02)
        for factor in ramp:
            self.move(pan * factor, tilt * factor, zoom * factor)
            time.sleep(step_sleep)
        self.stop()

    def stop(self) -> None:
        request = self.ptz_service.create_type("Stop")
        request.ProfileToken = self.profile_token
        request.PanTilt = True
        request.Zoom = True
        self.ptz_service.Stop(request)

    def go_home(self) -> None:
        try:
            request = self.ptz_service.create_type("GotoHomePosition")
            request.ProfileToken = self.profile_token
            self.ptz_service.GotoHomePosition(request)
        except Exception:
            pass

    # ── Imaging ──────────────────────────────────────────────────────────

    def _get_imaging_settings(self):
        req = self.imaging_service.create_type("GetImagingSettings")
        req.VideoSourceToken = self.video_source_token
        return self.imaging_service.GetImagingSettings(req)

    def _set_imaging(self, **kwargs) -> None:
        req = self.imaging_service.create_type("SetImagingSettings")
        req.VideoSourceToken = self.video_source_token
        req.ImagingSettings = kwargs
        self.imaging_service.SetImagingSettings(req)

    def cycle_ir_mode(self) -> str:
        self._ir_mode_index = (self._ir_mode_index + 1) % len(self.IR_MODES)
        mode = self.IR_MODES[self._ir_mode_index]
        try:
            self._set_imaging(IrCutFilter=mode)
        except Exception:
            pass
        return mode

    def get_ir_mode(self) -> str:
        return self.IR_MODES[self._ir_mode_index]

    def adjust_brightness(self, delta: float) -> float:
        settings = self._get_imaging_settings()
        new_val = max(0.0, min(100.0, (settings.Brightness or 50.0) + delta))
        self._set_imaging(Brightness=new_val)
        return new_val

    def adjust_contrast(self, delta: float) -> float:
        settings = self._get_imaging_settings()
        new_val = max(0.0, min(100.0, (settings.Contrast or 50.0) + delta))
        self._set_imaging(Contrast=new_val)
        return new_val

    def adjust_sharpness(self, delta: float) -> float:
        settings = self._get_imaging_settings()
        new_val = max(0.0, min(100.0, (settings.Sharpness or 50.0) + delta))
        self._set_imaging(Sharpness=new_val)
        return new_val

    def adjust_saturation(self, delta: float) -> float:
        settings = self._get_imaging_settings()
        new_val = max(0.0, min(100.0, (settings.ColorSaturation or 50.0) + delta))
        self._set_imaging(ColorSaturation=new_val)
        return new_val

    def focus_background(self) -> None:
        """Tenta forcar foco no fundo: NearLimit alto + nitidez maxima."""
        try:
            self._set_imaging(
                Focus={"AutoFocusMode": "MANUAL", "NearLimit": 10.0},
                Sharpness=100.0,
            )
        except Exception:
            try:
                self._set_imaging(Sharpness=100.0)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description="Visualizar e controlar Tapo C200")
    parser.add_argument("--ip", default=config.get("ip"), help="IP da camera")
    parser.add_argument("--user", default=config.get("user"), help="Usuario da camera")
    parser.add_argument("--password", default=config.get("password"), help="Senha da camera")
    parser.add_argument(
        "--stream",
        default=config.get("stream", "stream1"),
        choices=["stream1", "stream2"],
    )
    parser.add_argument(
        "--onvif-port",
        type=int,
        default=int(config.get("onvif_port", 2020)),
        help="Porta ONVIF",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=float(config.get("speed", 0.5)),
        help="Velocidade do movimento",
    )
    parser.add_argument(
        "--pulse-duration",
        type=float,
        default=float(config.get("pulse_duration", 0.22)),
        help="Duracao do pulso suave em segundos",
    )
    args = parser.parse_args()

    missing = [name for name in ("ip", "user", "password") if not getattr(args, name)]
    if missing:
        parser.error(
            "configure tapo_config.json, variaveis TAPO_* ou informe --ip, --user e --password"
        )

    return args


# ── HUD overlay ──────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, state: dict) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Barra inferior semi-transparente
    bar_h = 38
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    color = (200, 200, 200)
    thick = 1
    y = h - 12

    sharpen_name = SHARPEN_LEVELS[state.get("sharpen_index", 0)][0]
    night_mode = state.get("night_mode", "OFF")
    tl = state.get("timelapse_active", False)
    tl_count = state.get("timelapse_frames", 0)

    parts = [
        f"IR: {state.get('ir_mode', 'AUTO')}",
        f"Face: {'ON' if state.get('face_detect') else 'OFF'}",
        f"Foco: {sharpen_name}",
        f"Ceu: {night_mode}",
    ]
    if tl:
        parts.append(f"TL: REC {tl_count}")

    rec = state.get("recorder")
    rec_prefix = ""
    if rec and rec.active:
        rec_prefix = f"REC {rec.elapsed} (ate {rec.stop_label})  |  "

    if isinstance(state.get('brightness'), (int, float)):
        parts.append(f"Bri: {state['brightness']:.0f}")

    text = "  |  ".join(parts)
    x_start = 10
    # Desenhar REC em vermelho se gravando
    if rec_prefix:
        cv2.putText(frame, rec_prefix, (x_start, y), font, scale, (0, 0, 255), thick, cv2.LINE_AA)
        x_start += cv2.getTextSize(rec_prefix, font, scale, thick)[0][0]
    cv2.putText(frame, text, (x_start, y), font, scale, color, thick, cv2.LINE_AA)

    # Faces label
    face_count = state.get("face_count", 0)
    if face_count > 0:
        label = f"Faces: {face_count}"
        cv2.putText(frame, label, (w - 120, 30), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Sol info (canto superior direito)
    sun = state.get("sun", {})
    if sun:
        is_day = sun.get("is_day", True)
        sun_icon = "Dia" if is_day else "Noite"
        sun_color = (0, 200, 255) if is_day else (200, 150, 50)
        elev = sun.get("elevacao", 0)
        az = sun.get("azimute", 0)
        nascer = sun.get("nascer", "--:--")
        por = sun.get("por", "--:--")

        sun_lines = [
            f"{sun_icon}  Elev: {elev:.1f}  Az: {az:.1f}",
            f"Nascer: {nascer}  Por: {por}",
        ]
        for i, sl in enumerate(sun_lines):
            tx = w - 310
            ty = 55 + i * 18
            cv2.putText(frame, sl, (tx, ty), font, 0.4, sun_color, 1, cv2.LINE_AA)

    # Help hint top-left
    if state.get("show_help"):
        draw_help(frame)


HELP_LINES = [
    "=== Controles ===",
    "W/A/S/D  Movimentar camera",
    "Z/X      Zoom in / out",
    "Espaco   Parar movimento",
    "H        Home position",
    "N        Ciclar visao noturna IR (AUTO/ON/OFF)",
    "M        Modo ceu noturno (Realce/Stack/Ambos)",
    "F        Liga/desliga deteccao facial",
    "R        Gravar video continuamente",
    "T        Iniciar/parar timelapse",
    "Y        Ciclar intervalo timelapse",
    "U        Compilar timelapse em video",
    "B/V      Brilho +/-",
    "C/G      Contraste +/-",
    "K/L      Nitidez HW +/-",
    "E        Ciclar nitidez SW (foco fundo)",
    "I/O      Saturacao +/-",
    "F11      Tela cheia / janela",
    "F1       Mostrar/esconder ajuda",
    "P        Screenshot",
    "Q        Sair",
]


def draw_help(frame: np.ndarray) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    line_h = 20
    pad = 12
    box_w = 320
    box_h = pad * 2 + line_h * len(HELP_LINES)

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(HELP_LINES):
        color = (0, 200, 255) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (16, 8 + pad + line_h * (i + 1) - 4), font, scale, color, thick, cv2.LINE_AA)


# ── Deteccao facial ─────────────────────────────────────────────────────────

# ── Filtro de nitidez por software (unsharp mask) ────────────────────────────

SHARPEN_LEVELS = (
    ("OFF", 0.0),
    ("Leve", 0.8),
    ("Medio", 1.5),
    ("Forte", 2.5),
    ("Max", 4.0),
)


def apply_sharpen(frame: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return frame
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    return cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)


# ── Timelapse ────────────────────────────────────────────────────────────────

TIMELAPSE_DIR = Path(__file__).resolve().parent / "timelapse"


class TimelapseRecorder:
    """Captura frames em intervalo para montar timelapse."""

    def __init__(self, interval: float = 5.0) -> None:
        self.interval = interval  # segundos entre frames
        self.active = False
        self._session_dir: Path | None = None
        self._frame_count = 0
        self._last_capture = 0.0

    def toggle(self) -> bool:
        self.active = not self.active
        if self.active:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._session_dir = TIMELAPSE_DIR / ts
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._frame_count = 0
            self._last_capture = 0.0
        return self.active

    def tick(self, frame: np.ndarray) -> bool:
        """Retorna True se um frame foi salvo nesta iteracao."""
        if not self.active:
            return False
        now = time.time()
        if now - self._last_capture < self.interval:
            return False
        self._last_capture = now
        fname = self._session_dir / f"frame_{self._frame_count:06d}.jpg"
        cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self._frame_count += 1
        return True

    def cycle_interval(self) -> float:
        options = [2.0, 5.0, 10.0, 30.0, 60.0]
        try:
            idx = options.index(self.interval)
            self.interval = options[(idx + 1) % len(options)]
        except ValueError:
            self.interval = 5.0
        return self.interval

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def session_dir(self) -> str:
        return str(self._session_dir) if self._session_dir else ""

    def compile_video(self, fps: int = 30) -> str | None:
        """Compila frames salvos em video MP4."""
        if not self._session_dir or self._frame_count == 0:
            return None
        frames = sorted(self._session_dir.glob("frame_*.jpg"))
        if not frames:
            return None
        sample = cv2.imread(str(frames[0]))
        h, w = sample.shape[:2]
        out_path = str(self._session_dir / "timelapse.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is not None:
                writer.write(img)
        writer.release()
        return out_path


# ── Gravacao continua ────────────────────────────────────────────────────────

RECORDING_DIR = Path(__file__).resolve().parent / "recordings"


class ContinuousRecorder:
    """Grava video continuamente com segmentos de 1 hora. Auto-para no horario programado."""

    SEGMENT_SECS = 3600  # 1 hora por segmento

    def __init__(self, stop_time: float | None = None) -> None:
        self.active = False
        self._stop_time = stop_time  # timestamp UTC para auto-stop
        self._writer: cv2.VideoWriter | None = None
        self._session_dir: Path | None = None
        self._segment_index = 0
        self._segment_start = 0.0
        self._start_time = 0.0
        self._frame_size: tuple[int, int] | None = None

    def start(self, stop_time: float | None = None) -> None:
        if self.active:
            return
        if stop_time is not None:
            self._stop_time = stop_time
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._session_dir = RECORDING_DIR / ts
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._segment_index = 0
        self._start_time = time.time()
        self._segment_start = self._start_time
        self.active = True

    def stop(self) -> str:
        if self._writer:
            self._writer.release()
            self._writer = None
        self.active = False
        return str(self._session_dir) if self._session_dir else ""

    def toggle(self, stop_time: float | None = None) -> bool:
        if self.active:
            self.stop()
        else:
            self.start(stop_time)
        return self.active

    def feed(self, frame: np.ndarray) -> None:
        """Alimenta um frame para gravacao. Gerencia segmentos automaticamente."""
        if not self.active:
            return

        now = time.time()

        # Auto-stop no horario programado
        if self._stop_time and now >= self._stop_time:
            self.stop()
            return

        h, w = frame.shape[:2]

        # Novo segmento se necessario
        if self._writer is None or (now - self._segment_start) >= self.SEGMENT_SECS:
            if self._writer:
                self._writer.release()
            self._frame_size = (w, h)
            seg_name = f"seg_{self._segment_index:04d}.mp4"
            path = str(self._session_dir / seg_name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(path, fourcc, 15.0, (w, h))
            self._segment_start = now
            self._segment_index += 1

        self._writer.write(frame)

    @property
    def elapsed(self) -> str:
        if not self.active:
            return "00:00:00"
        secs = int(time.time() - self._start_time)
        hh, rem = divmod(secs, 3600)
        mm, ss = divmod(rem, 60)
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    @property
    def stop_label(self) -> str:
        if self._stop_time:
            return time.strftime("%H:%M", time.localtime(self._stop_time))
        return "--:--"

    @property
    def segments(self) -> int:
        return self._segment_index


# ── Modo ceu noturno (Night Sky Enhance) ─────────────────────────────────────

NIGHT_MODES = (
    "OFF",
    "Denoise",       # reducao de ruido forte
    "Realce",        # denoise + CLAHE + gamma
    "Stacking",      # media de N frames (reducao de ruido temporal)
    "Realce+Stack",  # todos combinados
)


class NightSkyEnhancer:
    """Melhora imagem noturna/IR com denoise, CLAHE, gamma e frame stacking."""

    STACK_SIZE = 12  # frames para media temporal

    def __init__(self) -> None:
        self._mode_index = 0
        self._stack: list[np.ndarray] = []
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @property
    def mode(self) -> str:
        return NIGHT_MODES[self._mode_index]

    def cycle_mode(self) -> str:
        self._mode_index = (self._mode_index + 1) % len(NIGHT_MODES)
        if "Stack" not in self.mode:
            self._stack.clear()
        return self.mode

    def process(self, frame: np.ndarray) -> np.ndarray:
        mode = self.mode
        if mode == "OFF":
            return frame

        result = frame

        # 1) Stacking primeiro (reducao de ruido temporal)
        if "Stack" in mode:
            self._stack.append(frame.astype(np.float32))
            if len(self._stack) > self.STACK_SIZE:
                self._stack.pop(0)
            if len(self._stack) >= 2:
                result = np.mean(self._stack, axis=0).astype(np.uint8)

        # 2) Denoise forte (sempre que nao OFF)
        result = self._denoise(result)

        # 3) Realce suave (CLAHE + gamma) por ultimo
        if "Realce" in mode:
            result = self._enhance(result)

        return result

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        # Denoise agressivo para IR noturno
        return cv2.fastNlMeansDenoisingColored(frame, None, 12, 12, 7, 21)

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        # CLAHE suave no canal L (LAB)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        enhanced = cv2.merge([l_ch, a_ch, b_ch])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Gamma leve para levantar sombras sem explodir ruido
        gamma = 0.75
        table = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        enhanced = cv2.LUT(enhanced, table)

        return enhanced


# ── Deteccao facial ─────────────────────────────────────────────────────────

def detect_faces(frame: np.ndarray, cascade: cv2.CascadeClassifier) -> list:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
    return faces if faces is not None and len(faces) > 0 else []


def draw_faces(frame: np.ndarray, faces) -> None:
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


# ── Controle principal ───────────────────────────────────────────────────────

def handle_key(key: int, controller: TapoCameraController, state: dict) -> bool:
    speed = state["speed"]
    pulse = state["pulse_duration"]

    # Movimento PTZ
    if key == ord("a"):
        controller.smooth_pulse(-speed, 0.0, duration=pulse)
    elif key == ord("d"):
        controller.smooth_pulse(speed, 0.0, duration=pulse)
    elif key == ord("w"):
        controller.smooth_pulse(0.0, speed, duration=pulse)
    elif key == ord("s"):
        controller.smooth_pulse(0.0, -speed, duration=pulse)
    elif key == ord("z"):
        controller.smooth_pulse(0.0, 0.0, speed, duration=pulse)
    elif key == ord("x"):
        controller.smooth_pulse(0.0, 0.0, -speed, duration=pulse)
    elif key == ord(" "):
        controller.stop()
    elif key == ord("h"):
        controller.go_home()

    # Visao noturna (IR)
    elif key == ord("n"):
        mode = controller.cycle_ir_mode()
        state["ir_mode"] = mode
        state["status_msg"] = f"Visao noturna: {mode}"
        state["status_until"] = time.time() + 2.0

    # Deteccao facial
    elif key == ord("f"):
        state["face_detect"] = not state["face_detect"]
        label = "ON" if state["face_detect"] else "OFF"
        state["status_msg"] = f"Deteccao facial: {label}"
        state["status_until"] = time.time() + 2.0

    # Brilho
    elif key == ord("b"):
        val = controller.adjust_brightness(+10)
        state["brightness"] = val
        state["status_msg"] = f"Brilho: {val:.0f}"
        state["status_until"] = time.time() + 1.5
    elif key == ord("v"):
        val = controller.adjust_brightness(-10)
        state["brightness"] = val
        state["status_msg"] = f"Brilho: {val:.0f}"
        state["status_until"] = time.time() + 1.5

    # Contraste
    elif key == ord("c"):
        val = controller.adjust_contrast(+10)
        state["status_msg"] = f"Contraste: {val:.0f}"
        state["status_until"] = time.time() + 1.5
    elif key == ord("g"):
        val = controller.adjust_contrast(-10)
        state["status_msg"] = f"Contraste: {val:.0f}"
        state["status_until"] = time.time() + 1.5

    # Nitidez
    elif key == ord("k"):
        val = controller.adjust_sharpness(+10)
        state["status_msg"] = f"Nitidez: {val:.0f}"
        state["status_until"] = time.time() + 1.5
    elif key == ord("l"):
        val = controller.adjust_sharpness(-10)
        state["status_msg"] = f"Nitidez: {val:.0f}"
        state["status_until"] = time.time() + 1.5

    # Saturacao
    elif key == ord("i"):
        val = controller.adjust_saturation(+10)
        state["status_msg"] = f"Saturacao: {val:.0f}"
        state["status_until"] = time.time() + 1.5
    elif key == ord("o"):
        val = controller.adjust_saturation(-10)
        state["status_msg"] = f"Saturacao: {val:.0f}"
        state["status_until"] = time.time() + 1.5

    # Nitidez por software (ciclar niveis)
    elif key == ord("e"):
        idx = state.get("sharpen_index", 0)
        idx = (idx + 1) % len(SHARPEN_LEVELS)
        state["sharpen_index"] = idx
        name, _ = SHARPEN_LEVELS[idx]
        state["status_msg"] = f"Nitidez SW: {name}"
        state["status_until"] = time.time() + 2.0

    # Modo ceu noturno
    elif key == ord("m"):
        mode = state["night_enhancer"].cycle_mode()
        state["night_mode"] = mode
        state["status_msg"] = f"Ceu noturno: {mode}"
        state["status_until"] = time.time() + 2.0

    # Gravacao continua
    elif key == ord("r"):
        rec_cont = state["recorder"]
        active = rec_cont.toggle()
        if active:
            state["status_msg"] = f"REC ON (ate {rec_cont.stop_label})"
        else:
            state["status_msg"] = "REC OFF"
        state["status_until"] = time.time() + 2.5

    # Timelapse
    elif key == ord("t"):
        rec = state["timelapse"]
        active = rec.toggle()
        state["timelapse_active"] = active
        if active:
            state["status_msg"] = f"Timelapse ON ({rec.interval:.0f}s)"
        else:
            state["status_msg"] = f"Timelapse OFF ({rec.frame_count} frames)"
        state["status_until"] = time.time() + 2.5

    elif key == ord("y"):
        rec = state["timelapse"]
        interval = rec.cycle_interval()
        state["status_msg"] = f"Timelapse intervalo: {interval:.0f}s"
        state["status_until"] = time.time() + 2.0

    elif key == ord("u"):
        rec = state["timelapse"]
        state["status_msg"] = "Compilando timelapse..."
        state["status_until"] = time.time() + 1.0
        path = rec.compile_video()
        if path:
            state["status_msg"] = f"Video: {path}"
        else:
            state["status_msg"] = "Nenhum frame para compilar"
        state["status_until"] = time.time() + 3.0

    # Screenshot
    elif key == ord("p"):
        state["take_screenshot"] = True

    # Help
    elif key == 0x70:  # F1
        state["show_help"] = not state.get("show_help", False)

    # Fullscreen toggle (F11 = scancode varies, use 0x7A on some systems)
    elif key in (0x7A, 0x00):
        toggle_fullscreen(state)

    # Sair
    elif key == ord("q"):
        controller.stop()
        return False

    return True


def toggle_fullscreen(state: dict) -> None:
    state["fullscreen"] = not state.get("fullscreen", False)
    if state["fullscreen"]:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


def main() -> None:
    args = parse_args()

    source = build_rtsp_url(args.ip, args.user, args.password, args.stream)

    try:
        controller = TapoCameraController(args.ip, args.user, args.password, args.onvif_port)
    except Exception as error:
        print(f"Erro ao iniciar controle ONVIF: {error}")
        return

    # Silenciar logs do ffmpeg
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

    camera = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Preferir TCP para menos erros de decodificacao
    camera.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    camera.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    if not camera.isOpened():
        # Tentar com RTSP sobre TCP
        camera.release()
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        camera = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        print("Erro: nao foi possivel abrir o stream RTSP.")
        print(f"Fonte usada: {source}")
        return

    # Janela redimensionavel que suporta fullscreen
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    # Cascade para face detection
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    # Rastreamento solar
    sun_tracker = SunTracker()

    # Forcar foco no fundo ao iniciar
    controller.focus_background()

    timelapse = TimelapseRecorder(interval=5.0)
    timelapse.toggle()  # Auto-iniciar timelapse
    night_enhancer = NightSkyEnhancer()

    # Gravacao continua: parar amanha as 07:00 BRT (10:00 UTC)
    import datetime
    now_dt = datetime.datetime.now()
    if now_dt.hour >= 7:
        stop_dt = now_dt.replace(hour=7, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
    else:
        stop_dt = now_dt.replace(hour=7, minute=0, second=0, microsecond=0)
    recorder = ContinuousRecorder(stop_time=stop_dt.timestamp())
    recorder.start()

    state: dict = {
        "speed": args.speed,
        "pulse_duration": args.pulse_duration,
        "fullscreen": False,
        "face_detect": False,
        "face_count": 0,
        "ir_mode": controller.get_ir_mode(),
        "brightness": 50.0,
        "sharpen_index": 2,  # Medio por padrao
        "show_help": True,
        "status_msg": "",
        "status_until": 0.0,
        "take_screenshot": False,
        "timelapse": timelapse,
        "timelapse_active": True,
        "timelapse_frames": 0,
        "night_enhancer": night_enhancer,
        "night_mode": "OFF",
        "recorder": recorder,
    }

    print("Tapo C200 pronto (foco fundo ativado). Pressione F1 para ver os controles.")

    running = True
    drop_count = 0
    while running:
        ok, frame = camera.read()
        if not ok:
            drop_count += 1
            if drop_count > 30:
                print("Erro: stream perdido apos 30 tentativas.")
                break
            continue
        drop_count = 0

        # Atualizar posicao do sol
        state["sun"] = sun_tracker.update()

        # Filtro de nitidez por software (foco no fundo)
        _, sharpen_str = SHARPEN_LEVELS[state.get("sharpen_index", 0)]
        if sharpen_str > 0:
            frame = apply_sharpen(frame, sharpen_str)

        # Modo ceu noturno (IA: CLAHE + stacking + denoise)
        frame = night_enhancer.process(frame)

        # Timelapse (captura o frame processado, sem HUD)
        if timelapse.active:
            if timelapse.tick(frame):
                state["timelapse_frames"] = timelapse.frame_count

        # Gravacao continua (frame processado, sem HUD)
        recorder.feed(frame)

        # Deteccao facial
        if state["face_detect"]:
            faces = detect_faces(frame, face_cascade)
            state["face_count"] = len(faces)
            draw_faces(frame, faces)
        else:
            state["face_count"] = 0

        # HUD
        draw_hud(frame, state)

        # Mensagem de status temporaria
        now = time.time()
        if state["status_msg"] and now < state["status_until"]:
            h = frame.shape[0]
            cv2.putText(
                frame, state["status_msg"],
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )

        # Screenshot
        if state.get("take_screenshot"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"tapo_screenshot_{ts}.png"
            cv2.imwrite(fname, frame)
            state["status_msg"] = f"Salvo: {fname}"
            state["status_until"] = time.time() + 2.0
            state["take_screenshot"] = False

        cv2.imshow(WINDOW_NAME, frame)
        raw_key = cv2.waitKeyEx(1)

        # F11 (Linux X11 keycode = 0x500000 + 122 = 65480)
        if raw_key == 65480:
            toggle_fullscreen(state)
        # F1 (65470 on Linux)
        elif raw_key == 65470:
            state["show_help"] = not state.get("show_help", False)
        elif raw_key != -1:
            key = raw_key & 0xFF
            running = handle_key(key, controller, state)

    controller.stop()
    recorder.stop()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()