"""Kiriko — Interface Streamlit para câmera Tapo C200.

Controle PTZ, ajuste de imagem, visão noturna, rastreamento solar,
detecção facial e melhoria de céu noturno via interface web.

Para uso local:
    streamlit run streamlit_app.py

Para Streamlit Cloud:
    Configure os secrets no painel do Streamlit Cloud (veja secrets.toml.example).
    A câmera precisa estar acessível pela internet (port-forward das portas 554 e ONVIF).
"""

import json
import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import streamlit as st

# ── Page config (deve ser o primeiro comando Streamlit) ──────────────────────

st.set_page_config(
    page_title="Kiriko — Tapo C200",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dependências opcionais ───────────────────────────────────────────────────

_HAS_EPHEM = False
try:
    import ephem
    _HAS_EPHEM = True
except ImportError:
    pass

_HAS_ONVIF = False
try:
    from onvif import ONVIFCamera
    _HAS_ONVIF = True
except ImportError:
    pass

# ── Constantes ───────────────────────────────────────────────────────────────

CONFIG_FILE = Path(__file__).resolve().parent / "tapo_config.json"
FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

OBSERVER_LAT = "-21.7946"
OBSERVER_LON = "-48.1766"
OBSERVER_ELEV = 664
BRT_OFFSET = timedelta(hours=3)

NIGHT_MODES = ("OFF", "Denoise", "Realce", "Stacking", "Realce+Stack")


# ── Configuração ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Carrega config de st.secrets → tapo_config.json → dict vazio."""
    try:
        return dict(st.secrets["tapo"])
    except (FileNotFoundError, KeyError):
        pass
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {}


def build_rtsp_url(ip: str, user: str, pw: str, stream: str) -> str:
    return f"rtsp://{quote(user, safe='')}:{quote(pw, safe='')}@{ip}:554/{stream}"


def get_wsdl_dir() -> str:
    """Localiza diretório WSDL do onvif-zeep."""
    try:
        import onvif as _m
        d = Path(_m.__file__).resolve().parent.parent / "wsdl"
        if d.is_dir():
            return str(d)
    except Exception:
        pass
    lib = Path(__file__).resolve().parent / ".venv" / "lib"
    for p in lib.glob("python*/site-packages/wsdl"):
        return str(p)
    return ""


# ── Controlador da câmera (ONVIF) ───────────────────────────────────────────

class CameraController:
    """Controle PTZ + Imaging via ONVIF."""

    IR_MODES = ("AUTO", "ON", "OFF")

    def __init__(self, ip: str, user: str, pw: str, port: int) -> None:
        self.cam = ONVIFCamera(ip, port, user, pw, get_wsdl_dir())
        self.media = self.cam.create_media_service()
        self.ptz = self.cam.create_ptz_service()
        self.profile = self.media.GetProfiles()[0]
        self.token = self.profile.token
        self.vs_token = self.profile.VideoSourceConfiguration.SourceToken

        req = self.ptz.create_type("GetConfigurationOptions")
        req.ConfigurationToken = self.profile.PTZConfiguration.token
        opts = self.ptz.GetConfigurationOptions(req)
        self.vel_space = opts.Spaces.ContinuousPanTiltVelocitySpace[0].URI

        self.imaging = self.cam.create_imaging_service()

    # PTZ

    def move(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        req = self.ptz.create_type("ContinuousMove")
        req.ProfileToken = self.token
        req.Velocity = {
            "PanTilt": {"x": pan, "y": tilt, "space": self.vel_space},
            "Zoom": {"x": zoom},
        }
        self.ptz.ContinuousMove(req)

    def pulse(self, pan: float, tilt: float, zoom: float = 0.0, dur: float = 0.22) -> None:
        ramp = (0.35, 0.7, 1.0, 0.7, 0.35)
        step = max(dur / len(ramp), 0.02)
        for f in ramp:
            self.move(pan * f, tilt * f, zoom * f)
            time.sleep(step)
        self.stop()

    def stop(self) -> None:
        req = self.ptz.create_type("Stop")
        req.ProfileToken = self.token
        req.PanTilt = True
        req.Zoom = True
        self.ptz.Stop(req)

    def home(self) -> None:
        try:
            req = self.ptz.create_type("GotoHomePosition")
            req.ProfileToken = self.token
            self.ptz.GotoHomePosition(req)
        except Exception:
            pass

    # Imaging

    def _get_settings(self):
        req = self.imaging.create_type("GetImagingSettings")
        req.VideoSourceToken = self.vs_token
        return self.imaging.GetImagingSettings(req)

    def _set(self, **kw) -> None:
        req = self.imaging.create_type("SetImagingSettings")
        req.VideoSourceToken = self.vs_token
        req.ImagingSettings = kw
        self.imaging.SetImagingSettings(req)

    def set_brightness(self, v: float) -> None:
        self._set(Brightness=float(max(0, min(100, v))))

    def set_contrast(self, v: float) -> None:
        self._set(Contrast=float(max(0, min(100, v))))

    def set_sharpness(self, v: float) -> None:
        self._set(Sharpness=float(max(0, min(100, v))))

    def set_saturation(self, v: float) -> None:
        self._set(ColorSaturation=float(max(0, min(100, v))))

    def set_ir(self, mode: str) -> None:
        try:
            self._set(IrCutFilter=mode)
        except Exception:
            pass

    def read_values(self) -> dict:
        try:
            s = self._get_settings()
            return {
                "bri": int(s.Brightness or 50),
                "con": int(s.Contrast or 50),
                "sha": int(s.Sharpness or 50),
                "sat": int(s.ColorSaturation or 50),
            }
        except Exception:
            return {"bri": 50, "con": 50, "sha": 50, "sat": 50}


# ── Rastreamento solar ───────────────────────────────────────────────────────

class SunTracker:
    def __init__(self) -> None:
        self._obs = ephem.Observer()
        self._obs.lat = OBSERVER_LAT
        self._obs.lon = OBSERVER_LON
        self._obs.elevation = OBSERVER_ELEV
        self._sun = ephem.Sun()
        self._cache: dict = {}
        self._last = 0.0

    def update(self) -> dict:
        now = time.time()
        if now - self._last < 10:
            return self._cache
        self._last = now

        utc_now = datetime.now(UTC)
        self._obs.date = utc_now
        self._sun.compute(self._obs)

        elev = math.degrees(self._sun.alt)
        azim = math.degrees(self._sun.az)

        try:
            start = utc_now.replace(hour=3, minute=0, second=0, microsecond=0)
            if utc_now.hour < 3:
                start -= timedelta(days=1)
            o = ephem.Observer()
            o.lat = OBSERVER_LAT
            o.lon = OBSERVER_LON
            o.elevation = OBSERVER_ELEV
            o.date = start
            s = ephem.Sun()
            nascer = (o.next_rising(s).datetime() - BRT_OFFSET).strftime("%H:%M")
            por = (o.next_setting(s).datetime() - BRT_OFFSET).strftime("%H:%M")
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            nascer = "--:--"
            por = "--:--"

        self._cache = {
            "elevacao": elev,
            "azimute": azim,
            "nascer": nascer,
            "por": por,
            "is_day": elev > 0,
        }
        return self._cache


# ── Melhoria de céu noturno ──────────────────────────────────────────────────

class NightEnhancer:
    STACK_SIZE = 12

    def __init__(self) -> None:
        self._idx = 0
        self._stack: list[np.ndarray] = []
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._gamma_lut = np.array(
            [((i / 255.0) ** 0.75) * 255 for i in range(256)], dtype=np.uint8
        )

    @property
    def mode(self) -> str:
        return NIGHT_MODES[self._idx]

    def set_mode(self, idx: int) -> None:
        self._idx = idx % len(NIGHT_MODES)
        if "Stack" not in self.mode:
            self._stack.clear()

    def process(self, frame: np.ndarray) -> np.ndarray:
        m = self.mode
        if m == "OFF":
            return frame

        result = frame

        if "Stack" in m:
            self._stack.append(frame.astype(np.float32))
            if len(self._stack) > self.STACK_SIZE:
                self._stack.pop(0)
            if len(self._stack) >= 2:
                result = np.mean(self._stack, axis=0).astype(np.uint8)

        result = cv2.fastNlMeansDenoisingColored(result, None, 12, 12, 7, 21)

        if "Realce" in m:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = self._clahe.apply(l_ch)
            result = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
            result = cv2.LUT(result, self._gamma_lut)

        return result


# ── Recursos cacheados ───────────────────────────────────────────────────────

@st.cache_resource
def _init_ctrl(ip: str, user: str, pw: str, port: int):
    """Inicializa controlador ONVIF. Retorna objeto ou string de erro."""
    if not _HAS_ONVIF:
        return "Pacote onvif-zeep não instalado"
    try:
        return CameraController(ip, user, pw, port)
    except Exception as e:
        return str(e)


@st.cache_resource
def _init_cap(url: str):
    """Inicializa captura RTSP."""
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        return None
    return cap


@st.cache_resource
def _init_sun():
    return SunTracker() if _HAS_EPHEM else None


@st.cache_resource
def _init_enhancer():
    return NightEnhancer()


@st.cache_resource
def _init_cascade():
    return cv2.CascadeClassifier(FACE_CASCADE)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ctrl():
    """Retorna controlador ONVIF atual ou None."""
    if not st.session_state.get("connected"):
        return None
    c = _init_ctrl(
        st.session_state.cam_ip,
        st.session_state.cam_user,
        st.session_state.cam_pass,
        st.session_state.cam_port,
    )
    return c if isinstance(c, CameraController) else None


def _cap():
    """Retorna VideoCapture atual ou None."""
    if not st.session_state.get("connected"):
        return None
    url = build_rtsp_url(
        st.session_state.cam_ip,
        st.session_state.cam_user,
        st.session_state.cam_pass,
        st.session_state.cam_stream,
    )
    return _init_cap(url)


def _grab():
    """Captura o frame mais recente da câmera."""
    cap = _cap()
    if cap is None:
        return None
    # Flush do buffer para pegar frame recente
    cap.grab()
    ok, frame = cap.read()
    if not ok:
        # Tenta reconectar
        _init_cap.clear()
        cap = _cap()
        if cap is None:
            return None
        ok, frame = cap.read()
    return frame if ok else None


# ── Callbacks PTZ ────────────────────────────────────────────────────────────

def _ptz(pan: float, tilt: float, zoom: float = 0.0) -> None:
    c = _ctrl()
    if c:
        s = st.session_state.get("ptz_speed", 0.5)
        c.pulse(pan * s, tilt * s, zoom * s, dur=0.22)


def _ptz_stop() -> None:
    c = _ctrl()
    if c:
        c.stop()


def _ptz_home() -> None:
    c = _ctrl()
    if c:
        c.home()


# ── Callbacks Imaging ────────────────────────────────────────────────────────

def _on_brightness():
    c = _ctrl()
    if c:
        c.set_brightness(st.session_state.sl_bri)


def _on_contrast():
    c = _ctrl()
    if c:
        c.set_contrast(st.session_state.sl_con)


def _on_sharpness():
    c = _ctrl()
    if c:
        c.set_sharpness(st.session_state.sl_sha)


def _on_saturation():
    c = _ctrl()
    if c:
        c.set_saturation(st.session_state.sl_sat)


def _on_ir():
    c = _ctrl()
    if c:
        c.set_ir(st.session_state.sel_ir)


# ── App principal ────────────────────────────────────────────────────────────

def main() -> None:
    # Inicializar session state
    if "cam_ip" not in st.session_state:
        cfg = load_config()
        st.session_state.update({
            "cam_ip": str(cfg.get("ip", "")),
            "cam_user": str(cfg.get("user", "")),
            "cam_pass": str(cfg.get("password", "")),
            "cam_stream": str(cfg.get("stream", "stream1")),
            "cam_port": int(cfg.get("onvif_port", 2020)),
            "connected": False,
            "ptz_speed": float(cfg.get("speed", 0.5)),
            "sl_bri": 50,
            "sl_con": 50,
            "sl_sha": 50,
            "sl_sat": 50,
            "sel_ir": "AUTO",
            "night_idx": 0,
            "face_on": False,
        })

    # ── Sidebar ──────────────────────────────────────────────────────────

    with st.sidebar:
        st.markdown("# 🎥 Kiriko")

        # Conexão
        with st.expander("🔌 Conexão", expanded=not st.session_state.connected):
            st.text_input("IP da Câmera", key="cam_ip")
            st.text_input("Usuário", key="cam_user")
            st.text_input("Senha", type="password", key="cam_pass")
            c1, c2 = st.columns(2)
            with c1:
                streams = ["stream1", "stream2"]
                idx = streams.index(st.session_state.cam_stream) if st.session_state.cam_stream in streams else 0
                st.selectbox("Stream", streams, index=idx, key="cam_stream")
            with c2:
                st.number_input("Porta ONVIF", min_value=1, max_value=65535, key="cam_port")

            if not st.session_state.connected:
                if st.button("▶ Conectar", type="primary", use_container_width=True):
                    if not st.session_state.cam_ip:
                        st.error("Informe o IP da câmera")
                    else:
                        with st.spinner("Conectando à câmera..."):
                            result = _init_ctrl(
                                st.session_state.cam_ip,
                                st.session_state.cam_user,
                                st.session_state.cam_pass,
                                st.session_state.cam_port,
                            )
                            if isinstance(result, str):
                                st.error(f"Erro ONVIF: {result}")
                            else:
                                url = build_rtsp_url(
                                    st.session_state.cam_ip,
                                    st.session_state.cam_user,
                                    st.session_state.cam_pass,
                                    st.session_state.cam_stream,
                                )
                                cap = _init_cap(url)
                                if cap is None:
                                    st.error("Erro ao abrir stream RTSP")
                                else:
                                    # Ler valores atuais da câmera
                                    vals = result.read_values()
                                    st.session_state.sl_bri = vals["bri"]
                                    st.session_state.sl_con = vals["con"]
                                    st.session_state.sl_sha = vals["sha"]
                                    st.session_state.sl_sat = vals["sat"]
                                    st.session_state.connected = True
                                    st.rerun()
            else:
                st.success("Conectado")
                if st.button("⏹ Desconectar", use_container_width=True):
                    c = _ctrl()
                    if c:
                        c.stop()
                    st.session_state.connected = False
                    _init_ctrl.clear()
                    _init_cap.clear()
                    st.rerun()

        # Controles só aparecem quando conectado
        if st.session_state.connected:
            st.divider()

            # Velocidade PTZ
            st.slider("🕹️ Velocidade PTZ", 0.1, 1.0, step=0.1, key="ptz_speed")

            st.divider()

            # Ajustes de imagem
            st.markdown("### 🎚️ Imagem")
            st.slider("Brilho", 0, 100, key="sl_bri", on_change=_on_brightness)
            st.slider("Contraste", 0, 100, key="sl_con", on_change=_on_contrast)
            st.slider("Nitidez", 0, 100, key="sl_sha", on_change=_on_sharpness)
            st.slider("Saturação", 0, 100, key="sl_sat", on_change=_on_saturation)

            st.divider()

            # Visão noturna
            st.markdown("### 🌙 Modo Noturno")
            st.selectbox(
                "Visão Noturna (IR)",
                CameraController.IR_MODES,
                key="sel_ir",
                on_change=_on_ir,
            )
            st.selectbox(
                "Céu Noturno (IA)",
                range(len(NIGHT_MODES)),
                format_func=lambda i: NIGHT_MODES[i],
                key="night_idx",
            )

            st.divider()

            # Detecção facial
            st.checkbox("👤 Detecção Facial", key="face_on")

    # ── Conteúdo principal ───────────────────────────────────────────────

    st.markdown(
        "<h1 style='margin-bottom:0'>📷 Kiriko <small style='font-size:0.4em;color:#888'>"
        "Tapo C200</small></h1>",
        unsafe_allow_html=True,
    )

    if not st.session_state.connected:
        st.info("👈 Configure a conexão na barra lateral e clique em **Conectar**")
        st.markdown("""
### Como usar

**Localmente:**
```bash
pip install streamlit opencv-python-headless numpy onvif-zeep ephem
streamlit run streamlit_app.py
```

**Streamlit Cloud:**
1. Suba o código para um repositório GitHub
2. Conecte ao [Streamlit Cloud](https://share.streamlit.io)
3. Configure os **Secrets** no painel:

```toml
[tapo]
ip = "SEU_IP_PUBLICO"
user = "usuario"
password = "senha"
stream = "stream1"
onvif_port = 2020
```

4. Faça **port-forward** no roteador:
   - Porta **554** (RTSP)
   - Porta **2020** (ONVIF)
        """)
        return

    # Layout: feed + controles
    feed_col, ctrl_col = st.columns([3, 1], gap="medium")

    with ctrl_col:
        # PTZ
        st.markdown("#### 🕹️ Movimento")

        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.button("↖", on_click=_ptz, args=(-1, 1), key="p_ul", use_container_width=True)
        r1c2.button("⬆", on_click=_ptz, args=(0, 1), key="p_u", use_container_width=True)
        r1c3.button("↗", on_click=_ptz, args=(1, 1), key="p_ur", use_container_width=True)

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.button("⬅", on_click=_ptz, args=(-1, 0), key="p_l", use_container_width=True)
        r2c2.button("⏹", on_click=_ptz_stop, key="p_s", use_container_width=True)
        r2c3.button("➡", on_click=_ptz, args=(1, 0), key="p_r", use_container_width=True)

        r3c1, r3c2, r3c3 = st.columns(3)
        r3c1.button("↙", on_click=_ptz, args=(-1, -1), key="p_dl", use_container_width=True)
        r3c2.button("⬇", on_click=_ptz, args=(0, -1), key="p_d", use_container_width=True)
        r3c3.button("↘", on_click=_ptz, args=(1, -1), key="p_dr", use_container_width=True)

        # Zoom
        st.markdown("#### 🔍 Zoom")
        zc1, zc2 = st.columns(2)
        zc1.button("➕ In", on_click=_ptz, args=(0, 0, 1), key="z_in", use_container_width=True)
        zc2.button("➖ Out", on_click=_ptz, args=(0, 0, -1), key="z_out", use_container_width=True)

        # Home
        st.button("🏠 Home", on_click=_ptz_home, key="p_home", use_container_width=True)

        st.divider()

        # Screenshot
        st.markdown("#### 📸 Captura")
        snap = st.session_state.get("_snap")
        if snap is not None:
            _, buf = cv2.imencode(".png", snap)
            ts = time.strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "📥 Baixar Screenshot",
                data=buf.tobytes(),
                file_name=f"kiriko_{ts}.png",
                mime="image/png",
                use_container_width=True,
            )

    # Feed da câmera (auto-refresh a cada 2s)
    with feed_col:
        @st.fragment(run_every=2)
        def live_feed():
            frame = _grab()
            if frame is None:
                st.warning("📡 Sem sinal da câmera — verifique a conexão")
                return

            # Céu noturno (IA)
            enhancer = _init_enhancer()
            enhancer.set_mode(st.session_state.get("night_idx", 0))
            frame = enhancer.process(frame)

            # Detecção facial
            if st.session_state.get("face_on", False):
                cascade = _init_cascade()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40)
                )
                if faces is not None and len(faces) > 0:
                    for x, y, w, h in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            frame, "Face", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                        )
                    st.caption(f"👤 {len(faces)} face(s) detectada(s)")

            # Exibir frame
            st.image(frame, channels="BGR", use_container_width=True)

            # Salvar para screenshot
            st.session_state["_snap"] = frame

        live_feed()

    # Informações solares
    sun_tracker = _init_sun()
    if sun_tracker:
        sun = sun_tracker.update()
        if sun:
            st.divider()
            st.markdown("### ☀️ Posição Solar — Araraquara")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Estado", "☀️ Dia" if sun["is_day"] else "🌙 Noite")
            c2.metric("Elevação", f"{sun['elevacao']:.1f}°")
            c3.metric("Azimute", f"{sun['azimute']:.1f}°")
            c4.metric("Nascer", sun["nascer"])
            c5.metric("Pôr do Sol", sun["por"])


if __name__ == "__main__":
    main()
