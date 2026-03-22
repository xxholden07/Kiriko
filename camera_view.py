import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

import cv2


CONFIG_FILE = Path(__file__).resolve().parent / "tapo_config.json"


def build_rtsp_url(ip: str, user: str, password: str, stream: str) -> str:
    # Stream principal da Tapo C200 geralmente e "/stream1".
    safe_user = quote(user, safe="")
    safe_password = quote(password, safe="")
    return f"rtsp://{safe_user}:{safe_password}@{ip}:554/{stream}"


def load_config() -> dict[str, str]:
    config: dict[str, str] = {}
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    env_map = {
        "ip": os.getenv("TAPO_IP"),
        "user": os.getenv("TAPO_USER"),
        "password": os.getenv("TAPO_PASSWORD"),
        "stream": os.getenv("TAPO_STREAM"),
        "url": os.getenv("TAPO_URL"),
    }
    for key, value in env_map.items():
        if value not in (None, ""):
            config[key] = value

    return config


def parse_args() -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description="Visualizar stream RTSP (Tapo C200)")
    parser.add_argument("--url", default=config.get("url"), help="URL RTSP completa")
    parser.add_argument("--ip", default=config.get("ip"), help="IP da camera (ex: 192.168.0.120)")
    parser.add_argument("--user", default=config.get("user"), help="Usuario da camera")
    parser.add_argument("--password", default=config.get("password"), help="Senha da camera")
    parser.add_argument(
        "--stream",
        default=config.get("stream", "stream1"),
        choices=["stream1", "stream2"],
        help="Qual stream usar (padrao: stream1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.url:
        source = args.url
    else:
        if not (args.ip and args.user and args.password):
            print("Erro: informe --url ou (--ip --user --password).")
            return
        source = build_rtsp_url(args.ip, args.user, args.password, args.stream)

    camera = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    if not camera.isOpened():
        print("Erro: nao foi possivel abrir o stream da camera.")
        print(f"Fonte usada: {source}")
        return

    print("Stream iniciado. Pressione 'q' para sair.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Erro: nao foi possivel capturar frame do stream.")
            break

        cv2.imshow("Tapo C200 - Visualizacao", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
