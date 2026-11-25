"""Cross-platform (Windows & Ubuntu) launcher for the UARCap Q2 Local Blind Web Annotation Tool."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uarcap_q2.annotation.web_app import create_annotation_server


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch UARCap Q2 Local Blind Web Annotation UI")
    ap.add_argument("--bundle", type=Path, required=True, help="Path to blind_tasks_e2_*.json or blind_tasks_e3_*.json")
    ap.add_argument("--save-dir", type=Path, required=True, help="Directory for per-annotator auto-saved progress")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    server, store = create_annotation_server(args.bundle, args.save_dir, host=args.host, port=args.port)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"[UARCap Q2 Annotator] Annotator={store.annotator_id} | Tasks={len(store.bundle.get('tasks', []))}")
    print(f"[UARCap Q2 Annotator] Auto-save dir: {store.save_dir}")
    print(f"[UARCap Q2 Annotator] Listening at: {url} (Ctrl+C to stop safely; progress is saved on every click)")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[UARCap Q2 Annotator] Stopped cleanly. All clicks were saved atomically.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
