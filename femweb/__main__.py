import argparse
import threading
import webbrowser

import uvicorn


def main():
    p = argparse.ArgumentParser(prog="femweb")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    url = f"http://{a.host}:{a.port}"
    print(f"FEM-LLM-web: {url}")
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run("femweb.server:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
