"""python -m mcp_tool_rag serve config.json [--host 127.0.0.1] [--port 8090]"""
import argparse
import logging


def main():
    ap = argparse.ArgumentParser(prog="python -m mcp_tool_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the MCP server")
    s.add_argument("config")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import uvicorn

    from .server import build_app, load_config
    uvicorn.run(build_app(load_config(a.config)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
