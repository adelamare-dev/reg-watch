"""Entry point: `python -m mcp_server`.

Defaults to Streamable HTTP so the agent can reach it as a service. Pass
`--stdio` to run it as a subprocess instead, which is how an MCP host such as
Claude Desktop launches it.
"""

from __future__ import annotations

import argparse
import os

from mcp_server.server import build_default_server

DEFAULT_PORT = 8300


def main() -> None:
    parser = argparse.ArgumentParser(prog="mcp_server", description=__doc__)
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="run over stdio instead of HTTP, for a local MCP host",
    )
    parser.add_argument(
        "--host", default=os.environ.get("MCP_HOST", "127.0.0.1"), help="bind address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MCP_PORT", DEFAULT_PORT)),
        help="port to listen on",
    )
    args = parser.parse_args()

    server = build_default_server()
    if args.stdio:
        server.run()
    else:
        server.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
