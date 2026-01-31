from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class NLIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self.send_error(411, "Content-Length required")
            return

        try:
            length = int(length_header)
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return

        body = self.rfile.read(length)
        try:
            json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        payload = {"label": "NEUTRAL", "confidence": 0.5}
        response = json.dumps(payload).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    server = HTTPServer(("127.0.0.1", 9000), NLIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
