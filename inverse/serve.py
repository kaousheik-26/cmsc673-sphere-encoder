"""
serve the results page locally. index.html holds only the page content (the
artifact host adds the document around it), so it is wrapped in a document
skeleton here: doctype, utf-8 and viewport. everything else, such as the
figures, is served as is.

usage:
    python inverse/serve.py              # http://localhost:8000
    python inverse/serve.py --port 8080
"""

import argparse
import functools
import http.server
import os.path as osp

HERE = osp.dirname(osp.abspath(__file__))
SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>body { margin: 0 } img { max-width: 100% } [hidden] { display: none !important }</style>
</head>
<body>
%PAGE%
</body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0].split("#")[0] not in ("/", "/index.html"):
            return super().do_GET()
        with open(osp.join(HERE, "index.html"), encoding="utf-8") as f:
            body = SKELETON.replace("%PAGE%", f.read()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # always the latest edit
        self.end_headers()
        self.wfile.write(body)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 keeps the page off the network")
    args = p.parse_args()

    server = http.server.ThreadingHTTPServer(
        (args.host, args.port), functools.partial(Handler, directory=HERE)
    )
    print(f"serving {HERE} at http://localhost:{args.port}  (ctrl+c to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
