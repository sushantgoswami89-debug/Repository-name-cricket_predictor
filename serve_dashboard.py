"""
Tiny local server so the dashboard (an HTML page) can read live_prediction.json
and auto-refresh, without needing any internet connection or complicated setup.

Run this, then open the printed localhost link in your browser.
"""
import http.server
import socketserver
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR  # this file lives at the project root
PORT = 8899

os.chdir(PROJECT_DIR)

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"\nDashboard running! Open this link in your browser:\n")
    print(f"    http://localhost:{PORT}/dashboard.html\n")
    print("Leave this Terminal window open while you use the dashboard.")
    print("Press Ctrl+C here to stop it.\n")
    httpd.serve_forever()
