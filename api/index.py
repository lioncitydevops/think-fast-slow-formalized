import sys
import os
from pathlib import Path

# Add project root to sys.path for Vercel Serverless Function
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from think_fast_slow.web.server import app

# Export ASGI handler for Vercel @vercel/python runtime
handler = app
app = app
