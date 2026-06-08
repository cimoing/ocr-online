"""Convenience launcher: python run.py  ->  http://127.0.0.1:8000"""
import os
import sys
from pathlib import Path

# Make the launcher independent of the current working directory so `app` is
# importable and static files resolve no matter where it is started from.
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
