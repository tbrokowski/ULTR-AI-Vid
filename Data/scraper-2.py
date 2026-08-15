"""Backward-compatible entry point for the Butterfly scraper.

The implementation lives in ``butterfly_scraper.py``. Credentials are
intentionally never stored in source code.
"""

from butterfly_scraper import main


if __name__ == "__main__":
    raise SystemExit(main())
