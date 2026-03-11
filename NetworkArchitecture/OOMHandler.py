"""
Minimal stub for OOMHandler to keep training scripts working.

The original project this code came from likely had a more
advanced OOM (out-of-memory) manager. In this repo, the
training scripts already implement their own try/except logic
around CUDA OOM, so we only need a no-op placeholder class
to satisfy the import.
"""

from contextlib import contextmanager
from typing import Callable, Optional


class OOMHandler:
    """
    Lightweight placeholder OOM handler.

    Usage in scripts can be:

        with OOMHandler():
            # forward / backward / optimizer.step()

    but in this implementation it simply yields and does not
    change behavior. Any OOM handling should be done where
    the try/except blocks already live in the training code.
    """

    def __init__(self, callback: Optional[Callable[[BaseException], None]] = None):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If an exception occurred and a callback is provided, call it
        if exc_val is not None and self.callback is not None:
            self.callback(exc_val)
        # Returning False so exceptions (including OOM) propagate normally
        return False


@contextmanager
def oom_guard(callback: Optional[Callable[[BaseException], None]] = None):
    """
    Convenience context manager with the same behavior as OOMHandler.
    """
    handler = OOMHandler(callback=callback)
    try:
        yield handler
    finally:
        # No special cleanup required in this minimal implementation
        pass

