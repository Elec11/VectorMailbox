"""The public entry point: `vectormailbox …` and `python -m vectormailbox …`.

The command tree itself lives in `mailbox.py`, beside the `cmd_*` functions it
dispatches to; splitting the parser away from the commands it calls would move
lines without moving responsibility. This module is the seam the console script
points at, so the entry point has a stable name even if that ever changes."""

from .mailbox import main

__all__ = ["main"]
