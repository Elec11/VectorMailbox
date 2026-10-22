"""Checkout shim: `python gui.py --port 8765` still works.

The module is `vectormailbox/gui.py` now; see the note in `mailbox.py` beside
this file for why the package owns the real copy. Nothing here runs on import —
only when this file is the one Python was told to start.
"""

if __name__ == "__main__":
    from vectormailbox.gui import main

    main()
