"""`python -m vectormailbox …` — the same command tree as the console script.

This form is also how the GUI starts an index run once the app is installed:
`sys.executable -m vectormailbox index`. An installed app has no `mailbox.py`
file to point a child process at, but there is always a package importable by
the interpreter that is already running the GUI — which is the interpreter that
has the dependencies."""

from .cli import main

if __name__ == "__main__":
    main()
