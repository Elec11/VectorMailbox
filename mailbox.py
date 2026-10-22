"""Checkout shim: `python mailbox.py index` still works.

The code lives in `vectormailbox/mailbox.py` now — it had to move, because a
directory of loose scripts is not installable and `uvx`/`pipx` install
packages. This file keeps the old command alive for a script, a shell history
or a README line you already typed a hundred times. What goes forward is
`vectormailbox …` (installed) or `python -m vectormailbox …` (checkout).

Note what this file's *name* does inside a checkout: it shadows the stdlib
`mailbox` module for anything running from here, exactly as it always did. That
is why the package keeps its copy one level deeper, where `import mailbox` is
`vectormailbox.mailbox` and belongs to nobody else.
"""

if __name__ == "__main__":
    from vectormailbox.mailbox import main

    main()
