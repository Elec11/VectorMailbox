"""The one version string.

`pyproject.toml` reads this file with a static AST parse, so it stays a plain
literal assignment: nothing imported, nothing computed. `vectormailbox
--version`, `/api/status` and the git tag are the same number because they all
come from here."""

__version__ = "0.1.0"
