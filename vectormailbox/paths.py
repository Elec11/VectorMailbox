"""Where the files live.

Two kinds of file, and conflating them is what breaks an installed app:

* **resources** — shipped with the app, read-only: `web/`.
* **data** — the user's: `config.json` (which holds the Gmail app password), the
  index, the preview sidecar beside it, and the log of the last index run.

In a checkout both are this folder, which is why it was never a question while
the app only ran from here. Installed with `uv` or `pipx` there is no *here*:
the resources sit in site-packages, and the current directory is whatever room
of the filesystem the person typing the command happened to be standing in. A
relative `db_path` resolved against *that* hands them a second, empty mailbox
every time they launch from a different folder - and a `config.json` written
into a Downloads directory they will never find again.

So: resources come from the package, data comes from the per-user data
directory, and a relative path in a config file means *beside that config file*.
One rule covers every search: **a location you named beats a location we
search**, and the last thing we search is the current directory - because that
is a checkout, and a checkout is the case this app grew up in.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("vectormailbox.paths")

#: What `--config` carries when nobody said otherwise, and what the CLI has
#: always passed. Treated as "unspecified" by `config_path`, so the precedence
#: chain below still applies to a plain `vectormailbox gui`.
DEFAULT_CONFIG_NAME = "config.json"

#: Where everything user-writable goes. Also the portable-mode switch: point it
#: at a folder next to the app and the whole mailbox travels with it.
HOME_ENV = "VECTORMAILBOX_HOME"

#: One config file, named explicitly, skipping the search.
CONFIG_ENV = "VECTORMAILBOX_CONFIG"

#: The folder's name inside the per-user data root.
APP_NAME = "vectormailbox"


def resource_root() -> Path:
    """The directory the shipped files live in, which is this module's folder.

    `web/index.html`, `web/app.css`, `web/app.js` and `web/mailhtml.js` are
    package data declared in pyproject.toml, so they land beside this file
    whether this is a checkout or site-packages.

    Not `importlib.resources.as_file`: `StaticFiles` is handed a directory and
    streams from it, so a distribution zipped into a wheel would need a real
    directory extracted anyway. That case is not worth carrying a branch for;
    an unpacked install on disk is the only one this app runs in.
    """
    return Path(__file__).resolve().parent


def _default_data_root() -> str:
    """The platform's data folder, as a string.

    Built with `os.path` and no `Path`, on purpose: this is the one place the
    three platforms disagree, and the disagreement is tested on whichever of
    them the suite happens to run on, where `Path` refuses to construct a
    foreign flavour.
    """
    home = os.path.expanduser("~")
    if os.name == "nt":
        root = (os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
                or os.path.join(home, "AppData", "Roaming"))
    elif sys.platform == "darwin":
        root = os.path.join(home, "Library", "Application Support")
    else:
        root = (os.environ.get("XDG_DATA_HOME", "").strip()
                or os.path.join(home, ".local", "share"))
    return os.path.join(root, APP_NAME)


def data_dir(create: bool = False) -> Path:
    """The user's own folder for config, index, sidecars and logs.

    Per-user by design: this app serves a whole mailbox with no login, so it
    should not put that mailbox somewhere a second account on the same machine
    can read, and it should not need administrator rights just to start.
    """
    override = os.environ.get(HOME_ENV, "").strip()
    base = Path(override).expanduser() if override else Path(_default_data_root())
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def config_path(value: str = "") -> Path:
    """Which `config.json` this run reads, and where a new one gets written.

    Order, first hit wins:

    1. `--config anything.json`: a name that is not the default is an
       instruction, so it is taken literally even if it does not exist yet
       (loading a missing file has always meant "defaults and environment").
    2. `$VECTORMAILBOX_CONFIG`, for a system that keeps its secrets elsewhere.
    3. `$VECTORMAILBOX_HOME/config.json`, when it is there. Naming the folder is
       a statement about where this mailbox lives, so its own config outranks
       the search below.
    4. `./config.json`: a checkout, which is the case every existing measurement
       and README line in this repo depends on.
    5. `<data dir>/config.json`, which is also where a *new* file belongs -
       never the current directory, because under `uvx` that is wherever the
       shell happened to be standing.

    `data_dir()` is not created here: asking which config file to read must not
    have a side effect. Writing one creates the folder when there is something
    to store.
    """
    named = str(value or "").strip()
    if named and named != DEFAULT_CONFIG_NAME:
        return Path(named).expanduser()
    env = os.environ.get(CONFIG_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    home_cfg = data_dir() / DEFAULT_CONFIG_NAME
    if os.environ.get(HOME_ENV, "").strip() and home_cfg.exists():
        return home_cfg
    here = Path.cwd() / DEFAULT_CONFIG_NAME
    if here.exists():
        return here
    return home_cfg


def beside(base: Path, value) -> Path:
    """A relative path means *beside `base`*; an absolute one means what it says."""
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else base / p


def from_cwd(value) -> Path:
    """Resolve a path the way a shell would: relative to where you are standing.

    For `--db baseline-legacy.db`, typed by a person who is standing next to
    that file. A path that came out of a config file is not this case: see
    `beside`.
    """
    return beside(Path.cwd(), value)


def index_log(db_path) -> Path:
    """The file an index run writes its progress to.

    Derived from the database it describes, under its own name, exactly like
    `media_db_path` derives the preview sidecar: `mailbox.db` gets
    `mailbox.index.log`, so `--db baseline-legacy.db` measures a frozen copy
    without appending to the live mailbox's run history - and the file lands
    beside the index it belongs to rather than wherever the GUI was launched.
    """
    p = Path(str(db_path)).expanduser()
    return p.with_suffix(".index.log")
