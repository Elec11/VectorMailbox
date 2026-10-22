"""`vectormailbox doctor` — the checks that have to be true before the first run.

This is a command rather than a paragraph in a README because every one of these
failures looks like a broken program from the inside. An endpoint on the wrong
port answers with a connection refused the GUI shows as a grey dot; a `dim` that
does not match the model returns results that are confidently wrong; a config file
in the folder you happened to run from is read by one command and not the next.
None of those say what they are. This does.

Three rules hold here.

**Everything is read-only.** The checks log in to Gmail and EXAMINE a folder, ask
the embedding endpoint one question, and open the index with `mode=ro`. There is
no write path in this module, and the one file it could name — the config — it
only stats. A person worried about what a diagnostic will do to their mailbox
should be able to read this file and see the absence.

**The verdicts come from the same probes the GUI runs.** `endpoint` and `gmail`
call the functions behind the settings card's *test* button, so a green light in
the terminal and a red one in the page cannot disagree, and the two surfaces do
not grow separate definitions of "works".

**A check answers `(status, detail)`; what it was checking is added when the
table is assembled.** Each check would otherwise repeat its own name in every
return, and the order of the table — which is the reading order of a diagnosis —
would be scattered across the module instead of being the one list below.

One limit worth stating: `numpy`, `requests` and `tqdm` are imported before this
command can start, so a missing one of those is a traceback rather than a line in
this table. Everything optional enough to go missing quietly — the sanitiser, the
thumbnails, the GUI's own two packages, the endpoint, the mailbox — gets a line
and a sentence about what to do.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import paths
from ._version import __version__

OK, WARN, FAIL = "ok", "warn", "fail"
_WORTH = {OK: 0, WARN: 1, FAIL: 2}

#: The floor `pyproject.toml` declares. Kept here so the check and the metadata
#: can be pinned against each other by a test, rather than drifting apart.
MIN_PYTHON = (3, 10)

#: Import-time dependencies and the sentence each one is load-bearing for. A
#: missing `PIL` costs you thumbnails; a missing `nh3` costs you the guarantee
#: that a stranger's HTML cannot put a script on your screen — so they are not the
#: same warning and do not get the same colour.
MODULES = (
    ("numpy", FAIL, "the vector store: cosine over float32 blobs"),
    ("PIL", WARN, "thumbnail copies of images and documents"),
    ("fastapi", FAIL, "the browser page and its /api routes"),
    ("uvicorn", FAIL, "serving the page on 127.0.0.1"),
)

Verdict = tuple   # (status, detail)
Row = tuple       # (status, subject, detail) — a verdict plus what it names


def check_python() -> Verdict:
    want = ".".join(str(x) for x in MIN_PYTHON)
    got = ".".join(str(x) for x in sys.version_info[:3])
    if sys.version_info[:2] < MIN_PYTHON:
        return (FAIL, f"{got} is older than the {want}+ this needs")
    return (OK, got)


def check_package() -> Verdict:
    """Which copy answered, and whether it was installed or cloned.

    `uv tool install`/`pipx` put one copy on PATH and a checkout is a second one;
    when the two disagree, the folder in this line is the fact that ends the
    argument.
    """
    here = Path(__file__).parent
    try:
        from importlib.metadata import version

        installed = version("vectormailbox")
    except Exception:
        return (WARN, f"{__version__} at {here} - no installed metadata: this is a"
                      " source checkout, not an installed command")
    note = f"{installed} at {here}"
    if installed != __version__:
        return (WARN, f"{note} - the metadata and the code disagree")
    return (OK, note)


def check_modules() -> Iterable[tuple]:
    """`(name, verdict)` per dependency: the name is the table's, not the check's."""
    import importlib

    for mod, level, why in MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            yield mod, (level, f"not importable ({why}): {exc}")
        else:
            yield mod, (OK, why)


def check_sanitizer() -> Verdict:
    """The one dependency tested instead of merely imported.

    `nh3` is what stands between mail written by a stranger and the page you are
    reading. An import can succeed on a build that strips too little, and that
    failure is invisible until it is a problem, so the check asks it to do the job
    on the smallest case that matters: a link with a script handler, an image with
    a handler, and a `javascript:` address.
    """
    try:
        import nh3
    except Exception as exc:
        return (FAIL, f"nh3 not importable: {exc}")
    try:
        clean = nh3.clean('<a href="javascript:alert(1)" onclick="y()">t'
                          '<img src="z" onerror="q">')
    except Exception as exc:
        return (FAIL, f"nh3 could not clean a sample: {exc}")
    left = [w for w in ("onclick", "onerror", "javascript") if w in clean]
    if left:
        return (FAIL, f"kept {', '.join(left)} in {clean!r}")
    return (OK, "drops script handlers and javascript: links")


def check_config(cfg: dict) -> Verdict:
    path = Path(str(cfg.get("config_path") or paths.config_path()))
    if not path.exists():
        return (WARN, f"nothing at {path} yet - the settings card in the browser"
                      " writes it, and until then the defaults run")
    info = path.stat()
    note = f"{path} - {info.st_size} bytes"
    if os.name != "posix":
        return (OK, note + " - modes are advisory on this OS")
    if stat.S_IMODE(info.st_mode) & 0o077:
        return (WARN, f"{note} - readable by others, and it holds an app password"
                      " - chmod 600")
    return (OK, note + " - mode " + oct(stat.S_IMODE(info.st_mode))[-3:])


def check_data_dir(cfg: dict) -> Verdict:
    """Writable, proved by writing.

    `os.access` reports what the metadata claims; a roaming profile or a folder
    owned by an installer can claim one thing and refuse the write. The probe
    creates a file and lets `NamedTemporaryFile` remove it, so the check leaves
    nothing behind in the folder it is checking.
    """
    folder = paths.data_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(folder), prefix=".doctor-"):
            pass
    except OSError as exc:
        return (FAIL, f"cannot write in {folder}: {exc}")
    return (OK, str(folder))


def check_index(cfg: dict) -> Verdict:
    db = str(cfg.get("db_path") or "")
    path = Path(db)
    if not db:
        return (FAIL, "the config has no db_path")
    if not path.exists():
        return (WARN, f"nothing at {db} yet - the refresh button reads the mailbox"
                      " and writes this file")
    try:
        con = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return (FAIL, f"{db} will not open: {exc}")
    try:
        def count(table: str) -> Optional[int]:
            # The table names are this program's schema, never user input.
            try:
                return con.execute(f"select count(*) from {table}").fetchone()[0]
            except sqlite3.Error:
                return None

        messages, chunks, vectors = (count("messages"), count("chunks"),
                                     count("embeddings"))
        if messages is None:
            return (FAIL, f"{db} is not a vectormailbox index")
        if not messages:
            return (WARN, f"{db} is empty - press refresh to read the mailbox")
        stored = con.execute("select dim, count(*) from embeddings"
                             " group by dim order by 2 desc limit 1").fetchone()
        want = int(cfg.get("dim") or 0)
        if stored and want and int(stored[0]) != want:
            return (FAIL, f"{db} holds {stored[0]}-dimensional vectors in"
                          f" {stored[1]} rows and the config says dim={want}: one of"
                          " the two is not the model you indexed with")
        state = dict(con.execute("select key, value from index_state").fetchall())
        last = str(state.get("last_indexed_date") or "")[:10]
        bits = [f"{messages:,} mail", f"{chunks:,} chunks", f"{vectors:,} vectors"]
        if stored:
            bits.append(f"dim {stored[0]}")
        bits.append(f"indexed through {last}" if last else "never fully indexed")
        # No vectors is not a broken index: the lexical leg still answers, so this
        # is an index half-built, which is a warning and not a failure.
        return (OK if vectors else WARN, f"{db} - " + " - ".join(bits))
    finally:
        con.close()


def _probe(fn: Callable[[], dict], idle: str) -> Verdict:
    """One of the two network checks, both of which are the GUI's own probes.

    A probe that could not run is a `warn` and not a `fail`: no fastapi in this
    interpreter is not a broken mailbox, and a `fail` here would send someone to
    retype a Gmail password over a missing wheel.
    """
    try:
        out = fn()
    except Exception as exc:
        return (WARN, f"skipped: {exc}")
    if out.get("ok"):
        return (OK, out.get("detail") or idle)
    return (FAIL, out.get("detail") or "no answer")


def check_endpoint(cfg: dict) -> Verdict:
    from .gui import test_embedding

    return _probe(lambda: test_embedding(cfg), "answered")


def check_gmail(cfg: dict) -> Verdict:
    from .gui import test_gmail

    return _probe(lambda: test_gmail(cfg), "logged in")


def checks(cfg: dict, offline: bool = False) -> list:
    """The table, in the order a first run should read it: the machine, then the
    files, then the two things that live on a network.

    The first line that is not `ok` is normally the whole problem, so nothing
    expensive goes before something cheap, and nothing that needs a network goes
    before something that only needs a disk.
    """
    named: list = [("python", check_python()), ("package", check_package())]
    named += list(check_modules())
    named += [("sanitiser", check_sanitizer()),
              ("config", check_config(cfg)),
              ("data folder", check_data_dir(cfg)),
              ("index", check_index(cfg))]
    if not offline:
        named += [("endpoint", check_endpoint(cfg)), ("gmail", check_gmail(cfg))]
    return [(str(status), str(subject), str(detail))
            for subject, (status, detail) in named]


def report(rows: Iterable[Row], write: Callable[[str], None] = print) -> int:
    """Print the table and return the exit code: 1 if anything failed.

    ASCII throughout, labels included: `main()` reconfigures stdout with
    errors="replace" because a Windows console is often cp1252, so an em dash in
    this output arrives as a `?` — in the one place this app is read on a terminal.
    """
    rows = list(rows)
    width = max(len(str(b)) for _a, b, _c in rows) + 2
    for status, subject, detail in rows:
        write(f"{status:<5} {str(subject):<{width}}{detail}")
    worst = max((_WORTH.get(str(s), 2) for s, _b, _c in rows), default=0)
    fails = [b for s, b, _c in rows if str(s) == FAIL]
    warns = [b for s, b, _c in rows if str(s) == WARN]
    write("")
    if fails:
        write(f"{len(fails)} of these "
              f"{'needs' if len(fails) == 1 else 'need'} fixing: "
              + ", ".join(map(str, fails)))
    elif warns:
        write("nothing is broken. " + f"{len(warns)} thing"
              f"{'' if len(warns) == 1 else 's'} worth knowing about: "
              + ", ".join(map(str, warns)))
    else:
        write("everything checked out. `vectormailbox gui` is the next line.")
    return 1 if worst >= _WORTH[FAIL] else 0


def cmd_doctor(args) -> None:
    """`vectormailbox doctor [--offline]`."""
    from .mailbox import load_config

    cfg = load_config(getattr(args, "config", "") or "",
                      getattr(args, "profile", "") or "")
    print(f"vectormailbox {__version__} - these checks only read: they open the index"
          " read-only, ask the endpoint one question, and EXAMINE one IMAP folder."
          " Nothing is sent, written, or deleted.")
    print("")
    sys.exit(report(checks(cfg, offline=bool(getattr(args, "offline", False)))))


def add_subcommand(sub) -> None:
    p = sub.add_parser("doctor",
                       help="check the install, the files, the endpoint and the"
                            " mailbox: read-only, and says what to fix")
    p.add_argument("--offline", action="store_true",
                   help="skip the two network checks (endpoint and Gmail) and look"
                        " only at this machine")
    p.set_defaults(func=cmd_doctor)
