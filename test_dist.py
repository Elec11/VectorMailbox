"""Tests for what actually ships.

Everything else in this repository is tested as source: the suites import the
package, so a missing file is an import error here and nowhere else. An installed
copy is a different object — a zip built by a backend that decides what to
include, on a machine with no checkout, run from whatever folder a shell happens
to be standing in. Four things can go wrong there and none of them are visible
from a source test:

* the front end is not in the wheel, so the page 404s on a machine that cannot
  fetch it from anywhere;
* the console script is bound to the wrong attribute, so the command is not on
  PATH;
* the artifact carries somebody's `config.json` or `mailbox.db`, which is their
  mail and their app password;
* the code reads a path relative to the current directory, so it works in a
  checkout and fails after install.

The wheel and sdist are built here rather than stubbed, because the thing under
test is the build configuration in `pyproject.toml` and nothing else. If `uv` is
not on PATH the artifact tests skip; the source-tree and foreign-folder tests
still run.
"""

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vectormailbox import doctor, paths

REPO = Path(__file__).resolve().parent
PACKAGE = REPO / "vectormailbox"
WEB = PACKAGE / "web"

#: What must never be in an artifact, whichever end it hurts from: a person's
#: mail and credentials (`config.json`, `*.db`), state that belongs to whoever
#: built the wheel (`*.log`, `__pycache__`, `.deps/`, `*.bak`), or the checkout's
#: own measuring tools, which are not part of the product and refer to files
#: nobody installing this will have (`eval/`, `cap_audit.py`).
FORBIDDEN_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".bak",
                      ".log", ".env", ".pyc")
FORBIDDEN_NAMES = ("config.json", "eval.jsonl", ".deps", "__pycache__",
                   "test_gui.py", "test_mailbox.py", "cap_audit.py",
                   "embed_server.py", "reindex_tails.py", ".git")


def forbidden(names) -> list:
    bad = []
    for name in names:
        flat = name.replace("\\", "/")
        parts = flat.split("/")
        if flat.endswith(FORBIDDEN_SUFFIXES) or parts[-1] in FORBIDDEN_NAMES:
            bad.append(name)
        elif any(part in (".deps", "__pycache__", ".git") for part in parts):
            bad.append(name)
    return bad


class BuildOnce(unittest.TestCase):
    """Build the real artifacts once per run, from a writable temp folder."""

    wheel: Path = None
    sdist: Path = None
    cache: dict = {}

    @classmethod
    def setUpClass(cls):
        if cls.cache.get("error"):
            raise unittest.SkipTest(cls.cache["error"])
        if cls.cache:
            cls.wheel, cls.sdist = cls.cache["wheel"], cls.cache["sdist"]
            return
        uv = shutil.which("uv")
        if not uv:
            raise unittest.SkipTest("uv is not on PATH: nothing to build the "
                                    "wheel with (pipx/`uv tool install` need it too)")
        tmp = Path(tempfile.mkdtemp(prefix="dist-build-"))
        with open(tmp / "build.log", "wb") as log:
            # `--cache-dir` inside the temp folder: uv's default cache lives in
            # the user profile, which a sandboxed test run cannot open.
            done = subprocess.run([uv, "build", "--out-dir", str(tmp),
                                   "--cache-dir", str(tmp / "cache")],
                                  cwd=str(REPO), stdout=log,
                                  stderr=subprocess.STDOUT)
        if done.returncode != 0:
            cls.cache["error"] = (f"`uv build` failed ({done.returncode}): see "
                                 f"{tmp / 'build.log'}")
            raise unittest.SkipTest(cls.cache["error"])
        cls.cache.update(wheel=next(tmp.glob("*.whl")),
                         sdist=next(tmp.glob("*.tar.gz")))
        cls.wheel, cls.sdist = cls.cache["wheel"], cls.cache["sdist"]

    @classmethod
    def wheel_names(cls):
        with zipfile.ZipFile(cls.wheel) as zf:
            return set(zf.namelist())

    @classmethod
    def wheel_text(cls, suffix: str) -> str:
        with zipfile.ZipFile(cls.wheel) as zf:
            name = next(n for n in zf.namelist() if n.endswith(suffix))
            return zf.read(name).decode("utf-8")

    @classmethod
    def sdist_names(cls):
        with tarfile.open(cls.sdist) as tf:
            return {m.name.split("/", 1)[1] for m in tf.getmembers() if "/" in m.name}


class TestWheel(BuildOnce):
    def test_the_front_end_is_in_the_package(self):
        """`web/` is served from inside the package. A wheel without it installs
        cleanly and then 404s on a machine that has no other way to get the
        page — the worst shape a packaging mistake can take."""
        names = self.wheel_names()
        for asset in ("index.html", "app.css", "app.js", "mailhtml.js"):
            self.assertIn(f"vectormailbox/web/{asset}", names)
        for asset in sorted(p.name for p in WEB.iterdir()):
            self.assertIn(f"vectormailbox/web/{asset}", names,
                          f"{asset} is in the source tree and not in the wheel")

    def test_every_module_of_the_package_is_in_the_wheel(self):
        """A module that is not declared, or a new subpackage the `packages` list
        does not name, is a `ModuleNotFoundError` on the installed machine and a
        green suite here."""
        source = {f"vectormailbox/{p.name}" for p in PACKAGE.glob("*.py")}
        self.assertTrue(source)
        self.assertEqual(source, {n for n in self.wheel_names()
                                  if n.startswith("vectormailbox/")
                                  and n.endswith(".py")})

    def test_the_command_on_path_is_bound_to_the_package(self):
        points = self.wheel_text("entry_points.txt")
        self.assertIn("[console_scripts]", points)
        self.assertIn("vectormailbox = vectormailbox.cli:main", points)

    def test_the_version_and_the_floor_travel_in_the_metadata(self):
        from vectormailbox._version import __version__

        meta = self.wheel_text("METADATA")
        self.assertIn(f"Version: {__version__}", meta)
        self.assertIn(f"Requires-Python: >={doctor.MIN_PYTHON[0]}."
                      f"{doctor.MIN_PYTHON[1]}", meta)

    def test_the_sanitizer_and_the_web_layer_are_required_not_optional(self):
        """`nh3` and `fastapi` are not extras: without one you get an unsanitised
        mail body, without the other there is no page."""
        requires = [line.split(": ", 1)[1] for line in
                    self.wheel_text("METADATA").splitlines()
                    if line.startswith("Requires-Dist:")]
        core = [r for r in requires if "extra ==" not in r]
        for need in ("nh3", "fastapi", "uvicorn", "numpy", "beautifulsoup4"):
            self.assertTrue([r for r in core if r.startswith(need)],
                            f"{need} is not a core requirement")
        self.assertTrue([r for r in requires if r.startswith("PyMuPDF")])
        self.assertFalse([r for r in core if r.startswith("PyMuPDF")],
                         "PyMuPDF belongs in the media extra, not in core")

    def test_it_carries_nobody_s_mail(self):
        bad = forbidden(self.wheel_names())
        self.assertEqual(bad, [])
        self.assertNotIn("vectormailbox/config.json", self.wheel_names())

    def test_the_record_lists_what_is_in_the_zip(self):
        """`RECORD` is what an installer verifies. A file present but unlisted is
        a file that survives no upgrade and passes no check — and RECORD itself is
        in the zip without being in its own list, which is why both sides drop it.
        """
        record = next(n for n in self.wheel_names() if n.endswith("/RECORD"))
        listed = {line.split(",")[0] for line in self.wheel_text("RECORD").splitlines()
                  if line.strip()}
        self.assertEqual(self.wheel_names() - {record}, listed - {record})


class TestSdist(BuildOnce):
    def test_a_rebuilder_gets_the_sources_and_the_recipe(self):
        names = self.sdist_names()
        for need in ("pyproject.toml", "PKG-INFO", "README.md",
                     "vectormailbox/gui.py", "vectormailbox/web/app.js"):
            self.assertIn(need, names)

    def test_it_carries_no_mail_no_state_and_no_measuring_rig(self):
        self.assertEqual(forbidden(self.sdist_names()), [])

    def test_the_readme_travels_because_it_is_the_long_description(self):
        """`uv tool install` prints nothing about how to start; the page people
        see on PyPI-like indexes, and `pip show -f`, come from this file."""
        self.assertIn("README.md", self.sdist_names())
        self.assertIn("vectormailbox gui", self.wheel_text("METADATA"))


class TestSourceTree(unittest.TestCase):
    """What the build is pointed at, checked without building."""

    def test_the_package_directory_holds_no_local_state(self):
        """`package-data` is a glob over the package folder: a `config.json` or a
        `mailbox.db` that lands there by accident gets *shipped* — this is the
        one place a stray file becomes a stranger's mailbox.

        `__pycache__` is exempt because `package-data` names `web/*` and never
        reaches beside it, and because git would not carry it. Widen that glob and
        the exemption becomes a hole, which is the reason it is written down.
        """
        offenders = [str(p.relative_to(REPO)) for p in PACKAGE.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and forbidden([str(p.relative_to(REPO))])]
        self.assertEqual(offenders, [])

    def test_the_declared_packages_are_the_ones_that_exist(self):
        """`packages = ["vectormailbox"]` is a promise. A subpackage that exists
        and is not named here is missing from the wheel and raises
        `ModuleNotFoundError` on the installed machine, never in a checkout."""
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        declared = re.findall(r'packages\s*=\s*\[([^\]]*)\]', text)
        listed = [m.strip().strip('"').strip("'") for chunk in declared
                  for m in chunk.split(",") if m.strip()]
        self.assertIn("vectormailbox", listed)
        for name in listed:
            self.assertTrue((REPO / Path(*name.split("."))).is_dir(),
                            f"{name} is declared and does not exist")
        subpackages = {p.name for p in PACKAGE.iterdir()
                       if p.is_dir() and (p / "__init__.py").exists()}
        declared_dirs = {d.rsplit(".", 1)[-1] for d in listed if d != "vectormailbox"}
        self.assertEqual(subpackages, declared_dirs)

    def test_the_web_glob_covers_every_asset_and_nothing_else(self):
        """The front end is four files with no build step: if a fifth appears
        without the glob being checked, or a file that is not a web asset sits in
        that folder, the artifact is quietly wrong."""
        names = {p.name for p in WEB.iterdir()}
        self.assertEqual(names, {"index.html", "app.css", "app.js", "mailhtml.js"})

    def test_the_package_is_importable_as_a_package_not_a_script(self):
        """`-m vectormailbox` is how the GUI starts the index, because an
        installed app has no script file beside it to exec. That needs
        `__main__.py` to exist and the console script's target to be callable."""
        from vectormailbox.cli import main

        self.assertTrue((PACKAGE / "__main__.py").is_file())
        self.assertTrue(callable(main))
        self.assertIn("main", (PACKAGE / "cli.py").read_text(encoding="utf-8"))


class TestForeignFolder(unittest.TestCase):
    """Run the installed entry points from a folder that is not a checkout, with
    a data home of their own: the shape of a first run after `uv tool install`."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dist-cwd-"))
        env = dict(os.environ)
        env.pop("VECTORMAILBOX_CONFIG", None)
        env["VECTORMAILBOX_HOME"] = str(self.tmp)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        self.env = env

    def run_cli(self, *args):
        out = self.tmp / "out.txt"
        with open(out, "wb") as fh:      # a file, not a pipe
            done = subprocess.run([sys.executable, "-m", "vectormailbox", *args],
                                  cwd=str(self.tmp), env=self.env, stdout=fh,
                                  stderr=subprocess.STDOUT)
        return done.returncode, out.read_text(encoding="utf-8", errors="replace")

    def test_the_folder_it_is_called_from_decides_nothing(self):
        code, said = self.run_cli("--version")
        self.assertEqual(code, 0, said)
        self.assertIn("vectormailbox", said)
        self.assertNotIn("Traceback", said)

    def test_help_works_before_there_is_anything_to_configure(self):
        code, said = self.run_cli("gui", "--help")
        self.assertEqual(code, 0, said)
        self.assertIn("--port", said)

    def test_the_first_run_looks_in_the_data_home_and_nowhere_else(self):
        """From a foreign folder, `doctor` must name the data home for both the
        config and the index: a first run that writes into the shell's current
        directory is the failure `paths.py` exists to prevent."""
        _code, said = self.run_cli("doctor", "--offline")
        self.assertNotIn("Traceback", said)
        self.assertIn(str(self.tmp), said)
        self.assertIn("config", said)
        self.assertIn("settings card", said)
        self.assertTrue(said.isascii(), [c for c in said if not c.isascii()][:10])
        self.assertIn(paths.APP_NAME, said)


if __name__ == "__main__":
    unittest.main(verbosity=2)
