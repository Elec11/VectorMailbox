"""Tests for `vectormailbox.paths` - where the app reads and writes its files.

These are the rules that decide whether an installed app finds *your* mailbox or
an empty one, so they are asserted directly rather than through the GUI:

* the data directory is per-user and overridable;
* `config.json` is found by precedence, and a new one is never written into
  whatever folder `uvx` happened to be called from;
* a relative path inside a config file means *beside that file*;
* a relative `--db` on the command line means *where I am standing*;
* the index run's log belongs to the index it describes.

Nothing here touches a real home directory: `VECTORMAILBOX_HOME`, `HOME` and the
current directory are all pointed at a temporary folder for the duration.
"""

import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vectormailbox import mailbox as mb          # noqa: E402
from vectormailbox import paths                  # noqa: E402

#: Anything a developer's own environment might have set that these tests
#: reason about; left in place it would decide the answer for them.
QUIET = ("VECTORMAILBOX_HOME", "VECTORMAILBOX_CONFIG", "APPDATA", "LOCALAPPDATA",
         "XDG_DATA_HOME", "GMAIL_USER", "GMAIL_APP_PASSWORD", "EMBED_ENDPOINT",
         "EMBED_API_KEY", "EMBED_MODEL", "DB_PATH")


@contextlib.contextmanager
def env(**values):
    """Environment with `values` applied (a value of None removes the key)."""
    saved = {k: os.environ.get(k) for k in list(values) + list(QUIET)}
    try:
        for key in QUIET:
            os.environ.pop(key, None)
        for key, val in values.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        yield
    finally:
        os.environ.clear()
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val


@contextlib.contextmanager
def cwd(path):
    """Run inside `path`, and go back after."""
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


class TestPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    # -- the data directory -------------------------------------------------

    def test_the_data_dir_is_under_the_home_dir_per_os(self):
        """Each platform gets the folder its own conventions expect, and none of
        them is the application folder: a mailbox should not need elevated
        rights to start, and should not sit where another account can read it.

        `_default_data_root` returns a string for exactly this test: patching
        `os.name` would otherwise make `Path` try to build a POSIX path on
        Windows, which it refuses to do.
        """
        home = self.root / "home"
        cases = (("posix", "darwin", home / "Library" / "Application Support"
                  / "vectormailbox"),
                 ("posix", "linux2", home / ".local" / "share" / "vectormailbox"),
                 ("posix", "linux2", home / "elsewhere" / "vectormailbox"))
        for os_name, platform, want in cases:
            overrides = {"HOME": str(home), "USERPROFILE": str(home)}
            if want.parent.name == "elsewhere":
                overrides["XDG_DATA_HOME"] = str(home / "elsewhere")
            with env(**overrides):
                with mock.patch.object(sys, "platform", platform), \
                        mock.patch.object(os, "name", os_name):
                    self.assertEqual(paths._default_data_root(), str(want))
        with env(HOME=str(home), USERPROFILE=str(home),
                 APPDATA=str(self.root / "roaming")):
            with mock.patch.object(sys, "platform", "win32"), \
                    mock.patch.object(os, "name", "nt"):
                self.assertEqual(paths._default_data_root(),
                                 str(self.root / "roaming" / "vectormailbox"))

    def test_vectormailbox_home_moves_the_whole_lot(self):
        """The portable-mode switch, and the only way to keep two mailboxes
        apart without passing --config at every command."""
        empty = self.root / "empty"
        empty.mkdir()
        with env(VECTORMAILBOX_HOME=str(self.root / "portable")), cwd(empty):
            self.assertEqual(paths.data_dir(), self.root / "portable")
            self.assertEqual(paths.config_path(),
                             self.root / "portable" / "config.json")

    def test_a_named_home_outranks_a_stray_config_in_the_cwd(self):
        """`VECTORMAILBOX_HOME` says where this mailbox lives, so its config
        answers before a `./config.json` someone else left in the folder the
        command was typed from - which would otherwise open a different
        mailbox, silently."""
        data = self.root / "portable"
        data.mkdir()
        (data / "config.json").write_text("{}", encoding="utf-8")
        stray = self.root / "stray"
        stray.mkdir()
        (stray / "config.json").write_text("{}", encoding="utf-8")
        with env(VECTORMAILBOX_HOME=str(data)), cwd(stray):
            self.assertEqual(paths.config_path(), data / "config.json")

    def test_asking_for_a_path_creates_nothing(self):
        """`data_dir()` is called on the way to *reading* a config; a question
        must not leave a directory behind."""
        with env(VECTORMAILBOX_HOME=str(self.root / "notyet")):
            paths.config_path()
            self.assertFalse((self.root / "notyet").exists())
            paths.data_dir(create=True)
            self.assertTrue((self.root / "notyet").is_dir())

    # -- which config.json --------------------------------------------------

    def test_a_config_in_the_current_directory_still_wins(self):
        """A checkout is a config file next to the code, and every measurement
        in this repo was taken that way."""
        cfg = self.root / "config.json"
        cfg.write_text("{}", encoding="utf-8")
        with env(VECTORMAILBOX_HOME=str(self.root / "data")), cwd(self.root):
            self.assertEqual(paths.config_path(), cfg)

    def test_with_no_config_anywhere_the_default_is_the_data_dir(self):
        """The case `uvx` creates: nothing has been set up yet, and the current
        directory is a room of the filesystem the user was standing in. A new
        config - the one holding the Gmail app password - belongs in the data
        directory, not in Downloads."""
        empty = self.root / "somewhere"
        empty.mkdir()
        with env(VECTORMAILBOX_HOME=str(self.root / "data")), cwd(empty):
            got = paths.config_path()
            self.assertEqual(got, self.root / "data" / "config.json")
            self.assertNotIn("somewhere", str(got))

    def test_naming_a_config_file_is_an_instruction(self):
        """`--config other.json` is taken literally even when it does not exist,
        because a missing config has always meant "defaults and environment" -
        silently answering with a different person's mailbox would be worse."""
        other = self.root / "other.json"
        with env(VECTORMAILBOX_HOME=str(self.root / "data"),
                 VECTORMAILBOX_CONFIG=str(self.root / "env.json")):
            self.assertEqual(paths.config_path(str(other)), other)
            self.assertFalse(other.exists())

    def test_the_environment_names_the_file_when_told_to(self):
        """$VECTORMAILBOX_CONFIG answers for a plain `vectormailbox gui`, because
        `--config config.json` is what the CLI passes when nobody said anything
        - it is not a statement, so it cannot outrank one."""
        with env(VECTORMAILBOX_HOME=str(self.root / "data"),
                 VECTORMAILBOX_CONFIG=str(self.root / "env.json")):
            self.assertEqual(paths.config_path(), self.root / "env.json")
            self.assertEqual(paths.config_path("config.json"),
                             self.root / "env.json")

    def test_the_data_dir_config_is_found_when_the_cwd_has_none(self):
        """The normal installed case: setup wrote one config, and from now on
        every command finds it from any directory."""
        data = self.root / "data"
        data.mkdir()
        (data / "config.json").write_text("{}", encoding="utf-8")
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        with env(VECTORMAILBOX_HOME=str(data)), cwd(elsewhere):
            self.assertEqual(paths.config_path(), data / "config.json")

    # -- relative paths -----------------------------------------------------

    def test_beside_leaves_an_absolute_path_alone(self):
        absolute = Path(tempfile.gettempdir()) / "box.db"
        self.assertEqual(paths.beside(self.root / "cfg", absolute), absolute)
        self.assertEqual(paths.beside(self.root / "cfg", str(absolute)), absolute)

    def test_a_tilde_from_a_terminal_is_expanded(self):
        """`--config ~/mail/config.json` reaches the shell as a literal tilde on
        some terminals; a path handed to open() with it still in there points at
        a folder called "~"."""
        with env(HOME=str(self.root), USERPROFILE=str(self.root)):
            self.assertEqual(paths.beside(self.root / "cfg", "~/box.db"),
                             self.root / "box.db")
            self.assertEqual(paths.config_path("~/other.json"),
                             self.root / "other.json")

    def test_beside_puts_a_relative_path_next_to_the_config(self):
        self.assertEqual(paths.beside(self.root / "cfg", "mailbox.db"),
                         self.root / "cfg" / "mailbox.db")

    def test_a_relative_db_path_lives_beside_its_config(self):
        """The bug this rule exists to stop: `db_path: "mailbox.db"` resolved
        against the current directory hands the same user a fresh empty index
        every time they launch from a different folder."""
        data = self.root / "data"
        data.mkdir()
        cfg = data / "config.json"
        cfg.write_text(json.dumps({"db_path": "mailbox.db"}), encoding="utf-8")
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        with env(VECTORMAILBOX_HOME=str(data)), cwd(elsewhere):
            loaded = mb.load_config()
        self.assertEqual(loaded["db_path"], str(data / "mailbox.db"))
        self.assertEqual(loaded["config_path"], str(cfg))

    def test_an_absolute_db_path_is_left_as_it_says(self):
        box = self.root / "elsewhere.db"
        cfg = self.root / "config.json"
        cfg.write_text(json.dumps({"db_path": str(box)}), encoding="utf-8")
        with env(VECTORMAILBOX_HOME=str(self.root / "data")):
            self.assertEqual(mb.load_config(str(cfg))["db_path"], str(box))

    def test_a_db_named_on_the_command_line_is_where_you_are_standing(self):
        """`--db baseline-legacy.db` is typed beside the file. A config-relative
        rule would answer with a database in the data directory that nobody
        meant, and the measurement would be of nothing."""
        cfg = self.root / "config.json"
        cfg.write_text(json.dumps({"db_path": "mailbox.db"}), encoding="utf-8")
        args = mock.Mock(db="baseline-legacy.db")
        with cwd(self.root):
            got = mb._apply_db(args, mb.load_config(str(cfg)))
        self.assertEqual(got["db_path"], str(self.root / "baseline-legacy.db"))

    # -- the index run's log ------------------------------------------------

    def test_the_log_belongs_to_the_index_it_describes(self):
        """`--db baseline.db` measures a frozen copy; its progress must not
        append to the live mailbox's run history, the way the preview sidecar
        already gets its own file per index."""
        self.assertEqual(paths.index_log(self.root / "mailbox.db"),
                         self.root / "mailbox.index.log")
        self.assertEqual(paths.index_log(self.root / "baseline-legacy.db"),
                         self.root / "baseline-legacy.index.log")

    def test_the_engine_puts_its_log_beside_its_index(self):
        from vectormailbox import gui
        box = self.root / "baseline.db"
        eng = gui.Engine({"db_path": str(box)})
        self.addCleanup(eng.close)
        self.assertEqual(eng.refresh_log, self.root / "baseline.index.log")


if __name__ == "__main__":
    unittest.main(verbosity=2)
