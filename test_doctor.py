"""Tests for `vectormailbox doctor`.

The command exists to be believed, so these tests are mostly about the two ways a
diagnostic lies: it says `ok` about something it did not look at, or it says
`fail` about something that is fine. Both are worse than no diagnostic — the first
sends a person to the next suspect, the second teaches them to ignore the tool.
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vectormailbox import doctor
from vectormailbox import gui
from vectormailbox._version import __version__

REPO = Path(__file__).resolve().parent


def row(rows, subject):
    for status, name, detail in rows:
        if name == subject:
            return status, detail
    raise AssertionError(f"no check reported on {subject!r}: {[r[1] for r in rows]}")


class TestVerdicts(unittest.TestCase):
    """`report`: the part a person acts on."""

    def setUp(self):
        self.lines = []
        self.write = self.lines.append

    def say(self, rows):
        code = doctor.report(rows, write=self.write)
        return code, "\n".join(self.lines)

    def test_only_a_failure_makes_the_command_fail(self):
        self.assertEqual(self.say([(doctor.OK, "a", "fine")])[0], 0)
        self.assertEqual(self.say([(doctor.WARN, "a", "worth knowing")])[0], 0)
        self.assertEqual(self.say([(doctor.FAIL, "a", "broken")])[0], 1)

    def test_one_of_these_needs_fixing(self):
        """The singular is the common case — one thing is wrong — and "1 of these
        need fixing" reads like the tool was written by something that cannot
        count, which is not the impression to leave at the moment of trust."""
        _code, text = self.say([(doctor.FAIL, "endpoint", "connection refused"),
                                (doctor.OK, "python", "3.12.0")])
        self.assertIn("1 of these needs fixing: endpoint", text)
        _code, two = self.say([(doctor.FAIL, "endpoint", "refused"),
                               (doctor.FAIL, "gmail", "refused")])
        self.assertIn("2 of these need fixing: endpoint, gmail", two)

    def test_a_warning_says_nothing_is_broken(self):
        _code, text = self.say([(doctor.OK, "a", "fine"),
                                (doctor.WARN, "index", "empty")])
        self.assertIn("nothing is broken", text)
        self.assertIn("1 thing worth knowing about: index", text)

    def test_all_clear_invites_the_next_command(self):
        _code, text = self.say([(doctor.OK, "a", "fine")])
        self.assertIn("everything checked out", text)
        self.assertIn("vectormailbox gui", text)

    def test_the_table_lines_up_what_you_read(self):
        """Subjects start at the same column whatever the label's length: the
        table is read as two columns, and a `fail` that shifts its detail by two
        characters is the line you misread."""
        _code, text = self.say([(doctor.OK, "python", "3.12.0"),
                                (doctor.FAIL, "sanitiser", "missing")])
        lines = [line for line in text.splitlines()
                 if line.startswith(("ok", "fail"))]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].index("python"), lines[1].index("sanitiser"))
        self.assertTrue(lines[1].startswith("fail"))

    def test_it_does_not_ask_the_console_for_unicode(self):
        """A Windows console is often cp1252, and `main()` reconfigures the stream
        with errors="replace": an em dash in a detail becomes `?`, in the one
        output this app is read on a terminal. ASCII here is a feature, not taste."""
        tmp = Path(tempfile.mkdtemp())
        cfg = {"config_path": str(tmp / "config.json"), "db_path": str(tmp / "m.db"),
               "dim": 8}
        # The whole assembled table, offline: the names the table prints are part
        # of what a Windows console has to be able to draw.
        rows = doctor.checks(cfg, offline=True)
        self.assertIn("sanitiser", [name for _s, name, _d in rows])
        text = "\n".join(f"{s} {n} {d}" for s, n, d in rows)
        self.assertTrue(text.isascii(), [c for c in text if not c.isascii()][:10])
        _code, said = self.say(rows)
        self.assertTrue(said.isascii(), [c for c in said if not c.isascii()][:10])


class TestMachine(unittest.TestCase):
    def test_the_floor_it_states_is_the_floor_the_package_states(self):
        """`doctor` tells you which Python you need. If pyproject and this module
        disagree, one of them is lying, and the person holding the terminal cannot
        tell which."""
        want = requires_python()
        self.assertEqual(doctor.MIN_PYTHON, want)

    def test_a_python_this_old_is_a_failure_not_a_note(self):
        with mock.patch.object(doctor.sys, "version_info", (3, 8, 12)):
            status, detail = doctor.check_python()
        self.assertEqual(status, doctor.FAIL)
        self.assertIn("3.8.12", detail)
        self.assertIn("3.10", detail)

    def test_the_sanitizer_is_tested_not_imported(self):
        """An import that succeeds proves a wheel opened. What matters is that a
        script handler does not survive, so the check sends one."""
        status, detail = doctor.check_sanitizer()
        if status == doctor.FAIL and "not importable" in detail:
            self.skipTest("nh3 is not installed here")
        self.assertEqual(status, doctor.OK)
        clean = __import__("nh3").clean('<a href="javascript:alert(1)" onclick="y()">t')
        self.assertNotIn("onclick", clean)

    def test_a_sanitizer_that_keeps_a_handler_is_reported_as_broken(self):
        fake = mock.Mock()
        fake.clean.return_value = '<a onclick="y()">t</a>'
        with mock.patch.dict(sys.modules, {"nh3": fake}):
            status, detail = doctor.check_sanitizer()
        self.assertEqual(status, doctor.FAIL)
        self.assertIn("onclick", detail)

    def test_a_source_checkout_says_so(self):
        """People install with `uv tool install` and then run the copy they cloned
        six weeks ago. The line that ends the argument is the folder."""
        status, detail = doctor.check_package()
        self.assertIn(str(Path(doctor.__file__).parent), detail)
        self.assertIn(__version__, detail)
        self.assertIn(status, (doctor.OK, doctor.WARN))


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def cfg(self, **over):
        cfg = {"config_path": str(self.tmp / "config.json"),
               "db_path": str(self.tmp / "mailbox.db"), "dim": 8}
        return {**cfg, **over}

    def index(self, dim=8, messages=2, vectors=2):
        con = sqlite3.connect(self.tmp / "mailbox.db")
        con.executescript("""create table messages(id integer primary key);
                             create table chunks(id integer primary key);
                             create table embeddings(chunk_id integer primary key,
                                                     vec blob, dim integer);
                             create table index_state(key text primary key,
                                                      value text);""")
        con.executemany("insert into messages values (?)", [(i,) for i in range(messages)])
        con.executemany("insert into chunks values (?)", [(i,) for i in range(messages)])
        con.executemany("insert into embeddings values (?, ?, ?)",
                        [(i, b"", dim) for i in range(vectors)])
        con.execute("insert into index_state values ('last_indexed_date', ?)",
                    ("2026-03-04T10:00:00+00:00",))
        con.commit()
        con.close()

    def test_an_index_that_does_not_exist_yet_is_a_warning(self):
        status, detail = doctor.check_index(self.cfg())
        self.assertEqual(status, doctor.WARN)
        self.assertIn("yet", detail)
        self.assertIn("refresh", detail)

    def test_a_file_that_is_not_a_database_fails_where_it_is_opened(self):
        """`sqlite3.connect` is lazy: a folder is the case that cannot open, so
        that is what the two failure messages split on."""
        (self.tmp / "mailbox.db").mkdir()
        status, detail = doctor.check_index(self.cfg())
        self.assertEqual(status, doctor.FAIL)
        self.assertIn("will not open", detail)

    def test_a_database_that_is_not_ours(self):
        """Read as a database, empty of our tables: a `mailbox.db` copied from
        another program, or an index from before the schema existed."""
        sqlite3.connect(self.tmp / "mailbox.db").close()
        status, detail = doctor.check_index(self.cfg())
        self.assertEqual(status, doctor.FAIL)
        self.assertIn("not a vectormailbox index", detail)

    def test_the_counted_index_reports_the_date_it_was_read_through(self):
        self.index()
        status, detail = doctor.check_index(self.cfg())
        self.assertEqual(status, doctor.OK)
        self.assertIn("2 mail", detail)
        self.assertIn("dim 8", detail)
        self.assertIn("indexed through 2026-03-04", detail)

    def test_vectors_of_a_different_size_are_a_failure_not_a_note(self):
        """The worst failure this app has: an index built at 768 and queried at
        384 still returns a ranked list, and the list is meaningless."""
        self.index(dim=8)
        status, detail = doctor.check_index(self.cfg(dim=768))
        self.assertEqual(status, doctor.FAIL)
        self.assertIn("8-dimensional", detail)
        self.assertIn("dim=768", detail)

    def test_an_indexed_mailbox_with_no_vectors_is_only_a_warning(self):
        """Lexical search works without them, so this is a half-built index, not a
        broken one."""
        self.index(vectors=0)
        status, _detail = doctor.check_index(self.cfg())
        self.assertEqual(status, doctor.WARN)

    def test_an_empty_index_tells_you_which_button_fills_it(self):
        self.index(messages=0, vectors=0)
        status, detail = doctor.check_index(self.cfg())
        self.assertEqual(status, doctor.WARN)
        self.assertIn("empty", detail)

    def test_a_missing_config_is_a_warning_that_names_where_it_goes(self):
        status, detail = doctor.check_config(self.cfg())
        self.assertEqual(status, doctor.WARN)
        self.assertIn(str(self.tmp / "config.json"), detail)
        self.assertIn("settings card", detail)

    def test_a_config_file_reports_its_own_path(self):
        path = self.tmp / "config.json"
        path.write_text(json.dumps({"dim": 8}), encoding="utf-8")
        status, detail = doctor.check_config(self.cfg())
        self.assertEqual(status, doctor.OK)
        self.assertIn(str(path), detail)
        self.assertIn("bytes", detail)
        if os.name == "posix":
            self.assertIn("mode", detail)
        else:
            self.assertIn("advisory", detail)   # chmod proves nothing on Windows

    def test_a_world_readable_config_is_warned_about(self):
        """It holds an app password. This is the one place the file's *mode* is
        worth a line, because the install instructions cannot reach a file a
        person copied into a shared folder."""
        path = self.tmp / "config.json"
        path.write_text("{}", encoding="utf-8")
        os.chmod(path, 0o644)
        status, detail = doctor.check_config(self.cfg())
        if os.name != "posix":
            self.skipTest("POSIX modes are not enforced here")
        self.assertEqual(status, doctor.WARN)
        self.assertIn("chmod 600", detail)
        os.chmod(path, 0o600)
        self.assertEqual(doctor.check_config(self.cfg())[0], doctor.OK)

    def test_the_data_folder_is_probed_by_writing_to_it(self):
        """`os.access` answers what the metadata says; this answers what happened.
        A roaming profile that refuses a write is exactly the failure an installed
        app can meet on its first run."""
        with mock.patch.object(doctor.paths, "data_dir", return_value=self.tmp):
            status, detail = doctor.check_data_dir(self.cfg())
        self.assertEqual(status, doctor.OK, detail)
        self.assertEqual([p.name for p in self.tmp.iterdir()], [])   # cleans up after

    def test_a_data_folder_that_refuses_the_write_fails_loudly(self):
        missing = self.tmp / "nope" / "deeper"
        with mock.patch.object(doctor.paths, "data_dir", return_value=missing):
            with mock.patch("tempfile.NamedTemporaryFile",
                            side_effect=OSError(13, "denied")):
                status, detail = doctor.check_data_dir(self.cfg())
        self.assertEqual(status, doctor.FAIL)
        self.assertIn("cannot write", detail)


class TestProbes(unittest.TestCase):
    """The two network rows must be the settings card's probes, not a second
    opinion: two definitions of "the endpoint works" is how a user ends up with a
    green terminal and a red page about the same server."""

    def cfg(self):
        return {"embed_endpoint": "http://127.0.0.1:9/v1/embeddings", "dim": 8}

    def test_the_endpoint_line_is_the_pages_answer(self):
        with mock.patch.object(gui, "test_embedding",
                               return_value={"ok": True, "detail": "8 dimensions"}):
            self.assertEqual(doctor.check_endpoint(self.cfg()),
                             (doctor.OK, "8 dimensions"))

    def test_the_gmail_line_is_the_pages_answer(self):
        with mock.patch.object(gui, "test_gmail",
                               return_value={"ok": False, "detail": "refused"}):
            self.assertEqual(doctor.check_gmail(self.cfg()),
                             (doctor.FAIL, "refused"))

    def test_a_probe_that_cannot_run_is_a_warning_not_a_failure(self):
        """No fastapi in the interpreter is not a broken mailbox, and a `fail`
        here would send someone to fix a Gmail password over a missing wheel."""
        with mock.patch.object(gui, "test_gmail", side_effect=ImportError("no fastapi")):
            status, detail = doctor.check_gmail(self.cfg())
        self.assertEqual(status, doctor.WARN)
        self.assertIn("skipped", detail)

    def test_a_probe_that_says_yes_without_a_sentence_still_says_yes(self):
        with mock.patch.object(gui, "test_embedding", return_value={"ok": True}):
            self.assertEqual(doctor.check_endpoint(self.cfg()),
                             (doctor.OK, "answered"))

    def test_the_offline_flag_leaves_the_network_out(self):
        cfg = {"config_path": "", "db_path": "", "dim": 8}
        with mock.patch.object(doctor, "check_endpoint",
                               side_effect=AssertionError("ran the network")):
            rows = doctor.checks(cfg, offline=True)
        names = [name for _s, name, _d in rows]
        self.assertNotIn("endpoint", names)
        self.assertNotIn("gmail", names)
        self.assertIn("index", names)


class TestCommand(unittest.TestCase):
    def test_the_command_is_registered_with_a_flag(self):
        import argparse

        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="command")
        doctor.add_subcommand(subs)
        args = parser.parse_args(["doctor", "--offline"])
        self.assertIs(args.func, doctor.cmd_doctor)
        self.assertTrue(args.offline)

    def test_it_exits_nonzero_when_something_failed(self):
        args = mock.Mock(config="", profile="", offline=True)
        with mock.patch.object(doctor, "checks",
                               return_value=[("fail", "index", "no file")]):
            with self.assertRaises(SystemExit) as got:
                doctor.cmd_doctor(args)
        self.assertEqual(got.exception.code, 1)

    def test_the_version_flag_answers_without_a_config(self):
        """`vectormailbox --version` is the line that answers "did the install
        work" before there is a config, an index, or a folder to stand in — which
        makes it the first thing a person on a new Mac will type."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.txt"
            with open(out, "wb") as fh:      # a file, not a pipe: no stdio handshake
                done = subprocess.run([sys.executable, "-m", "vectormailbox",
                                       "--version"], cwd=str(REPO), stdout=fh,
                                      stderr=subprocess.STDOUT)
            said = out.read_text(encoding="utf-8", errors="replace")
        self.assertEqual(done.returncode, 0, said)
        self.assertIn(f"vectormailbox {__version__}", said)
        self.assertTrue(said.isascii(), said)


def requires_python() -> tuple:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("requires-python"):
            # Split on the first `=` only: the value is `">=3.10"`, and splitting on
            # every one leaves `>` where the number should be.
            found = line.split("=", 1)[1].strip().strip("\"'")
            number = found.replace(">=", "").replace("~=", "").strip()
            major, minor = number.split(".")[:2]
            return (int(major), int(minor))
    raise AssertionError("no requires-python in pyproject.toml")


if __name__ == "__main__":
    unittest.main(verbosity=2)
