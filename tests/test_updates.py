from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from fiveflow import updates


class UpdatesTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.remote = self.root / 'remote.git'
        self.work = self.root / 'work'
        self.seed = self.root / 'seed'
        self.git(self.root, 'init', '--bare', str(self.remote))
        self.git(self.root, 'init', '-b', 'master', str(self.seed))
        for key, value in [('user.email', 'tests@example.invalid'), ('user.name', 'Tests')]:
            self.git(self.seed, 'config', key, value)
        (self.seed / 'requirements.txt').write_text('')
        (self.seed / 'app.txt').write_text('one')
        self.git(self.seed, 'add', '.')
        self.git(self.seed, 'commit', '-m', 'initial')
        self.git(self.seed, 'remote', 'add', 'origin', str(self.remote))
        self.git(self.seed, 'push', '-u', 'origin', 'master')
        self.git(self.root, 'clone', str(self.remote), str(self.work))
        for key, value in [('user.email', 'tests@example.invalid'), ('user.name', 'Tests')]:
            self.git(self.work, 'config', key, value)

    def git(self, root, *args):
        return subprocess.run(['git', '-C', str(root), *args], text=True,
                              capture_output=True, check=True).stdout.strip()

    def advance_remote(self):
        (self.seed / 'app.txt').write_text('two')
        self.git(self.seed, 'commit', '-am', 'update')
        self.git(self.seed, 'push')

    def fake_pip(self, fail=False):
        real_run = subprocess.run
        def run(args, **kwargs):
            if args[1:3] == ['-m', 'pip']:
                if fail:
                    raise subprocess.CalledProcessError(1, args)
                return subprocess.CompletedProcess(args, 0)
            return real_run(args, **kwargs)
        return mock.patch.object(updates.subprocess, 'run', side_effect=run)

    def test_clean_checkout_fast_forwards(self):
        self.advance_remote()
        target = updates.available_update(self.work)
        with self.fake_pip():
            updates.apply_update(self.work, target)
        self.assertEqual((self.work / 'app.txt').read_text(), 'two')

    def test_update_preserves_dirty_tracked_files(self):
        self.advance_remote()
        (self.work / 'app.txt').write_text('my changes')
        with self.assertRaisesRegex(RuntimeError, 'Local changes'):
            updates.apply_update(self.work, updates.available_update(self.work))
        self.assertEqual((self.work / 'app.txt').read_text(), 'my changes')

    def test_update_preserves_untracked_files(self):
        self.advance_remote()
        (self.work / 'notes.txt').write_text('keep me')
        with self.assertRaises(RuntimeError):
            updates.apply_update(self.work, updates.available_update(self.work))
        self.assertEqual((self.work / 'notes.txt').read_text(), 'keep me')

    def test_update_refuses_divergent_commits(self):
        self.advance_remote()
        (self.work / 'local.txt').write_text('local')
        self.git(self.work, 'add', '.')
        self.git(self.work, 'commit', '-m', 'local')
        before = self.git(self.work, 'rev-parse', 'HEAD')
        with self.assertRaises(subprocess.CalledProcessError):
            updates.apply_update(self.work, updates.available_update(self.work))
        self.assertEqual(self.git(self.work, 'rev-parse', 'HEAD'), before)

    def test_failed_pip_restores_previous_code(self):
        self.advance_remote()
        before = self.git(self.work, 'rev-parse', 'HEAD')
        target = updates.available_update(self.work)
        with self.fake_pip(fail=True):
            with self.assertRaises(subprocess.CalledProcessError):
                updates.apply_update(self.work, target)
        self.assertEqual(self.git(self.work, 'rev-parse', 'HEAD'), before)
        self.assertEqual((self.work / 'app.txt').read_text(), 'one')

    def test_local_ahead_is_not_reported_as_available_update(self):
        (self.work / 'local.txt').write_text('local')
        self.git(self.work, 'add', '.')
        self.git(self.work, 'commit', '-m', 'local')
        self.assertIsNone(updates.available_update(self.work))

    def test_renamed_remote_is_supported(self):
        self.git(self.work, 'remote', 'rename', 'origin', 'deshiyan')
        self.advance_remote()
        self.assertEqual(updates.available_update(self.work), 'deshiyan/master')
