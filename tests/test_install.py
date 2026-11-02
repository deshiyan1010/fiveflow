"""Run the installer with fake downloads/interpreters inside a disposable HOME."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


@unittest.skipUnless(sys.platform == 'darwin', 'installer targets macOS')
class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.home = Path(self.temp) / 'home with spaces'
        self.home.mkdir()
        self.bin = Path(self.temp) / 'bin'
        self.bin.mkdir()
        self.source = Path(__file__).parents[1]
        self.env = dict(os.environ, HOME=str(self.home), SHELL='/bin/zsh',
                        PATH=str(self.bin) + ':' + os.environ['PATH'],
                        FIVEFLOW_TEST_SOURCE=str(self.source), FIVEFLOW_REAL_PYTHON=sys.executable)
        self.script('uname', '#!/bin/sh\nif [ "$1" = "-m" ]; then echo arm64; else echo Darwin; fi\n')
        self.script('sw_vers', '#!/bin/sh\necho 14.0\n')
        self.script('git', '''#!/usr/bin/env python3
import os, pathlib, shutil, sys
args = sys.argv[1:]
if 'clone' in args:
    root = pathlib.Path(args[-1])
    (root / '.git').mkdir(parents=True)
    shutil.copytree(pathlib.Path(os.environ['FIVEFLOW_TEST_SOURCE']) / 'fiveflow', root / 'fiveflow')
    (root / 'stt_pipeline.py').touch()
    (root / 'requirements.txt').touch()
if os.environ.get('FIVEFLOW_GIT_FAIL'):
    sys.exit(1)
''')
        self.script('python3.13', '''#!/usr/bin/env python3
import os, pathlib, sys
if sys.argv[1] == '-c':
    print('arm64')
elif sys.argv[1:3] == ['-m', 'venv']:
    root = pathlib.Path(sys.argv[3]) / 'bin'
    root.mkdir(parents=True)
    python = root / 'python'
    python.write_text("#!/usr/bin/env python3\\nimport os, sys, json\\nif len(sys.argv)>1 and sys.argv[1].endswith('shell_config.py'):\\n    os.execv(os.environ['FIVEFLOW_REAL_PYTHON'], [os.environ['FIVEFLOW_REAL_PYTHON'], *sys.argv[1:]])\\nprint(json.dumps(sys.argv[1:]))\\nsys.exit(int(os.environ.get('FIVEFLOW_PYTHON_FAIL', '0')))\\n")
    python.chmod(0o755)
    pip = root / 'pip'
    pip.write_text('#!/bin/sh\\nexit 0\\n')
    pip.chmod(0o755)
''')
        self.rc = self.home / '.zshrc'
        self.rc.write_text('export KEEP_THIS=yes\n')

    def script(self, name, contents):
        path = self.bin / name
        path.write_text(textwrap.dedent(contents))
        path.chmod(0o755)

    def install(self):
        return subprocess.run(['/bin/bash', str(self.source / 'install.sh')], env=self.env,
                              capture_output=True, text=True, timeout=10)

    def shell(self, command):
        return subprocess.run(['/bin/bash', '-c', 'source "$HOME/.zshrc"\n' + command],
                              env=self.env, capture_output=True, text=True, timeout=10)

    def test_install_twice_preserves_existing_config_and_has_one_block(self):
        for _ in range(2):
            result = self.install()
            self.assertEqual(result.returncode, 0, result.stderr)
        text = self.rc.read_text()
        self.assertIn('export KEEP_THIS=yes', text)
        self.assertEqual(text.count('# >>> fiveflow >>>'), 1)
        result = self.shell("fiveflow 'one argument with spaces'")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.strip())[1:], ['one argument with spaces'])

    def test_update_command_propagates_errors_instead_of_claiming_success(self):
        self.assertEqual(self.install().returncode, 0)
        self.env['FIVEFLOW_PYTHON_FAIL'] = '7'
        result = self.shell('fiveflow-update')
        self.assertEqual(result.returncode, 7)
        self.assertNotIn('Done', result.stdout)

    def test_incomplete_existing_shell_block_is_preserved(self):
        original = 'keep\n# >>> fiveflow >>>\nmore settings\n'
        self.rc.write_text(original)
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual(self.rc.read_text(), original)

    def test_uninstall_removes_only_temp_install_and_managed_config(self):
        self.assertEqual(self.install().returncode, 0)
        result = self.shell('fiveflow-remove')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.home / '.fiveflow').exists())
        self.assertIn('export KEEP_THIS=yes', self.rc.read_text())
        self.assertNotIn('# >>> fiveflow >>>', self.rc.read_text())
