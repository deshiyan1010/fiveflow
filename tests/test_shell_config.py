from pathlib import Path
import os
import tempfile
import unittest
from fiveflow.shell_config import replace_block, write_config, command_block

BLOCK = '# >>> fiveflow >>>\nfiveflow() { true; }\n# <<< fiveflow <<<'


class ShellConfigTests(unittest.TestCase):
    def test_install_is_idempotent_and_keeps_unrelated_config(self):
        content = 'export EDITOR=vim\n'
        installed = replace_block(content, BLOCK)
        self.assertEqual(replace_block(installed, BLOCK), installed)
        self.assertIn('export EDITOR=vim', installed)
        self.assertEqual(installed.count('# >>> fiveflow >>>'), 1)

    def test_incomplete_block_never_deletes_following_config(self):
        with self.assertRaises(ValueError):
            replace_block('keep\n# >>> fiveflow >>>\nimportant settings\n', BLOCK)

    def test_nested_or_unmatched_markers_are_rejected(self):
        for content in ('# <<< fiveflow <<<\n', '# >>> fiveflow >>>\n# >>> fiveflow >>>\n# <<< fiveflow <<<\n'):
            with self.assertRaises(ValueError):
                replace_block(content, BLOCK)

    def test_uninstall_removes_only_managed_block(self):
        self.assertEqual(replace_block('before\n' + BLOCK + '\nafter\n'), 'before\nafter\n')

    def test_atomic_write_preserves_symlink_and_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'dotfile'
            target.write_text('keep\n')
            target.chmod(0o640)
            link = Path(directory) / '.zshrc'
            link.symlink_to(target)
            write_config(link, BLOCK)
            self.assertTrue(link.is_symlink())
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertIn('keep', target.read_text())

    def test_failed_validation_does_not_modify_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.zshrc'
            original = 'before\n# >>> fiveflow >>>\nafter\n'
            path.write_text(original)
            with self.assertRaises(ValueError):
                write_config(path, BLOCK)
            self.assertEqual(path.read_text(), original)

    def test_generated_commands_quote_special_characters_in_paths(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "folder with '$dollars' and `ticks`"
            (root / 'stt_env/bin').mkdir(parents=True)
            runner = root / 'stt_env/bin/python'
            runner.write_text('#!/bin/sh\nprintf \"%s\\n\" \"$@\"\n')
            runner.chmod(0o755)
            block = command_block(root, root / '.zshrc', root / 'whisper', root / 'gemma')
            result = subprocess.run(['/bin/bash', '-c', block + '\nfiveflow \"argument with spaces\"'],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [str(root / 'stt_pipeline.py'), 'argument with spaces'])
