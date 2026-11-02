import importlib.util
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys
import os


def load_utils():
    path = Path(__file__).parents[1] / 'fiveflow/utils.py'
    spec = importlib.util.spec_from_file_location('fiveflow._utils_test', path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {'Foundation': mock.MagicMock(), 'ApplicationServices': mock.MagicMock()}):
        spec.loader.exec_module(module)
    return module


utils = load_utils()


class UtilsTests(unittest.TestCase):
    def test_noninteractive_prompts_do_not_block(self):
        with mock.patch.object(utils.sys, 'stdin', StringIO('')):
            self.assertFalse(utils.ask_yes_no_arrow('update?'))
            utils.prompt_hf_token_if_needed()
            utils.check_for_updates()

    def test_existing_hf_token_is_preserved_without_prompting(self):
        with mock.patch.dict(os.environ, {'HF_TOKEN': 'existing-test-token'}):
            with mock.patch.object(utils.getpass, 'getpass') as prompt:
                utils.prompt_hf_token_if_needed()
        prompt.assert_not_called()

    def test_custom_complete_cache_does_not_prompt_for_token(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in (utils.WHISPER_MODEL_ID, utils.GEMMA_MODEL_ID):
                snapshot = Path(directory) / ('models--' + model.replace('/', '--')) / 'snapshots/test'
                snapshot.mkdir(parents=True)
                (snapshot / 'config.json').write_text('{}')
            with mock.patch.dict(os.environ, {'HF_HUB_CACHE': directory, 'HF_TOKEN': ''}):
                with mock.patch.object(utils.sys.stdin, 'isatty', return_value=True):
                    with mock.patch.object(utils.getpass, 'getpass') as prompt:
                        utils.prompt_hf_token_if_needed()
            prompt.assert_not_called()

    def test_incomplete_cache_still_offers_token_and_hides_input(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {'HF_HUB_CACHE': directory, 'HF_TOKEN': ''}):
                with mock.patch.object(utils.sys.stdin, 'isatty', return_value=True):
                    with mock.patch.object(utils.getpass, 'getpass', return_value='secret-test-token') as prompt:
                        utils.prompt_hf_token_if_needed()
                        self.assertEqual(os.environ['HF_TOKEN'], 'secret-test-token')
                prompt.assert_called_once()

    def test_ctrl_c_in_raw_terminal_cancels_and_restores_settings(self):
        stdin = mock.Mock()
        stdin.isatty.return_value = True
        stdin.fileno.return_value = 123
        self.enterContext(mock.patch.object(utils.os, 'read', return_value=b'\x03'))
        with mock.patch.object(utils.sys, 'stdin', stdin):
            with mock.patch.object(utils.termios, 'tcgetattr', return_value=['saved']):
                with mock.patch.object(utils.tty, 'setraw'):
                    with mock.patch.object(utils.termios, 'tcsetattr') as restore:
                        self.assertFalse(utils.ask_yes_no_arrow('update?'))
        restore.assert_called_once_with(123, utils.termios.TCSADRAIN, ['saved'])

    def test_arrow_keys_are_read_without_text_buffering_or_blocking(self):
        stdin = mock.Mock()
        stdin.isatty.return_value = True
        stdin.fileno.return_value = 123
        with mock.patch.object(utils.sys, 'stdin', stdin):
            with mock.patch.object(utils.termios, 'tcgetattr', return_value=['saved']):
                with mock.patch.object(utils.tty, 'setraw'):
                    with mock.patch.object(utils.termios, 'tcsetattr'):
                        with mock.patch.object(utils.select, 'select', return_value=([stdin], [], [])):
                            with mock.patch.object(utils.os, 'read', side_effect=[b'\x1b', b'[', b'D', b'\r']):
                                self.assertTrue(utils.ask_yes_no_arrow('update?'))
