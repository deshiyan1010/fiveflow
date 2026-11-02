"""Lifecycle regressions with fake OS/audio/model boundaries; no microphone or GPU."""
import importlib.util
from pathlib import Path
import queue
import sys
import threading
import unittest
from unittest import mock

import numpy as np

from fiveflow import state


def load_pipeline():
    # Import the real implementation while replacing only platform/model I/O.
    # This also runs on CI without AppKit, MLX, or downloaded model weights.
    names = ('sounddevice', 'torch', 'transformers', 'mlx_vlm', 'pynput',
             'pynput.keyboard', 'Cocoa', 'Quartz', 'fiveflow.ui.widgets')
    modules = {name: mock.MagicMock() for name in names}
    quartz = modules['Quartz']
    for name, value in {
        'kCGEventFlagsChanged': 12, 'kCGEventKeyDown': 10, 'kCGEventKeyUp': 11,
        'kCGEventTapDisabledByTimeout': 0xFFFFFFFE,
        'kCGEventTapDisabledByUserInput': 0xFFFFFFFF,
        'kCGEventSourceStateHIDSystemState': 1,
        'kCGEventSourceUnixProcessID': 41,
    }.items():
        setattr(quartz, name, value)
    spec = importlib.util.spec_from_file_location(
        'fiveflow._pipeline_test', Path(__file__).parents[1] / 'fiveflow/pipeline.py')
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


pipeline = load_pipeline()
FN = pipeline.kCGEventFlagMaskSecondaryFn
SHIFT = pipeline.kCGEventFlagMaskShift


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.p = pipeline.Pipeline()
        self.p._models_ready[0] = True
        self.enterContext(mock.patch.multiple(
            state, transcribe_keycode=63, listening_for_key=False,
            on_key_assigned=None, transcription_history=[], audio_level=0.0,
            ui_callbacks=queue.Queue()))
        self.widget = self.enterContext(mock.patch.object(pipeline, 'set_widget'))
        self.enterContext(mock.patch.object(pipeline, 'logger'))
        self.enterContext(mock.patch.object(pipeline, 'CGEventGetIntegerValueField',
                                          side_effect=lambda event, field: event[0] if field == 9 else 0))
        self.enterContext(mock.patch.object(pipeline, 'CGEventGetFlags',
                                          side_effect=lambda event: event[1]))
        self.hardware = self.enterContext(mock.patch.object(
            pipeline, 'CGEventSourceKeyState', return_value=True))
        self.enable = self.enterContext(mock.patch.object(pipeline, 'CGEventTapEnable'))
        self.enterContext(mock.patch.object(pipeline, 'CGEventTapIsEnabled', return_value=True))

    def event(self, keycode, flags, kind=None):
        return self.p.fn_event_callback(None, kind or pipeline.kCGEventFlagsChanged,
                                        (keycode, flags), None)

    def drain(self):
        items = []
        while not self.p._trigger_events.empty():
            items.append(self.p._trigger_events.get_nowait())
        return items

    def run_controller_events(self, events):
        # A finite queue lets us exercise the actual long-lived worker loop.
        self.p._trigger_events = mock.Mock()
        self.p._trigger_events.get.side_effect = [*events, StopIteration]
        with self.assertRaises(StopIteration):
            self.p._recording_controller()

    def test_fn_release_on_another_modifier_event_stops_capture(self):
        self.event(63, FN)
        self.event(56, SHIFT)  # Fn-up was coalesced with Shift's flags event.
        self.assertEqual(self.drain(), [(True, False), (False, True)])
        self.assertFalse(self.p._trigger_is_down)

    def test_unrelated_key_does_not_stop_held_fn(self):
        self.event(63, FN)
        self.event(56, FN | SHIFT)
        self.assertEqual(self.drain(), [(True, False)])
        self.assertTrue(self.p._trigger_is_down)

    def test_shift_is_latched_even_before_stream_opens(self):
        self.event(63, FN)
        self.event(56, FN | SHIFT)
        self.event(56, FN)
        self.event(63, 0)
        self.assertEqual(self.drain(), [(True, False), (False, True)])

    def test_both_tap_disabled_notifications_reenable_and_recover(self):
        for kind in (0xFFFFFFFE, 0xFFFFFFFF):
            with self.subTest(kind=kind):
                self.p._tap_ref[0] = 'tap'
                self.event(63, FN)
                self.p._trigger_down_since -= 1
                self.hardware.return_value = False
                self.p.fn_event_callback(None, kind, None, None)
                self.enable.assert_called_with('tap', True)
                self.assertEqual(self.drain(), [(True, False), (False, False)])

    def test_poll_recovers_missing_release_once(self):
        self.event(63, FN)
        self.p._trigger_down_since -= 1
        self.hardware.return_value = False
        self.p._poll_trigger_release()
        self.p._poll_trigger_release()
        self.event(63, 0)
        self.assertEqual(self.drain(), [(True, False), (False, False)])

    def test_poll_does_not_truncate_press_while_hid_state_catches_up(self):
        with mock.patch.object(pipeline.time, 'monotonic', return_value=10):
            self.event(63, FN)
            self.hardware.return_value = False
            self.p._poll_trigger_release()
        self.assertEqual(self.drain(), [(True, False)])
        self.event(63, 0)  # Explicit release has no grace period.
        self.assertEqual(self.drain(), [(False, False)])

    def test_watchdog_repairs_silently_disabled_tap(self):
        self.p._tap_ref[0] = 'tap'
        with mock.patch.object(pipeline, 'CGEventTapIsEnabled', return_value=False):
            self.p._poll_trigger_release()
        self.enable.assert_called_once_with('tap', True)

    def test_stale_hid_down_has_finite_capture_limit_and_rearms_on_release(self):
        self.event(63, FN | SHIFT)
        self.p._trigger_down_since -= pipeline.MAX_RECORD_SECONDS + 1
        self.p._poll_trigger_release()  # Hardware incorrectly remains down.
        self.event(63, FN)  # No restart from repeated down/flags events.
        self.assertEqual(self.drain(), [(True, True), (False, True)])
        self.hardware.return_value = False
        self.p._poll_trigger_release()
        self.event(63, FN)
        self.assertEqual(self.drain(), [(True, False)])

    def test_assigned_normal_key_recovers_dropped_keyup(self):
        state.transcribe_keycode = 0
        self.event(0, 0, pipeline.kCGEventKeyDown)
        self.p._trigger_down_since -= 1
        self.hardware.return_value = False
        self.p._poll_trigger_release()
        self.assertEqual(self.drain(), [(True, False), (False, False)])
        self.hardware.assert_called_with(pipeline.kCGEventSourceStateHIDSystemState, 0)

    def test_assignment_with_missing_release_can_be_used_again(self):
        state.listening_for_key = True
        self.event(0, 0, pipeline.kCGEventKeyDown)
        self.assertEqual(self.drain(), [])
        self.p._trigger_down_since -= 1
        self.hardware.return_value = False
        self.p._poll_trigger_release()
        self.event(0, 0, pipeline.kCGEventKeyDown)
        self.assertEqual(self.drain(), [(True, False)])

    def test_queue_order_is_atomic_against_concurrent_release(self):
        original_put = self.p._trigger_events.put
        started, attempting_release = threading.Event(), threading.Event()

        def slow_put(item):
            if item[0]:
                started.set()
                self.assertTrue(attempting_release.wait(2))
            original_put(item)

        def release():
            started.wait(2)
            attempting_release.set()
            self.p._set_trigger_pressed(False)

        worker = threading.Thread(target=release)
        worker.start()
        with mock.patch.object(self.p._trigger_events, 'put', side_effect=slow_put):
            self.p._set_trigger_pressed(True)
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.drain(), [(True, False), (False, False)])

    def test_abort_failure_still_closes_stream_and_resets_recording(self):
        stream = mock.Mock()
        stream.abort.side_effect = RuntimeError('device disconnected')
        self.p._stream = stream
        self.p._recording = True
        state.audio_level = 1.0
        with self.assertRaises(RuntimeError):
            self.p._close_audio_stream()
        stream.close.assert_called_once()
        self.assertFalse(self.p._recording)
        self.assertIsNone(self.p._stream)
        self.assertEqual(state.audio_level, 0)

    def test_controller_survives_device_error_and_handles_next_recording(self):
        stream = mock.Mock()
        with mock.patch.object(pipeline.sd, 'InputStream', side_effect=[RuntimeError('no mic'), stream]):
            self.run_controller_events([(True, False), (False, False), (True, False)])
        self.assertTrue(self.p._recording)
        stream.start.assert_called_once()
        self.assertEqual(self.p._trigger_events.task_done.call_count, 3)
        self.widget.assert_any_call('idle')

    def test_controller_survives_stream_start_failure_and_closes_partial_stream(self):
        stream = mock.Mock()
        stream.start.side_effect = RuntimeError('start failed')
        with mock.patch.object(pipeline.sd, 'InputStream', return_value=stream):
            self.run_controller_events([(True, False), (False, False)])
        stream.close.assert_called_once()
        self.assertFalse(self.p._recording)
        self.widget.assert_called_with('idle')

    def test_empty_capture_stops_and_is_processed_as_one_dimensional_audio(self):
        stream = mock.Mock()
        self.p._stream = stream
        self.p._recording = True
        with mock.patch.object(pipeline.threading, 'Thread') as thread:
            self.p._handle_trigger_transition(False, False)
        audio = thread.call_args.kwargs['args'][0]
        self.assertEqual(audio.shape, (0,))
        self.assertFalse(self.p._recording)
        stream.abort.assert_called_once()
        thread.return_value.start.assert_called_once()

    def test_quick_release_queued_before_stream_opens_is_not_lost(self):
        self.event(63, FN)
        self.event(63, 0)
        stream = mock.Mock()
        with mock.patch.object(pipeline.sd, 'InputStream', return_value=stream):
            with mock.patch.object(pipeline.threading, 'Thread'):
                self.run_controller_events(self.drain())
        stream.start.assert_called_once()
        stream.abort.assert_called_once()
        stream.close.assert_called_once()
        self.assertFalse(self.p._recording)

    def test_controller_survives_close_error_and_accepts_next_press(self):
        first, second = mock.Mock(), mock.Mock()
        first.abort.side_effect = RuntimeError('device disconnected')
        with mock.patch.object(pipeline.sd, 'InputStream', side_effect=[first, second]):
            self.run_controller_events([(True, False), (False, False), (True, False)])
        first.close.assert_called_once()
        second.start.assert_called_once()
        self.assertTrue(self.p._recording)

    def test_command_selection_is_fetched_only_after_microphone_is_closed(self):
        stream = mock.Mock()
        self.p._stream = stream
        self.p._recording = True
        self.p._frames = [np.zeros((pipeline.SAMPLE_RATE, 1), dtype=np.float32)]

        def selection():
            self.assertTrue(stream.close.called)
            self.assertFalse(self.p._recording)
            return 'selected text'

        self.p.get_selected_text = selection
        self.p.command_and_paste = mock.Mock()
        with mock.patch.object(pipeline.threading, 'Thread') as worker:
            self.p._handle_trigger_transition(False, True)
        with mock.patch.object(pipeline.time, 'sleep'):
            worker.call_args.kwargs['target'](*worker.call_args.kwargs['args'])
        self.p.command_and_paste.assert_called_once()
        self.assertEqual(self.p.command_and_paste.call_args.args[1], 'selected text')
        self.assertFalse(self.p._processing)

    def test_busy_pipeline_does_not_launch_overlapping_model_requests(self):
        self.p._processing = True
        with mock.patch.object(self.p, '_open_audio_stream') as start:
            self.p._handle_trigger_transition(True, False)
            self.p._handle_trigger_transition(False, False)
        start.assert_not_called()

    def test_processing_failure_always_restores_idle_and_allows_retry(self):
        for mode in ('transcribe', 'command'):
            with self.subTest(mode=mode):
                self.p._processing = True
                self.p.transcribe_and_paste = mock.Mock(side_effect=RuntimeError('Whisper failed'))
                self.p.get_selected_text = mock.Mock(side_effect=RuntimeError('selection failed'))
                with mock.patch.object(pipeline.time, 'sleep'):
                    self.p._process_audio(np.zeros(pipeline.SAMPLE_RATE), mode)
                self.assertFalse(self.p._processing)
                self.widget.assert_called_with('idle')

    def test_formatting_failure_preserves_raw_transcript(self):
        self.p._pipe[0] = mock.Mock(return_value={'text': 'keep these words'})
        self.p._generate_text = mock.Mock(side_effect=TimeoutError('Gemma stalled'))
        self.p.paste_at_cursor = mock.Mock()
        self.p.transcribe_and_paste(np.zeros(pipeline.SAMPLE_RATE))
        self.p.paste_at_cursor.assert_called_once_with('keep these words')

    def test_gemma_failure_does_not_kill_worker_or_strand_next_request(self):
        first, second = queue.Queue(), queue.Queue()
        self.p._gemma_in = mock.Mock()
        self.p._gemma_in.get.side_effect = [
            (('transcribe', 'one'), first), (('transcribe', 'two'), second), StopIteration]
        self.p._gemma_reply = mock.Mock(side_effect=[RuntimeError('generation failed'), 'Two.'])
        ready = threading.Event()
        with mock.patch.object(pipeline, 'vlm_load', return_value=(object(), object())):
            with self.assertRaises(StopIteration):
                self.p._gemma_worker(ready)
        self.assertTrue(ready.is_set())
        self.assertIsInstance(first.get_nowait(), RuntimeError)
        self.assertEqual(second.get_nowait(), 'Two.')
        self.assertEqual(self.p._gemma_in.task_done.call_count, 2)

    def test_gemma_timeout_and_late_reply_do_not_contaminate_next_request(self):
        with mock.patch.object(pipeline, 'GEMMA_TIMEOUT_SECONDS', 0):
            with self.assertRaises(TimeoutError):
                self.p._generate_text(('transcribe', 'old'))
        _, old_response = self.p._gemma_in.get_nowait()
        with self.assertRaisesRegex(RuntimeError, 'previous request'):
            self.p._generate_text(('transcribe', 'still busy'))
        old_response.put('Old result')
        self.p._gemma_pending.release()  # The old model call finally finishes.

        def answer_new_request(item):
            _, response = item
            response.put('New result')

        with mock.patch.object(self.p._gemma_in, 'put', side_effect=answer_new_request):
            self.assertEqual(self.p._generate_text(('transcribe', 'new')), 'New result')

    def test_empty_whisper_result_never_reaches_gemma_or_paste(self):
        self.p._pipe[0] = mock.Mock(return_value={'text': '   '})
        self.p._generate_text = mock.Mock()
        self.p.paste_at_cursor = mock.Mock()
        self.p.transcribe_and_paste(np.zeros(pipeline.SAMPLE_RATE))
        self.p._generate_text.assert_not_called()
        self.p.paste_at_cursor.assert_not_called()
        self.widget.assert_called_with('active', 'No speech detected')

    def test_empty_formatter_response_keeps_transcript(self):
        self.p._pipe[0] = mock.Mock(return_value={'text': 'keep this'})
        self.p._generate_text = mock.Mock(return_value='')
        self.p.paste_at_cursor = mock.Mock()
        self.p.transcribe_and_paste(np.zeros(pipeline.SAMPLE_RATE))
        self.p.paste_at_cursor.assert_called_once_with('keep this')

    def test_whisper_timeout_prevents_overlapping_native_calls(self):
        entered, finish = threading.Event(), threading.Event()
        def stalled(*args, **kwargs):
            entered.set()
            finish.wait(2)
            return {'text': 'late'}
        self.p._pipe[0] = mock.Mock(side_effect=stalled)
        try:
            with mock.patch.object(pipeline, 'WHISPER_TIMEOUT_SECONDS', 0.02):
                with self.assertRaises(TimeoutError):
                    self.p._recognize_audio(np.zeros(10))
            self.assertTrue(entered.is_set())
            with self.assertRaisesRegex(RuntimeError, 'previous request'):
                self.p._recognize_audio(np.zeros(10))
            self.p._pipe[0].assert_called_once()
        finally:
            finish.set()

    def test_whisper_exception_propagates_and_releases_busy_guard(self):
        self.p._pipe[0] = mock.Mock(side_effect=RuntimeError('inference failure'))
        with self.assertRaisesRegex(RuntimeError, 'inference failure'):
            self.p._recognize_audio(np.zeros(10))
        self.assertFalse(self.p._whisper_pending.locked())

    def test_gemma_load_failure_signals_readiness_error_instead_of_hanging(self):
        ready = threading.Event()
        with mock.patch.object(pipeline, 'vlm_load', side_effect=RuntimeError('download failed')):
            self.p._gemma_worker(ready)
        self.assertTrue(ready.is_set())
        name, error = self.p._load_errors.get_nowait()
        self.assertEqual(name, 'Gemma')
        self.assertIsInstance(error, RuntimeError)

    def test_model_load_error_never_enables_recording(self):
        def fail_loader():
            self.p._load_errors.put(('Whisper', RuntimeError('bad weights')))
        with mock.patch.object(pipeline.threading, 'Thread') as worker:
            worker.return_value.start.side_effect = fail_loader
            self.p._models_ready[0] = False
            self.p.load_models()
        self.assertFalse(self.p._models_ready[0])
        self.widget.assert_called_with('error', 'Whisper load failed')

    def test_model_load_timeout_shows_error_instead_of_loading_forever(self):
        self.p._models_ready[0] = False
        with mock.patch.object(pipeline.threading, 'Thread'):
            with mock.patch.object(pipeline, 'MODEL_LOAD_TIMEOUT_SECONDS', 0):
                self.p.load_models()
        self.assertFalse(self.p._models_ready[0])
        self.widget.assert_called_with('error', 'Model load failed')

    def test_key_assignment_escape_cancels_and_queues_ui_callback(self):
        state.listening_for_key = True
        state.on_key_assigned = mock.Mock()
        self.event(53, 0, pipeline.kCGEventKeyDown)
        self.assertFalse(state.listening_for_key)
        self.assertEqual(state.transcribe_keycode, 63)
        state.on_key_assigned.assert_not_called()
        self.assertEqual(state.ui_callbacks.get_nowait(), (state.on_key_assigned, 63))

    def test_modifier_release_does_not_get_assigned_as_new_hotkey(self):
        state.listening_for_key = True
        self.event(56, 0)
        self.assertTrue(state.listening_for_key)
        self.assertEqual(state.transcribe_keycode, 63)

    def test_synthetic_shortcuts_do_not_trigger_recording(self):
        with mock.patch.object(pipeline, 'CGEventGetIntegerValueField',
                               side_effect=lambda event, field: pipeline.os.getpid() if field == 41 else 63):
            self.event(63, FN)
        self.assertEqual(self.drain(), [])

    def test_inactive_audio_stream_recovers_while_trigger_remains_held(self):
        self.event(63, FN)
        self.p._stream = mock.Mock(active=False)
        self.p._recording = True
        self.p._stream_started_at -= 1
        self.p._poll_trigger_release()
        self.assertEqual(self.drain(), [(True, False), (False, False)])

    def test_caps_lock_trigger_records_even_when_toggling_flag_off(self):
        state.transcribe_keycode = 57
        self.event(57, 0)
        self.assertEqual(self.drain(), [(True, False)])

    def test_shutdown_closes_microphone_disables_tap_and_prevents_new_capture(self):
        stream = mock.Mock()
        self.p._stream = stream
        self.p._recording = True
        self.p._tap_ref[0] = 'tap'
        self.p.shutdown()
        self.p.shutdown()  # idempotent
        stream.abort.assert_called_once()
        stream.close.assert_called_once()
        self.enable.assert_called_with('tap', False)
        with mock.patch.object(self.p, '_open_audio_stream') as start:
            self.p._handle_trigger_transition(True, False)
        start.assert_not_called()

    def test_shortcuts_wait_for_modifiers_and_suppress_injected_triggers(self):
        keyboard = mock.Mock()
        keyboard.pressed.return_value.__enter__ = mock.Mock()
        keyboard.pressed.return_value.__exit__ = mock.Mock(return_value=False)
        with mock.patch.object(pipeline, 'CGEventSourceFlagsState', side_effect=[SHIFT, 0, 0]):
            with mock.patch.object(pipeline.time, 'sleep') as wait:
                with mock.patch.object(pipeline, 'KeyboardController', return_value=keyboard):
                    self.p._send_shortcut('v')
        wait.assert_called_once_with(0.01)
        keyboard.press.assert_called_once_with('v')
        self.assertFalse(self.p._injecting_keys)

    def test_shutdown_worker_sentinel_exits_cleanly(self):
        self.p.shutdown()
        self.p._recording_controller()
        self.assertEqual(self.p._trigger_events.unfinished_tasks, 0)

    def test_formatter_cannot_silently_drop_or_add_dictated_words(self):
        for reply in ('these words', 'Here is your text: keep all these words'):
            with self.subTest(reply=reply):
                self.p._recognize_audio = mock.Mock(return_value='keep all these words')
                self.p._generate_text = mock.Mock(return_value=reply)
                self.p.paste_at_cursor = mock.Mock()
                self.p.transcribe_and_paste(np.zeros(10))
                self.p.paste_at_cursor.assert_called_once_with('keep all these words')

    def test_formatter_can_correct_punctuation_capitals_and_apostrophe_style(self):
        self.p._recognize_audio = mock.Mock(return_value="it's ready keep going")
        self.p._generate_text = mock.Mock(return_value='It’s ready, keep going!')
        self.p.paste_at_cursor = mock.Mock()
        self.p.transcribe_and_paste(np.zeros(10))
        self.p.paste_at_cursor.assert_called_once_with('It’s ready, keep going!')

    def test_accessibility_empty_selection_does_not_copy_whole_editor_line(self):
        with mock.patch.object(pipeline.clipboard, 'accessibility_selection', return_value=''):
            with mock.patch.object(pipeline.clipboard, 'selected_text') as copy:
                self.assertEqual(self.p.get_selected_text(), '')
        copy.assert_not_called()

    def test_unsupported_accessibility_selection_falls_back_to_clipboard(self):
        with mock.patch.object(pipeline.clipboard, 'accessibility_selection', return_value=None):
            with mock.patch.object(pipeline.clipboard, 'selected_text', return_value='selected'):
                self.assertEqual(self.p.get_selected_text(), 'selected')

    def test_caps_lock_toggle_off_is_assignable_and_other_flags_do_not_release_it(self):
        state.listening_for_key = True
        self.event(57, 0)
        self.assertEqual(state.transcribe_keycode, 57)
        self.p._trigger_down_since -= 1
        self.hardware.return_value = False
        self.p._poll_trigger_release()
        self.hardware.return_value = True
        self.event(57, 0)
        self.event(56, 0)
        self.assertEqual(self.drain(), [(True, False)])


if __name__ == '__main__':
    unittest.main()
