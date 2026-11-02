import os
import sys
import time
import queue
import threading
import subprocess
import datetime
import logging
import re

import numpy as np
import sounddevice as sd
import torch
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Mock AutoTokenizer.register to prevent crash in mlx_lm with transformers >= 5.13
_original_register = AutoTokenizer.register
def _safe_register(config_class, *args, **kwargs):
    if isinstance(config_class, str):
        return
    return _original_register(config_class, *args, **kwargs)
AutoTokenizer.register = _safe_register

from mlx_vlm import load as vlm_load, generate as vlm_generate
from pynput.keyboard import Key, Controller as KeyboardController

import Cocoa
from Quartz import (
    CGEventTapCreate, CGEventTapEnable, kCGSessionEventTap, kCGHeadInsertEventTap,
    kCGEventTapOptionDefault, CGEventMaskBit, kCGEventFlagsChanged,
    CFRunLoopAddSource, kCFRunLoopCommonModes, CFRunLoopGetMain,
    CGEventGetIntegerValueField, CGEventGetFlags,
    CGEventSourceKeyState, CGEventSourceFlagsState,
    kCGEventSourceStateHIDSystemState,
    kCGEventKeyDown, kCGEventKeyUp,
    kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput,
    CGEventTapIsEnabled,
    kCGEventSourceUnixProcessID,
)

from . import state, clipboard
from .config import (
    SAMPLE_RATE, WHISPER_MODEL_ID, GEMMA_MODEL_ID,
    kCGKeyboardEventKeycode, kCGEventFlagMaskSecondaryFn, kCGEventFlagMaskShift,
)
from .ui.widgets import set_widget

logger = logging.getLogger(__name__)
MAX_RECORD_SECONDS = 120
GEMMA_TIMEOUT_SECONDS = 120
WHISPER_TIMEOUT_SECONDS = 120
MODEL_LOAD_TIMEOUT_SECONDS = 1800


class Pipeline:
    def __init__(self):
        self._recording    = False
        self._frames       = []
        self._stream       = None
        self._lock         = threading.Lock()
        self._trigger_lock = threading.RLock()
        self._processing   = False
        self._stopping = threading.Event()
        self._injecting_keys = False
        self._load_errors = queue.Queue()
        self._stream_started_at = 0.0
        self._shift_held   = False
        self._models_ready = [False]
        self._pipe         = [None]
        self._tap_ref      = [None]
        self._gemma_in     = queue.Queue()
        self._gemma_pending = threading.Lock()
        self._whisper_pending = threading.Lock()
        self._trigger_events = queue.Queue()
        self._trigger_is_down = False
        self._ignore_trigger_until_release = False
        self._command_mode = False
        self._trigger_down_since = 0.0
        self._trigger_command_mode = False

    @staticmethod
    def _modifier_mask_for_keycode(keycode):
        return {
            54: 0x00100000, 55: 0x00100000,  # Command
            56: kCGEventFlagMaskShift, 60: kCGEventFlagMaskShift,
            57: 0x00010000,                  # Caps Lock
            58: 0x00080000, 61: 0x00080000,  # Option
            59: 0x00040000, 62: 0x00040000,  # Control
            63: kCGEventFlagMaskSecondaryFn,
        }.get(keycode)

    def _set_trigger_pressed(self, pressed, shift_held=None):
        """Queue a trigger transition once, regardless of its source."""
        with self._trigger_lock:
            if not pressed:
                self._ignore_trigger_until_release = False
            if pressed == self._trigger_is_down or self._ignore_trigger_until_release:
                return
            self._trigger_is_down = pressed
            if pressed:
                self._trigger_down_since = time.monotonic()
            if shift_held is None:
                shift_held = self._shift_held
            if pressed:
                self._trigger_command_mode = shift_held
            else:
                shift_held = shift_held or self._trigger_command_mode
                self._trigger_command_mode = False
            # Keep the state change and queue insertion atomic with respect to
            # the tap and watchdog; a stale poll must not stop the next press.
            self._trigger_events.put((pressed, shift_held))

    def _trigger_is_physically_down(self):
        # The HID state follows the physical key for Fn, modifiers, and normal
        # keys. Combined session state can report a held assigned key as up.
        return bool(CGEventSourceKeyState(
            kCGEventSourceStateHIDSystemState, state.transcribe_keycode
        ))

    def _poll_trigger_release(self):
        """Reconcile hardware state even if the tap never delivers key-up."""
        if self._tap_ref[0] and not CGEventTapIsEnabled(self._tap_ref[0]):
            CGEventTapEnable(self._tap_ref[0], True)
        with self._trigger_lock:
            if state.listening_for_key:
                return
            if not (self._trigger_is_down or self._ignore_trigger_until_release):
                return
            elapsed = time.monotonic() - self._trigger_down_since
            stream = self._stream
            if (self._recording and stream is not None
                    and time.monotonic() - self._stream_started_at >= 0.15
                    and not stream.active):
                logger.warning("Audio device stopped unexpectedly")
                self._set_trigger_pressed(False)
                self._ignore_trigger_until_release = True
                return
            # HID state can lag the press callback briefly. Explicit release
            # events still stop immediately, including very short taps.
            if elapsed >= 0.15 and not self._trigger_is_physically_down():
                self._set_trigger_pressed(False)
            elif self._trigger_is_down and elapsed >= MAX_RECORD_SECONDS:
                logger.warning("Recording limit reached; recovering missing trigger release")
                self._set_trigger_pressed(False)
                self._ignore_trigger_until_release = True

    def _watch_trigger_release(self):
        while not self._stopping.is_set():
            try:
                self._poll_trigger_release()
            except Exception:
                logger.exception("Trigger watchdog failed; retrying")
            self._stopping.wait(0.05)

    def _generate_text(self, task):
        # Replies belong to a request. A timed-out result must never be pasted
        # as the response to a later recording.
        if not self._gemma_pending.acquire(blocking=False):
            raise RuntimeError("Gemma is still finishing a previous request; restart if it remains stuck")
        response = queue.Queue(maxsize=1)
        self._gemma_in.put((task, response))
        try:
            result = response.get(timeout=GEMMA_TIMEOUT_SECONDS)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Gemma did not respond within {GEMMA_TIMEOUT_SECONDS} seconds"
            ) from exc
        if isinstance(result, Exception):
            raise result
        return result

    def _recognize_audio(self, audio):
        if not self._whisper_pending.acquire(blocking=False):
            raise RuntimeError("Whisper is still finishing a previous request; restart if it remains stuck")
        response = queue.Queue(maxsize=1)

        def infer():
            try:
                result = self._pipe[0](
                    {"array": audio, "sampling_rate": SAMPLE_RATE}, return_timestamps=True)
            except Exception as exc:
                result = exc
            finally:
                self._whisper_pending.release()
            response.put(result)

        try:
            threading.Thread(target=infer, daemon=True).start()
        except Exception:
            self._whisper_pending.release()
            raise
        try:
            result = response.get(timeout=WHISPER_TIMEOUT_SECONDS)
        except queue.Empty as exc:
            raise TimeoutError("Whisper transcription timed out; restart if it remains stuck") from exc
        if isinstance(result, Exception):
            raise result
        return result["text"].strip()

    def _send_shortcut(self, key):
        if self._stopping.is_set():
            raise RuntimeError("Fiveflow is shutting down")
        # If a modifier is the trigger, wait until it is physically released so
        # Shift+Cmd+V cannot turn a paste into another application shortcut.
        deadline = time.monotonic() + 2
        modifier_mask = (kCGEventFlagMaskShift | kCGEventFlagMaskSecondaryFn
                         | 0x00100000 | 0x00080000 | 0x00040000)

        def modifiers_down():
            return bool(CGEventSourceFlagsState(kCGEventSourceStateHIDSystemState) & modifier_mask)

        while modifiers_down() and time.monotonic() < deadline:
            if self._stopping.is_set():
                raise RuntimeError("Fiveflow is shutting down")
            time.sleep(0.01)
        if modifiers_down():
            raise RuntimeError("Release modifier keys before copying or pasting")
        with self._trigger_lock:
            self._injecting_keys = True
        try:
            kb = KeyboardController()
            with kb.pressed(Key.cmd):
                kb.press(key)
                kb.release(key)
        finally:
            with self._trigger_lock:
                self._injecting_keys = False

    def paste_at_cursor(self, text):
        clipboard.paste_text(text, lambda: self._send_shortcut('v'))

    def get_selected_text(self):
        selected = clipboard.accessibility_selection()
        if selected is not None:
            return selected
        return clipboard.selected_text(lambda: self._send_shortcut('c'))

    def transcribe_and_paste(self, audio):
        set_widget('active', 'Transcribing...')
        transcript = self._recognize_audio(audio)
        if not transcript:
            set_widget('active', 'No speech detected')
            return

        set_widget('active', 'Formatting...')
        try:
            formatted = self._generate_text(('transcribe', transcript)) or transcript
        except Exception:
            logger.exception("Formatting failed; preserving the Whisper transcript")
            formatted = transcript
        # The editor is allowed to change punctuation/capitalization only.
        # In particular, never paste a model response cut off by its token cap.
        def words(text):
            return re.findall(r"\w+(?:'\w+)*", text.replace('’', "'").casefold())

        if words(formatted) != words(transcript):
            logger.warning("Formatter changed or omitted words; preserving the transcript")
            formatted = transcript

        state.add_history({
            'time': datetime.datetime.now(), 'mode': 'transcribe', 'text': formatted,
        })
        self.paste_at_cursor(formatted)
        set_widget('active', 'Done')

    def command_and_paste(self, audio, selected):
        set_widget('active', 'Transcribing...')
        command = self._recognize_audio(audio)
        if not command:
            set_widget('active', 'No speech detected')
            return

        set_widget('active', 'Thinking...')
        output = self._generate_text(('command', command, selected))
        if not output:
            raise RuntimeError('The command model returned an empty response')

        state.add_history({
            'time': datetime.datetime.now(), 'mode': 'command', 'command': command, 'text': output,
        })
        self.paste_at_cursor(output)
        set_widget('active', 'Done')

    def _open_audio_stream(self):
        self._frames = []
        frames = self._frames

        def audio_callback(indata, _frame_count, _time_info, _status):
            frames.append(indata.copy())
            rms = float(np.sqrt(np.mean(indata ** 2)))
            state.audio_level = min(1.0, rms * 15)

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fd2  = os.dup(2)
        os.dup2(devnull_fd, 2)
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=audio_callback,
            )
            self._stream.start()
        finally:
            os.dup2(saved_fd2, 2)
            os.close(saved_fd2)
            os.close(devnull_fd)
        self._recording = True
        self._stream_started_at = time.monotonic()
        set_widget('recording')

    def _close_audio_stream(self):
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                try:
                    # Input has no output buffers to drain; stop capture now.
                    stream.abort(ignore_errors=False)
                finally:
                    stream.close()
        finally:
            state.audio_level = 0.0
            self._recording = False

    def _process_audio(self, audio, mode):
        try:
            if len(audio) < SAMPLE_RATE:
                set_widget('active', 'Too short')
            elif mode == 'command':
                self.command_and_paste(audio, self.get_selected_text())
            else:
                self.transcribe_and_paste(audio)
        except Exception:
            logger.exception("Audio processing failed")
            set_widget('active', 'Processing failed')
        finally:
            time.sleep(1.5)
            set_widget('idle')
            with self._lock:
                self._processing = False

    def _handle_trigger_transition(self, pressed, shift_held):
        if not self._models_ready[0] or self._stopping.is_set():
            return
        with self._lock:
            if pressed and not self._recording and not self._processing:
                self._command_mode = shift_held
                self._open_audio_stream()
                print("Recording started.")
            elif not pressed and self._recording:
                mode = 'command' if self._command_mode or shift_held else 'transcribe'
                self._close_audio_stream()
                print(f"Recording stopped. mode={mode}")
                audio = (
                    np.concatenate(self._frames, axis=0).reshape(-1)
                    if self._frames
                    else np.empty(0, dtype=np.float32)
                )
                self._frames = []
                self._command_mode = False
                self._processing = True
                set_widget('active', 'Transcribing...' if audio.size >= SAMPLE_RATE else 'Too short')
                threading.Thread(target=self._process_audio, args=(audio, mode), daemon=True).start()

    def _recording_controller(self):
        while True:
            transition = self._trigger_events.get()
            if transition is None:
                self._trigger_events.task_done()
                return
            pressed, shift_held = transition
            try:
                self._handle_trigger_transition(pressed, shift_held)
            except Exception:
                # One PortAudio/selection failure must not kill the only thread
                # that can stop future recordings.
                logger.exception("Recording transition failed")
                try:
                    self._close_audio_stream()
                except Exception:
                    logger.exception("Audio stream cleanup failed")
                with self._lock:
                    self._command_mode = False
                    self._processing = False
                    self._frames = []
                set_widget('idle')
            finally:
                self._trigger_events.task_done()

    def fn_event_callback(self, proxy, event_type, event, refcon):
        if event_type in (kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput):
            if self._tap_ref[0]:
                CGEventTapEnable(self._tap_ref[0], True)
            # A key-up event may have occurred while the tap was disabled.
            self._poll_trigger_release()
            return event

        if self._stopping.is_set() or self._injecting_keys:
            return event
        if CGEventGetIntegerValueField(event, kCGEventSourceUnixProcessID) == os.getpid():
            return event
        keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        flags = CGEventGetFlags(event)
        self._shift_held = bool(flags & kCGEventFlagMaskShift)
        with self._trigger_lock:
            if self._trigger_is_down and self._shift_held:
                self._trigger_command_mode = True

        if state.listening_for_key:
            if keycode == 53 and event_type == kCGEventKeyDown:
                state.listening_for_key = False
                if callable(state.on_key_assigned):
                    state.ui_callbacks.put((state.on_key_assigned, state.transcribe_keycode))
                return None
            if event_type == kCGEventFlagsChanged:
                mask = self._modifier_mask_for_keycode(keycode)
                if keycode != 57 and mask is not None and not (flags & mask):
                    return event
            if event_type in (kCGEventKeyDown, kCGEventFlagsChanged):
                state.transcribe_keycode = keycode
                state.listening_for_key = False
                with self._trigger_lock:
                    self._set_trigger_pressed(False)
                    self._ignore_trigger_until_release = True
                    self._trigger_down_since = time.monotonic()
                print(f"New trigger key set: {keycode}")
                if callable(state.on_key_assigned):
                    state.ui_callbacks.put((state.on_key_assigned, keycode))
                return None
            return event

        if keycode != state.transcribe_keycode:
            # Fn-up may be coalesced into another modifier's flags event.
            # Its flag disappearing is release evidence even with another keycode.
            modifier_mask = self._modifier_mask_for_keycode(state.transcribe_keycode)
            if (state.transcribe_keycode != 57 and event_type == kCGEventFlagsChanged and modifier_mask is not None
                    and not (flags & modifier_mask)):
                self._set_trigger_pressed(False, self._shift_held)
            return event

        if self._ignore_trigger_until_release:
            if event_type == kCGEventKeyUp:
                self._ignore_trigger_until_release = False
            elif event_type == kCGEventFlagsChanged:
                modifier_mask = self._modifier_mask_for_keycode(keycode)
                if modifier_mask is not None and not (flags & modifier_mask):
                    self._ignore_trigger_until_release = False
            return None

        if event_type == kCGEventKeyDown and not self._trigger_is_down:
            self._set_trigger_pressed(True, self._shift_held)
        elif event_type == kCGEventKeyUp and self._trigger_is_down:
            self._set_trigger_pressed(False, self._shift_held)
        elif event_type == kCGEventFlagsChanged:
            # Modifier keys, including Fn, do not emit key down/up events.
            modifier_mask = self._modifier_mask_for_keycode(keycode)
            if modifier_mask is not None:
                pressed = bool(flags & modifier_mask)
                if keycode == 57:  # Caps Lock's flag is a toggle, not key-down state.
                    pressed = True
                self._set_trigger_pressed(pressed, self._shift_held)
        return None

    def setup_event_tap(self):
        tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionDefault,
            CGEventMaskBit(kCGEventFlagsChanged) | CGEventMaskBit(kCGEventKeyDown) | CGEventMaskBit(kCGEventKeyUp),
            self.fn_event_callback,
            None,
        )
        if not tap:
            print("CGEventTap failed - ensure Accessibility permission is granted.")
            sys.exit(1)
        self._tap_ref[0] = tap
        CFRunLoopAddSource(
            CFRunLoopGetMain(),
            Cocoa.CFMachPortCreateRunLoopSource(None, tap, 0),
            kCFRunLoopCommonModes,
        )

    def _gemma_reply(self, g_model, g_proc, task):
        if task[0] == 'transcribe':
            sys_prompt = (
                "You are a transcription editor for spoken English. "
                "Your only job is to fix punctuation and sentence flow. "
                "You must not change, add, or remove any words.\n\n"
                "Rules:\n"
                "- When two phrases are part of the same continuous thought, join them with a comma, "
                "em-dash, semicolon, or conjunction - never a full stop\n"
                "- Only place a full stop when the thought is genuinely complete and the next clause "
                "is an independent, separate idea\n"
                "- Do not rephrase, restructure, or paraphrase anything\n"
                "- Do not add any words, filler, or commentary that was not in the original\n"
                "- Return only the corrected transcription, nothing else"
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": task[1]},
            ]
            max_tok, temp = 384, 0.15
        else:
            command, selected = task[1], task[2]
            sys_prompt = (
                "You are a precise voice command executor. "
                "Execute the user's command exactly and return only the result - "
                "no preamble, label, explanation, or commentary of any kind.\n\n"
                "You can handle any text task, including but not limited to:\n"
                "- Rewrite / rephrase / paraphrase\n"
                "- Make formal, casual, concise, assertive, empathetic, or humorous\n"
                "- Fix grammar, spelling, punctuation, or clarity\n"
                "- Summarise, expand, shorten, or elaborate\n"
                "- Convert format: bullet points, numbered list, table, paragraph, markdown, plain text\n"
                "- Translate to any language\n"
                "- Explain, simplify, or add detail\n"
                "- Refactor, explain, comment, or convert code\n"
                "- Extract key points, action items, names, dates, or any other structured data\n"
                "- Continue, complete, or extend a piece of text\n"
                "- Answer a question or generate fresh content when no text is selected\n\n"
                "If the command is ambiguous, make the most reasonable interpretation without asking. "
                "Preserve formatting of the original text unless the command explicitly changes it."
            )
            if selected:
                user_content = (
                    f"Selected text:\n{selected}\n\n"
                    f"Voice command: {command}"
                )
            else:
                user_content = f"Voice command: {command}"
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_content},
            ]
            max_tok, temp = 1024, 0.4
        prompt = g_proc.apply_chat_template(messages, add_generation_prompt=True)
        reply = vlm_generate(g_model, g_proc, prompt=prompt,
                             max_tokens=max_tok, temperature=temp, verbose=False)
        return reply.text.strip()

    def _gemma_worker(self, ready):
        print(f"Loading {GEMMA_MODEL_ID}...")
        try:
            g_model, g_proc = vlm_load(GEMMA_MODEL_ID)
        except Exception as exc:
            logger.exception("Gemma loading failed")
            self._load_errors.put(("Gemma", exc))
            ready.set()
            return
        print("Gemma ready.")
        ready.set()
        while True:
            request = self._gemma_in.get()
            if request is None:
                self._gemma_in.task_done()
                return
            task, response = request
            try:
                result = self._gemma_reply(g_model, g_proc, task)
            except Exception as exc:
                logger.exception("Gemma request failed")
                result = exc
            finally:
                if self._gemma_pending.locked():
                    self._gemma_pending.release()
                self._gemma_in.task_done()
            response.put(result)

    def load_models(self):
        set_widget('active', 'Loading models...')
        whisper_done = threading.Event()
        gemma_done   = threading.Event()

        def _load_whisper():
            try:
                print("Loading Whisper...")
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                dtype  = torch.float16 if device == "mps" else torch.float32
                w_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    WHISPER_MODEL_ID, dtype=dtype,
                    low_cpu_mem_usage=True, use_safetensors=True,
                )
                w_model.to(device)
                w_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
                self._pipe[0] = pipeline(
                    "automatic-speech-recognition",
                    model=w_model, tokenizer=w_processor.tokenizer,
                    feature_extractor=w_processor.feature_extractor,
                    torch_dtype=dtype, device=device,
                )
                print("Whisper ready.")

            except Exception as exc:
                logger.exception("Whisper loading failed")
                self._load_errors.put(("Whisper", exc))
            finally:
                whisper_done.set()

        threading.Thread(target=_load_whisper, daemon=True).start()
        threading.Thread(target=self._gemma_worker, args=(gemma_done,), daemon=True).start()

        deadline = time.monotonic() + MODEL_LOAD_TIMEOUT_SECONDS
        while not (whisper_done.is_set() and gemma_done.is_set()):
            if self._stopping.wait(0.1):
                return
            if not self._load_errors.empty():
                break
            if time.monotonic() >= deadline:
                self._load_errors.put(("Model", TimeoutError("Model loading timed out")))
                break
            if whisper_done.is_set() and not gemma_done.is_set():
                set_widget('active', 'Loading Gemma...')
            elif gemma_done.is_set() and not whisper_done.is_set():
                set_widget('active', 'Loading Whisper...')
        if not self._load_errors.empty():
            name, error = self._load_errors.get_nowait()
            logger.error("%s could not load: %s", name, error)
            set_widget('error', f'{name} load failed')
            return
        if self._stopping.is_set():
            return

        self._models_ready[0] = True
        set_widget('idle')
        print("Models ready. Hold Fn to record, release to transcribe.\n")
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def shutdown(self):
        if self._stopping.is_set():
            return
        self._stopping.set()
        if self._tap_ref[0]:
            CGEventTapEnable(self._tap_ref[0], False)
        self._trigger_events.put(None)
        self._gemma_in.put(None)
        with self._lock:
            try:
                self._close_audio_stream()
            except Exception:
                logger.exception("Could not close the audio device during shutdown")
        state.listening_for_key = False

    def run(self):
        self.setup_event_tap()
        threading.Thread(target=self._recording_controller, daemon=True).start()
        threading.Thread(target=self._watch_trigger_release, daemon=True).start()
        set_widget('active', 'Loading models...')
        threading.Thread(target=self.load_models, daemon=True).start()

        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        from .ui.app_delegate import AppDelegate

        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        delegate = AppDelegate.alloc().init()
        delegate._pipeline = self
        app.setDelegate_(delegate)
        try:
            app.run()
        finally:
            self.shutdown()
