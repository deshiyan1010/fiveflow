import os
import sys
import time
import queue
import threading
import subprocess
import datetime
import logging

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
    CGEventSourceKeyState,
    kCGEventSourceStateHIDSystemState,
    kCGEventKeyDown, kCGEventKeyUp,
    kCGEventTapDisabledByTimeout, kCGEventTapDisabledByUserInput,
    CGEventTapIsEnabled,
)

from . import state
from .config import (
    SAMPLE_RATE, WHISPER_MODEL_ID, GEMMA_MODEL_ID,
    kCGKeyboardEventKeycode, kCGEventFlagMaskSecondaryFn, kCGEventFlagMaskShift,
)
from .ui.widgets import set_widget

logger = logging.getLogger(__name__)
MAX_RECORD_SECONDS = 120
GEMMA_TIMEOUT_SECONDS = 120


class Pipeline:
    def __init__(self):
        self._recording    = False
        self._frames       = []
        self._stream       = None
        self._lock         = threading.Lock()
        self._trigger_lock = threading.RLock()
        self._processing   = False
        self._shift_held   = False
        self._models_ready = [False]
        self._pipe         = [None]
        self._tap_ref      = [None]
        self._gemma_in     = queue.Queue()
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
            # HID state can lag the press callback briefly. Explicit release
            # events still stop immediately, including very short taps.
            if elapsed >= 0.15 and not self._trigger_is_physically_down():
                self._set_trigger_pressed(False)
            elif self._trigger_is_down and elapsed >= MAX_RECORD_SECONDS:
                logger.warning("Recording limit reached; recovering missing trigger release")
                self._set_trigger_pressed(False)
                self._ignore_trigger_until_release = True

    def _watch_trigger_release(self):
        while True:
            try:
                self._poll_trigger_release()
            except Exception:
                logger.exception("Trigger watchdog failed; retrying")
            time.sleep(0.05)

    def _generate_text(self, task):
        # Replies belong to a request. A timed-out result must never be pasted
        # as the response to a later recording.
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

    def paste_at_cursor(self, text):
        original = subprocess.run(["pbpaste"], capture_output=True).stdout
        subprocess.run(["pbcopy"], input=text.encode())
        time.sleep(0.05)
        kb = KeyboardController()
        with kb.pressed(Key.cmd):
            kb.press("v")
            kb.release("v")
        time.sleep(0.1)
        subprocess.run(["pbcopy"], input=original)

    def get_selected_text(self):
        before = subprocess.run(["pbpaste"], capture_output=True).stdout
        kb = KeyboardController()
        with kb.pressed(Key.cmd):
            kb.press("c")
            kb.release("c")
        time.sleep(0.05)
        after = subprocess.run(["pbpaste"], capture_output=True).stdout
        if after == before:
            return ''
        selected = after.decode("utf-8", errors="replace").strip()
        subprocess.run(["pbcopy"], input=before)
        return selected

    def transcribe_and_paste(self, audio):
        set_widget('active', 'Transcribing...')
        result = self._pipe[0]({"array": audio, "sampling_rate": SAMPLE_RATE}, return_timestamps=True)
        transcript = result["text"].strip()
        print(f"Transcript: {transcript}")

        set_widget('active', 'Formatting...')
        try:
            formatted = self._generate_text(('transcribe', transcript))
        except Exception:
            logger.exception("Formatting failed; preserving the Whisper transcript")
            formatted = transcript
        print(f"Formatted:  {formatted}")

        state.transcription_history.append({
            'time': datetime.datetime.now(), 'mode': 'transcribe', 'text': formatted,
        })
        self.paste_at_cursor(formatted)
        set_widget('active', 'Done')

    def command_and_paste(self, audio, selected):
        set_widget('active', 'Transcribing...')
        result = self._pipe[0]({"array": audio, "sampling_rate": SAMPLE_RATE}, return_timestamps=True)
        command = result["text"].strip()
        print(f"Command:  {command}")
        print(f"Selected: {selected!r}")

        set_widget('active', 'Thinking...')
        output = self._generate_text(('command', command, selected))
        print(f"Output:   {output}")

        state.transcription_history.append({
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
        if not self._models_ready[0]:
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
            pressed, shift_held = self._trigger_events.get()
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

        keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        flags = CGEventGetFlags(event)
        self._shift_held = bool(flags & kCGEventFlagMaskShift)
        with self._trigger_lock:
            if self._trigger_is_down and self._shift_held:
                self._trigger_command_mode = True

        if state.listening_for_key:
            if event_type in (kCGEventKeyDown, kCGEventFlagsChanged):
                state.transcribe_keycode = keycode
                state.listening_for_key = False
                with self._trigger_lock:
                    self._set_trigger_pressed(False)
                    self._ignore_trigger_until_release = True
                    self._trigger_down_since = time.monotonic()
                print(f"New trigger key set: {keycode}")
                if callable(state.on_key_assigned):
                    state.on_key_assigned(keycode)
                return None
            return event

        if keycode != state.transcribe_keycode:
            # Fn-up may be coalesced into another modifier's flags event.
            # Its flag disappearing is release evidence even with another keycode.
            modifier_mask = self._modifier_mask_for_keycode(state.transcribe_keycode)
            if (event_type == kCGEventFlagsChanged and modifier_mask is not None
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
                self._set_trigger_pressed(bool(flags & modifier_mask), self._shift_held)
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
        g_model, g_proc = vlm_load(GEMMA_MODEL_ID)
        print("Gemma ready.")
        ready.set()
        while True:
            task, response = self._gemma_in.get()
            try:
                response.put(self._gemma_reply(g_model, g_proc, task))
            except Exception as exc:
                logger.exception("Gemma request failed")
                response.put(exc)
            finally:
                self._gemma_in.task_done()

    def load_models(self):
        set_widget('active', 'Loading models...')
        whisper_done = threading.Event()
        gemma_done   = threading.Event()

        def _load_whisper():
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
            whisper_done.set()

        threading.Thread(target=_load_whisper, daemon=True).start()
        threading.Thread(target=self._gemma_worker, args=(gemma_done,), daemon=True).start()

        while not (whisper_done.is_set() and gemma_done.is_set()):
            time.sleep(0.3)
            if whisper_done.is_set() and not gemma_done.is_set():
                set_widget('active', 'Loading Gemma...')
            elif gemma_done.is_set() and not whisper_done.is_set():
                set_widget('active', 'Loading Whisper...')

        self._models_ready[0] = True
        set_widget('idle')
        print("Models ready. Hold Fn to record, release to transcribe.\n")
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        app.setDelegate_(delegate)
        app.run()
