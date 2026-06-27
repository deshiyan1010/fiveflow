import os
import sys
import time
import queue
import threading
import subprocess
import datetime

import numpy as np
import sounddevice as sd
import torch
from mlx_vlm import load as vlm_load, generate as vlm_generate
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pynput.keyboard import Key, Controller as KeyboardController

import Cocoa
from Quartz import (
    CGEventTapCreate, CGEventTapEnable, kCGSessionEventTap, kCGHeadInsertEventTap,
    kCGEventTapOptionDefault, CGEventMaskBit, kCGEventFlagsChanged,
    CFRunLoopAddSource, kCFRunLoopCommonModes, CFRunLoopGetMain,
    CGEventGetIntegerValueField, CGEventGetFlags,
)

from . import state
from .config import (
    SAMPLE_RATE, WHISPER_MODEL_ID, GEMMA_MODEL_ID,
    kCGKeyboardEventKeycode, kCGEventFlagMaskSecondaryFn, kCGEventFlagMaskShift,
)
from .ui.widgets import set_widget


class Pipeline:
    def __init__(self):
        self._recording    = False
        self._frames       = []
        self._stream       = None
        self._lock         = threading.Lock()
        self._fn_pressed   = False
        self._shift_held   = False
        self._models_ready = [False]
        self._pipe         = [None]
        self._tap_ref      = [None]
        self._record_start = [0.0]
        self._gemma_in     = queue.Queue()
        self._gemma_out    = queue.Queue()

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
        self._gemma_in.put(('transcribe', transcript))
        formatted = self._gemma_out.get()
        print(f"Formatted:  {formatted}")

        state.transcription_history.append({
            'time': datetime.datetime.now(), 'mode': 'transcribe', 'text': formatted,
        })
        self.paste_at_cursor(formatted)
        set_widget('active', 'Done')
        time.sleep(1.5)
        set_widget('idle')

    def command_and_paste(self, audio, selected):
        set_widget('active', 'Transcribing...')
        result = self._pipe[0]({"array": audio, "sampling_rate": SAMPLE_RATE}, return_timestamps=True)
        command = result["text"].strip()
        print(f"Command:  {command}")
        print(f"Selected: {selected!r}")

        set_widget('active', 'Thinking...')
        self._gemma_in.put(('command', command, selected))
        output = self._gemma_out.get()
        print(f"Output:   {output}")

        state.transcription_history.append({
            'time': datetime.datetime.now(), 'mode': 'command', 'command': command, 'text': output,
        })
        self.paste_at_cursor(output)
        set_widget('active', 'Done')
        time.sleep(1.5)
        set_widget('idle')

    def _open_audio_stream(self):
        self._frames = []

        def audio_callback(indata, _frame_count, _time_info, _status):
            self._frames.append(indata.copy())
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
        self._record_start[0] = time.monotonic()
        set_widget('recording')

    def _close_audio_stream(self):
        self._stream.stop()
        self._stream.close()
        state.audio_level = 0.0
        self._recording = False

    def _process_audio(self, audio, mode):
        if len(audio) >= SAMPLE_RATE:
            if mode == 'command':
                threading.Thread(target=self.command_and_paste, args=(audio, self.get_selected_text()), daemon=True).start()
            else:
                threading.Thread(target=self.transcribe_and_paste, args=(audio,), daemon=True).start()
        else:
            set_widget('active', 'Too short')
            time.sleep(2)
            set_widget('idle')

    def _recording_controller(self):
        fn_was_down    = False
        shift_seen     = False
        debounce_until = 0.0

        while True:
            time.sleep(0.01)

            if not self._models_ready[0]:
                continue

            fn_down = self._fn_pressed
            sh_down = self._shift_held
            now     = time.monotonic()

            if fn_down and not fn_was_down:
                if now < debounce_until:
                    fn_was_down = fn_down
                    continue
                debounce_until = now + 0.08

                action = None
                with self._lock:
                    if not self._recording:
                        self._open_audio_stream()
                        shift_seen = sh_down
                        print("Recording started.")
                    else:
                        mode = 'command' if shift_seen else 'transcribe'
                        self._close_audio_stream()
                        print(f"Recording stopped (toggle). mode={mode}")
                        audio = np.concatenate(self._frames, axis=0).squeeze()
                        shift_seen = False
                        action = ('process', audio, mode)

                if action:
                    self._process_audio(action[1], action[2])

            elif fn_down and fn_was_down:
                if sh_down:
                    shift_seen = True

            elif not fn_down and fn_was_down:
                debounce_until = now + 0.08
                action = None
                with self._lock:
                    if self._recording:
                        mode = 'command' if shift_seen else 'transcribe'
                        self._close_audio_stream()
                        print(f"Recording stopped. mode={mode}")
                        audio = np.concatenate(self._frames, axis=0).squeeze()
                        shift_seen = False
                        action = ('process', audio, mode)

                if action:
                    self._process_audio(action[1], action[2])

            fn_was_down = fn_down

    def fn_event_callback(self, proxy, event_type, event, refcon):
        if event_type == 0xFFFFFFFE:
            if self._tap_ref[0]:
                CGEventTapEnable(self._tap_ref[0], True)
            return event

        keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        if keycode != 63:
            return event

        flags  = CGEventGetFlags(event)
        fn_now = bool(flags & kCGEventFlagMaskSecondaryFn)
        sh_now = bool(flags & kCGEventFlagMaskShift)

        self._fn_pressed = fn_now
        self._shift_held = sh_now
        return None

    def setup_event_tap(self):
        tap = CGEventTapCreate(
            kCGSessionEventTap,
            kCGHeadInsertEventTap,
            kCGEventTapOptionDefault,
            CGEventMaskBit(kCGEventFlagsChanged),
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

    def _watchdog(self):
        while True:
            time.sleep(5)
            if self._recording and (time.monotonic() - self._record_start[0]) > 30:
                print("Watchdog: recording stuck - forcing stop.")
                self._fn_pressed = False

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

        def _gemma_worker():
            print(f"Loading {GEMMA_MODEL_ID}...")
            g_model, g_proc = vlm_load(GEMMA_MODEL_ID)
            print("Gemma ready.")
            gemma_done.set()
            while True:
                task = self._gemma_in.get()
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
                self._gemma_out.put(reply.text.strip())

        threading.Thread(target=_load_whisper, daemon=True).start()
        threading.Thread(target=_gemma_worker, daemon=True).start()

        while not (whisper_done.is_set() and gemma_done.is_set()):
            time.sleep(0.3)
            if whisper_done.is_set() and not gemma_done.is_set():
                set_widget('active', 'Loading Gemma...')
            elif gemma_done.is_set() and not whisper_done.is_set():
                set_widget('active', 'Loading Whisper...')

        self._models_ready[0] = True
        self._fn_pressed = False
        set_widget('idle')
        print("Models ready. Hold Fn to record, release to transcribe.\n")
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(self):
        self.setup_event_tap()
        threading.Thread(target=self._watchdog, daemon=True).start()
        threading.Thread(target=self._recording_controller, daemon=True).start()
        set_widget('active', 'Loading models...')
        threading.Thread(target=self.load_models, daemon=True).start()

        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
        from .ui.app_delegate import AppDelegate

        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        delegate = AppDelegate.alloc().init()
        app.setDelegate_(delegate)
        app.run()
