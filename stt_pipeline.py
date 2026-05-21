import os
os.environ["HF_TOKEN"] = ""

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

import queue
import threading
import subprocess
import time
import math

import numpy as np
import sounddevice as sd
import torch
from mlx_vlm import load as vlm_load, generate as vlm_generate

import Cocoa
from Quartz import (
    CGEventTapCreate, kCGSessionEventTap, kCGHeadInsertEventTap,
    kCGEventTapOptionDefault, CGEventMaskBit, kCGEventFlagsChanged,
    CFRunLoopAddSource, kCFRunLoopCommonModes, CFRunLoopGetMain,
    CGEventGetIntegerValueField, CGEventGetFlags,
)
from pynput.keyboard import Key, Controller as KeyboardController
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from ApplicationServices import AXIsProcessTrustedWithOptions

kCGKeyboardEventKeycode     = 9
kCGEventFlagMaskSecondaryFn = 0x00800000
kCGEventFlagMaskShift       = 0x00020000

GEMMA_MODEL_ID = "mlx-community/gemma-4-e2b-it-4bit"

import objc
from AppKit import (
    NSApplication, NSPanel, NSView, NSColor, NSBezierPath,
    NSFont, NSAttributedString, NSForegroundColorAttributeName, NSFontAttributeName,
    NSScreen, NSStatusWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorTransient,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSApplicationActivationPolicyAccessory,
    NSTrackingArea, NSTrackingMouseEnteredAndExited, NSTrackingActiveAlways,
)
from Foundation import NSObject, NSTimer, NSMakeRect, NSMakePoint, NSDictionary

SAMPLE_RATE  = 16000
WIN_W, WIN_H = 280, 100

NSBorderlessWindowMask   = 0
NSNonactivatingPanelMask = 1 << 7
NSBackingStoreBuffered   = 2

_state_queue  = queue.Queue()
_audio_level  = 0.0  # RMS amplitude 0–1, written by audio callback, read by UI


def request_accessibility():
    opts = NSDictionary.dictionaryWithObject_forKey_(True, "AXTrustedCheckOptionPrompt")
    if not AXIsProcessTrustedWithOptions(opts):
        print("Accessibility permission required. Grant it in System Settings and restart.")
        import sys; sys.exit(1)


# ── Floating pill widget ──────────────────────────────────────────────────────

class PillView(NSView):
    _state      = 'idle'
    _label      = ''
    _anim_tick  = 0
    _pill_scale = 0.0   # 0 = small idle pill, 1 = full recording pill
    _hovering   = False

    # ── close-button geometry (view coords, y=0 at bottom) ───────────────────
    # Idle pill right edge: (WIN_W+60)/2 = 170. Button center: 170 + 15 = 185.
    _CLOSE_CX = (WIN_W + 60) / 2 + 15
    _CLOSE_CY = 11.0
    _CLOSE_R  = 7.0

    def updateTrackingAreas(self):
        for area in list(self.trackingAreas()):
            self.removeTrackingArea_(area)
        bw = self.bounds().size.width
        track_rect = NSMakeRect((bw - 60) / 2 - 5, 0, 110, 26)
        area = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
            track_rect,
            NSTrackingMouseEnteredAndExited | NSTrackingActiveAlways,
            self,
            None,
        )
        self.addTrackingArea_(area)

    def mouseEntered_(self, event):
        if self._state == 'idle':
            self._hovering = True
            self.setNeedsDisplay_(True)

    def mouseExited_(self, event):
        self._hovering = False
        self.setNeedsDisplay_(True)

    def hitTest_(self, point):
        # Pass all clicks through except when cursor is on the close button.
        if self._hovering and self._state == 'idle':
            dx = point.x - self._CLOSE_CX
            dy = point.y - self._CLOSE_CY
            if dx * dx + dy * dy <= self._CLOSE_R ** 2:
                return self
        return None

    def mouseUp_(self, event):
        loc = self.convertPoint_fromView_(event.locationInWindow(), None)
        dx = loc.x - self._CLOSE_CX
        dy = loc.y - self._CLOSE_CY
        if dx * dx + dy * dy <= self._CLOSE_R ** 2:
            NSApplication.sharedApplication().terminate_(None)

    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(self.bounds())
        bw = self.bounds().size.width
        bh = self.bounds().size.height
        dark = NSColor.colorWithRed_green_blue_alpha_(0.11, 0.11, 0.11, 1.0)
        gray = NSColor.colorWithRed_green_blue_alpha_(0.557, 0.557, 0.576, 1.0)
        s    = self._pill_scale

        if self._state == 'recording' or s > 0:
            # Interpolate from small idle pill → full recording pill
            pw = 60  + (160 - 60)  * s
            ph = 6   + (38  - 6)   * s
            px = (bw - pw) / 2
            py = 8   + ((bh - 38) / 2 - 8) * s

            self._pill(px, py, pw, ph, dark if s > 0.4 else gray)

            # Bars fade in after pill is mostly expanded, scale height by live audio
            if s > 0.4:
                bar_alpha = min(1.0, (s - 0.4) / 0.3)
                level     = _audio_level
                n, bar_w, gap = 7, 3, 5
                total_w = n * bar_w + (n - 1) * gap
                x0      = px + (pw - total_w) / 2
                max_h   = (38 - 14) * s
                min_h   = 4
                NSColor.colorWithRed_green_blue_alpha_(1, 1, 1, bar_alpha).set()
                for i in range(n):
                    phase  = self._anim_tick * 0.18 + i * 0.75
                    amp    = 0.15 + 0.85 * level          # quiet → small, loud → tall
                    height = max(min_h, min_h + (max_h - min_h) * (0.5 + 0.5 * math.sin(phase)) * amp)
                    bx     = x0 + i * (bar_w + gap)
                    by     = py + (ph - height) / 2
                    path   = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                        NSMakeRect(bx, by, bar_w, height), bar_w / 2, bar_w / 2
                    )
                    path.fill()

        elif self._state == 'idle':
            self._pill((bw - 60) / 2, 8, 60, 6, gray)
            if self._hovering:
                cx, cy, r = self._CLOSE_CX, self._CLOSE_CY, self._CLOSE_R
                gray.set()
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(cx - r, cy - r, r * 2, r * 2)
                ).fill()
                self._text('×', cx, cy, 11)

        else:
            # active state: text label
            pw, ph = 200, 38
            px, py = (bw - pw) / 2, (bh - ph) / 2
            self._pill(px, py, pw, ph, dark)
            self._text(self._label, px + pw / 2, py + ph / 2, 13, bold=True)

    @objc.python_method
    def _pill(self, x, y, w, h, color):
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(x, y, w, h), h / 2, h / 2
        )
        color.set()
        path.fill()

    @objc.python_method
    def _text(self, text, x, y, size=13, bold=False):
        font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
        attrs = NSDictionary.dictionaryWithObjects_forKeys_(
            [NSColor.whiteColor(), font],
            [NSForegroundColorAttributeName, NSFontAttributeName],
        )
        astr = NSAttributedString.alloc().initWithString_attributes_(text, attrs)
        sz = astr.size()
        astr.drawAtPoint_(NSMakePoint(x - sz.width / 2, y - sz.height / 2))

    def checkQueue_(self, timer):
        changed = False
        try:
            while True:
                self._state, self._label = _state_queue.get_nowait()
                changed = True
        except queue.Empty:
            pass

        if self._state == 'recording':
            self._pill_scale = min(1.0, self._pill_scale + 0.12)  # ~400 ms expand
            self._anim_tick += 1
            changed = True
        elif self._pill_scale > 0:
            self._pill_scale = max(0.0, self._pill_scale - 0.12)  # ~400 ms collapse
            changed = True

        if changed:
            self.setNeedsDisplay_(True)


class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        sf = NSScreen.mainScreen().frame()
        x = (sf.size.width - WIN_W) / 2 + sf.origin.x
        y = sf.origin.y + 20

        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, WIN_W, WIN_H),
            NSBorderlessWindowMask | NSNonactivatingPanelMask,
            NSBackingStoreBuffered,
            False,
        )
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setOpaque_(False)
        panel.setLevel_(NSStatusWindowLevel)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorTransient
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setHasShadow_(False)

        view = PillView.alloc().initWithFrame_(NSMakeRect(0, 0, WIN_W, WIN_H))
        panel.setContentView_(view)
        panel.orderFrontRegardless()
        self._panel = panel


        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.05, view, b'checkQueue:', None, True
        )


def set_widget(state, label=''):
    _state_queue.put((state, label))


# ── STT logic ─────────────────────────────────────────────────────────────────

def main():
    request_accessibility()

    recording    = False
    frames       = []
    stream       = None
    lock         = threading.Lock()
    fn_held      = [False]
    shift_seen   = [False]
    models_ready = [False]
    _pipe        = [None]          # set by _load_models
    _gemma_in    = queue.Queue()
    _gemma_out   = queue.Queue()

    def paste_at_cursor(text):
        original = subprocess.run(["pbpaste"], capture_output=True).stdout
        subprocess.run(["pbcopy"], input=text.encode())
        time.sleep(0.05)
        kb = KeyboardController()
        with kb.pressed(Key.cmd):
            kb.press("v")
            kb.release("v")
        time.sleep(0.1)
        subprocess.run(["pbcopy"], input=original)

    def get_selected_text():
        # Non-destructive: Cmd+C only. Detect selection by comparing clipboard
        # before and after — if unchanged, nothing was selected.
        before = subprocess.run(["pbpaste"], capture_output=True).stdout
        kb = KeyboardController()
        with kb.pressed(Key.cmd):
            kb.press("c")
            kb.release("c")
        time.sleep(0.05)
        after = subprocess.run(["pbpaste"], capture_output=True).stdout
        if after == before:
            return ''  # nothing selected
        selected = after.decode("utf-8", errors="replace").strip()
        subprocess.run(["pbcopy"], input=before)  # restore clipboard
        return selected

    def transcribe_and_paste(audio):
        set_widget('active', 'Transcribing...')
        result = _pipe[0]({"array": audio, "sampling_rate": SAMPLE_RATE}, return_timestamps=True)
        transcript = result["text"].strip()
        print(f"Transcript: {transcript}")

        set_widget('active', 'Correcting...')
        _gemma_in.put(('transcribe', transcript))
        corrected = _gemma_out.get()
        print(f"Corrected:  {corrected}")

        paste_at_cursor(corrected)
        set_widget('active', 'Done')
        time.sleep(1.5)
        set_widget('idle')

    def command_and_paste(audio, selected):
        set_widget('active', 'Transcribing...')
        result = _pipe[0]({"array": audio, "sampling_rate": SAMPLE_RATE}, return_timestamps=True)
        command = result["text"].strip()
        print(f"Command:  {command}")
        print(f"Selected: {selected!r}")

        set_widget('active', 'Thinking...')
        _gemma_in.put(('command', command, selected))
        output = _gemma_out.get()
        print(f"Output:   {output}")

        paste_at_cursor(output)
        set_widget('active', 'Done')
        time.sleep(1.5)
        set_widget('idle')

    def start_recording():
        nonlocal recording, frames, stream
        global _audio_level
        with lock:
            if recording:
                return
            frames = []

            def audio_callback(indata, _frame_count, _time_info, _status):
                global _audio_level
                frames.append(indata.copy())
                rms = float(np.sqrt(np.mean(indata ** 2)))
                _audio_level = min(1.0, rms * 15)

            stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=audio_callback,
            )
            stream.start()
            recording = True
            set_widget('recording')
            print("Recording started.")

    def stop_recording(mode='transcribe'):
        nonlocal recording, frames, stream
        global _audio_level
        # Grab selection here (background thread) — pynput works fine outside the CGEventTap callback
        selected = get_selected_text() if mode == 'command' else ''
        with lock:
            if not recording:
                return
            stream.stop()
            stream.close()
            _audio_level = 0.0
            recording = False
            print(f"Recording stopped. mode={mode} selected={selected!r}")
            audio = np.concatenate(frames, axis=0).squeeze()
            if len(audio) < SAMPLE_RATE:
                set_widget('active', 'Too short')
                time.sleep(2)
                set_widget('idle')
                return
            if mode == 'command':
                threading.Thread(target=command_and_paste, args=(audio, selected), daemon=True).start()
            else:
                threading.Thread(target=transcribe_and_paste, args=(audio,), daemon=True).start()

    def fn_event_callback(proxy, event_type, event, refcon):
        if not models_ready[0]:
            return event
        keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        flags   = CGEventGetFlags(event)
        fn_now  = bool(flags & kCGEventFlagMaskSecondaryFn)
        sh_now  = bool(flags & kCGEventFlagMaskShift)

        if keycode == 63 and fn_now and not fn_held[0]:
            fn_held[0]    = True
            shift_seen[0] = sh_now
            threading.Thread(target=start_recording, daemon=True).start()
            return None

        if fn_held[0] and sh_now:
            shift_seen[0] = True

        if fn_held[0] and not fn_now:
            mode = 'command' if shift_seen[0] else 'transcribe'
            fn_held[0]    = False
            shift_seen[0] = False
            threading.Thread(target=stop_recording, args=(mode,), daemon=True).start()
            return None

        return event

    tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionDefault,
        CGEventMaskBit(kCGEventFlagsChanged),
        fn_event_callback,
        None,
    )
    if not tap:
        print("CGEventTap failed — ensure Accessibility permission is granted.")
        import sys; sys.exit(1)
    CFRunLoopAddSource(
        CFRunLoopGetMain(),
        Cocoa.CFMachPortCreateRunLoopSource(None, tap, 0),
        kCFRunLoopCommonModes,
    )

    def _load_models():
        set_widget('active', 'Loading models...')
        whisper_done = threading.Event()
        gemma_done   = threading.Event()

        def _load_whisper():
            print("Loading Whisper...")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype  = torch.float16 if device == "mps" else torch.float32
            w_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-large-v3-turbo", dtype=dtype,
                low_cpu_mem_usage=True, use_safetensors=True,
            )
            w_model.to(device)
            w_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
            _pipe[0] = pipeline(
                "automatic-speech-recognition",
                model=w_model, tokenizer=w_processor.tokenizer,
                feature_extractor=w_processor.feature_extractor,
                torch_dtype=dtype, device=device,
            )
            print("Whisper ready.")
            whisper_done.set()

        # Gemma must load and infer on the same thread (MLX streams are thread-local)
        def _gemma_worker():
            print(f"Loading {GEMMA_MODEL_ID}...")
            g_model, g_proc = vlm_load(GEMMA_MODEL_ID)
            print("Gemma ready.")
            gemma_done.set()
            while True:
                task = _gemma_in.get()
                if task[0] == 'transcribe':
                    content = (
                        "Fix only punctuation and obvious transcription errors. "
                        "Return the corrected text and nothing else.\n\n" + task[1]
                    )
                else:  # command
                    command, selected = task[1], task[2]
                    if selected:
                        content = (
                            f"Selected text:\n{selected}\n\n"
                            f"Command: {command}\n\n"
                            "Apply the command to the selected text. "
                            "Return only the result, no explanation."
                        )
                    else:
                        content = (
                            f"Command: {command}\n\n"
                            "Execute this command and return only the result, no explanation."
                        )
                messages = [{"role": "user", "content": content}]
                prompt = g_proc.apply_chat_template(messages, add_generation_prompt=True)
                reply = vlm_generate(g_model, g_proc, prompt=prompt,
                                     max_tokens=512, temperature=0.3, verbose=False)
                _gemma_out.put(reply.text.strip())

        threading.Thread(target=_load_whisper, daemon=True).start()
        threading.Thread(target=_gemma_worker, daemon=True).start()

        # Update pill as each model finishes
        while not (whisper_done.is_set() and gemma_done.is_set()):
            time.sleep(0.3)
            if whisper_done.is_set() and not gemma_done.is_set():
                set_widget('active', 'Loading Gemma...')
            elif gemma_done.is_set() and not whisper_done.is_set():
                set_widget('active', 'Loading Whisper...')

        models_ready[0] = True
        set_widget('idle')
        print("Models ready. Hold Fn to record, release to transcribe.\n")

    # Show pill immediately, load models in background
    set_widget('active', 'Loading models...')
    threading.Thread(target=_load_models, daemon=True).start()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
