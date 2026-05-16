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

from pynput import keyboard
from pynput.keyboard import Key, Controller as KeyboardController
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from ApplicationServices import AXIsProcessTrustedWithOptions

import objc
from AppKit import (
    NSApplication, NSPanel, NSView, NSColor, NSBezierPath,
    NSFont, NSAttributedString, NSForegroundColorAttributeName, NSFontAttributeName,
    NSScreen, NSStatusWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorTransient,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSApplicationActivationPolicyAccessory,
)
from Foundation import NSObject, NSTimer, NSMakeRect, NSMakePoint, NSDictionary

SAMPLE_RATE  = 16000
WIN_W, WIN_H = 280, 100

NSBorderlessWindowMask   = 0
NSNonactivatingPanelMask = 1 << 7
NSBackingStoreBuffered   = 2

_state_queue = queue.Queue()


def request_accessibility():
    opts = NSDictionary.dictionaryWithObject_forKey_(True, "AXTrustedCheckOptionPrompt")
    if not AXIsProcessTrustedWithOptions(opts):
        print("Accessibility permission required. Grant it in System Settings and restart.")
        import sys; sys.exit(1)


# ── Floating pill widget ──────────────────────────────────────────────────────

class PillView(NSView):
    _state     = 'idle'
    _label     = ''
    _anim_tick = 0

    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(self.bounds())
        bw = self.bounds().size.width
        bh = self.bounds().size.height
        dark = NSColor.colorWithRed_green_blue_alpha_(0.11, 0.11, 0.11, 1.0)

        if self._state == 'idle':
            self._pill((bw - 60) / 2, 8, 60, 6,
                       NSColor.colorWithRed_green_blue_alpha_(0.557, 0.557, 0.576, 1.0))

        elif self._state == 'recording':
            pw, ph = 160, 38
            px, py = (bw - pw) / 2, (bh - ph) / 2
            self._pill(px, py, pw, ph, dark)
            n, bar_w, gap = 7, 3, 5
            total_w = n * bar_w + (n - 1) * gap
            x0 = px + (pw - total_w) / 2
            max_h, min_h = ph - 14, 4
            NSColor.whiteColor().set()
            for i in range(n):
                phase  = self._anim_tick * 0.18 + i * 0.75
                height = min_h + (max_h - min_h) * (0.5 + 0.5 * math.sin(phase))
                bx     = x0 + i * (bar_w + gap)
                by     = py + (ph - height) / 2
                path   = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(bx, by, bar_w, height), bar_w / 2, bar_w / 2
                )
                path.fill()

        else:
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
            self._anim_tick += 1
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
        panel.setIgnoresMouseEvents_(True)

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
    print("Loading model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype  = torch.float16 if device == "mps" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3-turbo", dtype=dtype,
        low_cpu_mem_usage=True, use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype, device=device,
    )
    print("Model ready. Press Left Control to start/stop recording.\n")

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

    def transcribe_and_paste(audio):
        set_widget('active', 'Transcribing...')
        result = pipe({"array": audio, "sampling_rate": SAMPLE_RATE})
        text = result["text"].strip()
        print(f"Transcript: {text}\n")
        paste_at_cursor(text)
        set_widget('active', 'Pasted!')
        time.sleep(2)
        set_widget('idle')

    recording = False
    frames    = []
    stream    = None
    lock      = threading.Lock()

    def toggle_recording():
        nonlocal recording, frames, stream
        with lock:
            if not recording:
                frames = []
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    callback=lambda indata, *_: frames.append(indata.copy()),
                )
                stream.start()
                recording = True
                set_widget('recording')
                print("Recording started.")
            else:
                stream.stop()
                stream.close()
                recording = False
                print("Recording stopped.")
                audio = np.concatenate(frames, axis=0).squeeze()
                if len(audio) < SAMPLE_RATE:
                    set_widget('active', 'Too short')
                    time.sleep(2)
                    set_widget('idle')
                    return
                threading.Thread(
                    target=transcribe_and_paste, args=(audio,), daemon=True
                ).start()

    def on_press(key):
        if key == Key.ctrl_l:
            threading.Thread(target=toggle_recording, daemon=True).start()

    keyboard.Listener(on_press=on_press).start()

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
