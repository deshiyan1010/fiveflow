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
import datetime

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
    NSScrollView, NSTextField, NSButton,
)
from Foundation import NSObject, NSTimer, NSMakeRect, NSMakePoint, NSMakeSize, NSDictionary, NSMutableAttributedString

SAMPLE_RATE  = 16000
WIN_W, WIN_H = 280, 100

NSBorderlessWindowMask   = 0
NSNonactivatingPanelMask = 1 << 7
NSBackingStoreBuffered   = 2

_state_queue  = queue.Queue()
_audio_level  = 0.0  # RMS amplitude 0–1, written by audio callback, read by UI

_transcription_history   = []     # list of dicts: {time, mode, text}
_toggle_history_callback = [None]  # called when idle pill is clicked


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
        if self._state != 'idle':
            return None
        bw = self.bounds().size.width
        # Close button (only when hovering)
        if self._hovering:
            dx = point.x - self._CLOSE_CX
            dy = point.y - self._CLOSE_CY
            if dx * dx + dy * dy <= self._CLOSE_R ** 2:
                return self
        # Pill body — expanded hit area so it's easy to click
        px0 = (bw - 60) / 2 - 8
        px1 = (bw + 60) / 2 + 8
        if px0 <= point.x <= px1 and 0 <= point.y <= 24:
            return self
        return None

    def mouseUp_(self, event):
        loc = self.convertPoint_fromView_(event.locationInWindow(), None)
        # Close button
        dx = loc.x - self._CLOSE_CX
        dy = loc.y - self._CLOSE_CY
        if self._hovering and dx * dx + dy * dy <= self._CLOSE_R ** 2:
            NSApplication.sharedApplication().terminate_(None)
            return
        # Pill body → toggle history window
        if _toggle_history_callback[0]:
            _toggle_history_callback[0]()

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


class _CopyButton(NSButton):
    _copy_text = ''

    def performCopy_(self, sender):
        subprocess.run(["pbcopy"], input=self._copy_text.encode())
        self.setTitle_("Copied!")
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.5, self, b'resetCopyTitle:', None, False
        )

    def resetCopyTitle_(self, timer):
        self.setTitle_("Copy")


class _DividerView(NSView):
    def drawRect_(self, rect):
        NSColor.colorWithRed_green_blue_alpha_(0.85, 0.85, 0.87, 1.0).set()
        NSBezierPath.fillRect_(self.bounds())


class _EntryCardView(NSView):
    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(self.bounds())
        bw = self.bounds().size.width
        bh = self.bounds().size.height
        NSColor.colorWithRed_green_blue_alpha_(0.96, 0.96, 0.97, 1.0).set()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(0, 0, bw, bh), 10, 10
        ).fill()


class _HistoryContainerView(NSView):
    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(self.bounds())
        bw = self.bounds().size.width
        bh = self.bounds().size.height
        NSColor.whiteColor().set()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(0, 0, bw, bh), 16, 16
        ).fill()


class HistoryWindowController(NSObject):
    _panel        = None
    _scroll_view  = None
    _content_view = None
    _visible      = False
    _content_W    = 0
    _W = 440
    _H = 480

    @objc.python_method
    def setup(self):
        W, H = self._W, self._H
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H),
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
        panel.setHasShadow_(True)

        container = _HistoryContainerView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        panel.setContentView_(container)

        PAD = 16
        # Title
        title = NSTextField.alloc().initWithFrame_(NSMakeRect(PAD, H - PAD - 20, W - 2*PAD, 20))
        title.setEditable_(False)
        title.setSelectable_(False)
        title.setBordered_(False)
        title.setBackgroundColor_(NSColor.clearColor())
        title.setStringValue_("Transcription History")
        title.setFont_(NSFont.boldSystemFontOfSize_(14))
        title.setTextColor_(NSColor.colorWithRed_green_blue_alpha_(0.11, 0.11, 0.13, 1.0))
        container.addSubview_(title)

        # Divider below title
        div_y = H - PAD - 20 - 9
        container.addSubview_(
            _DividerView.alloc().initWithFrame_(NSMakeRect(PAD, div_y, W - 2*PAD, 1))
        )

        scroll_y = PAD
        scroll_h = div_y - 8 - scroll_y
        scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(PAD, scroll_y, W - 2*PAD, scroll_h)
        )
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setDrawsBackground_(False)
        scroll.setBorderType_(0)

        content_w = W - 2*PAD - 16
        content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, content_w, scroll_h))
        scroll.setDocumentView_(content)
        container.addSubview_(scroll)

        self._panel        = panel
        self._scroll_view  = scroll
        self._content_view = content
        self._content_W    = content_w

    @objc.python_method
    def toggle_near_frame(self, pill_frame):
        if self._visible:
            self._panel.orderOut_(None)
            self._visible = False
            return
        W, H = self._W, self._H
        px = pill_frame.origin.x + (pill_frame.size.width - W) / 2
        py = pill_frame.origin.y + pill_frame.size.height + 8
        sr = NSScreen.mainScreen().frame()
        py = min(py, sr.origin.y + sr.size.height - H - 10)
        px = max(sr.origin.x + 10, min(px, sr.origin.x + sr.size.width - W - 10))
        self._panel.setFrameOrigin_(NSMakePoint(px, py))
        self._update_content()
        self._panel.orderFrontRegardless()
        self._visible = True

    @objc.python_method
    def _text_height(self, text, font, width):
        attrs = NSDictionary.dictionaryWithObjects_forKeys_([font], [NSFontAttributeName])
        astr  = NSAttributedString.alloc().initWithString_attributes_(text or ' ', attrs)
        br    = astr.boundingRectWithSize_options_(NSMakeSize(width, 100000), 1)
        return math.ceil(br.size.height) + 6

    @objc.python_method
    def _make_label(self, text, font, color, frame):
        tf = NSTextField.alloc().initWithFrame_(frame)
        tf.setEditable_(False)
        tf.setSelectable_(True)
        tf.setBordered_(False)
        tf.setBackgroundColor_(NSColor.clearColor())
        tf.setStringValue_(text)
        tf.setFont_(font)
        tf.setTextColor_(color)
        tf.cell().setWraps_(True)
        tf.cell().setUsesSingleLineMode_(False)
        tf.cell().setLineBreakMode_(0)
        return tf

    @objc.python_method
    def _make_copy_btn(self, copy_text, frame):
        btn = _CopyButton.alloc().initWithFrame_(frame)
        btn._copy_text = copy_text
        btn.setTitle_("Copy")
        btn.setBezelStyle_(12)
        btn.setButtonType_(0)
        btn.setFont_(NSFont.systemFontOfSize_(11))
        btn.setTarget_(btn)
        btn.setAction_(b'performCopy:')
        return btn

    @objc.python_method
    def _update_content(self):
        for sv in list(self._content_view.subviews()):
            sv.removeFromSuperview()

        history = list(_transcription_history)
        W = self._content_W

        CARD_X    = 4     # outer horizontal margin inside content view
        IV        = 12    # card vertical inner padding (top & bottom)
        IH        = 14    # card horizontal inner padding
        GAP       = 8     # gap between cards
        BTN_W     = 56
        BTN_H     = 22
        HEAD_H    = 16    # time + mode header row
        ROW_H     = BTN_H # label + copy-button row height
        LABEL_H   = 13
        DIV_TOTAL = 17    # space reserved for divider line (8 + 1 + 8)

        card_w = W - 2 * CARD_X
        text_w = card_w - 2 * IH

        body_font  = NSFont.systemFontOfSize_(13)
        meta_font  = NSFont.systemFontOfSize_(11)
        label_font = NSFont.boldSystemFontOfSize_(10)

        near_black = NSColor.colorWithRed_green_blue_alpha_(0.11, 0.11, 0.13, 1.0)
        mid_gray   = NSColor.colorWithRed_green_blue_alpha_(0.53, 0.53, 0.56, 1.0)
        blue       = NSColor.colorWithRed_green_blue_alpha_(0.20, 0.47, 0.96, 1.0)
        orange     = NSColor.colorWithRed_green_blue_alpha_(0.85, 0.40, 0.10, 1.0)

        if not history:
            self._content_view.addSubview_(self._make_label(
                "No transcriptions yet.", NSFont.systemFontOfSize_(13), mid_gray,
                NSMakeRect(CARD_X + IH, 20, card_w - 2*IH, 24)
            ))
            return

        # ── First pass: compute card heights ────────────────────────────────
        items = []
        for entry in reversed(history):
            if entry['mode'] == 'transcribe':
                th = self._text_height(entry['text'], body_font, text_w)
                card_h = IV + HEAD_H + 8 + th + 6 + BTN_H + IV
                items.append((entry, card_h, {'th': th}))
            else:
                cmd = entry.get('command', '')
                ch  = self._text_height(cmd, body_font, text_w)
                oh  = self._text_height(entry['text'], body_font, text_w)
                card_h = IV + HEAD_H + 8 + ROW_H + 4 + ch + DIV_TOTAL + ROW_H + 4 + oh + IV
                items.append((entry, card_h, {'ch': ch, 'oh': oh, 'cmd': cmd}))

        total_h = GAP + sum(card_h + GAP for _, card_h, _ in items)
        total_h = max(total_h, self._scroll_view.frame().size.height)

        self._content_view.setFrame_(NSMakeRect(0, 0, W, total_h))

        # ── Second pass: lay out cards top-to-bottom ─────────────────────────
        y_cursor = total_h - GAP

        for entry, card_h, dims in items:
            card_y = y_cursor - card_h
            card = _EntryCardView.alloc().initWithFrame_(
                NSMakeRect(CARD_X, card_y, card_w, card_h)
            )
            self._content_view.addSubview_(card)

            y = card_h - IV  # tracks position within card, decreasing toward bottom

            # Header: time (gray) + mode (colored)
            time_str   = entry['time'].strftime("%-I:%M %p")
            mode_str   = "Transcribe" if entry['mode'] == 'transcribe' else "Command"
            mode_color = blue if entry['mode'] == 'transcribe' else orange
            card.addSubview_(self._make_label(
                time_str, meta_font, mid_gray,
                NSMakeRect(IH, y - HEAD_H, 66, HEAD_H)
            ))
            card.addSubview_(self._make_label(
                "·  " + mode_str, meta_font, mode_color,
                NSMakeRect(IH + 70, y - HEAD_H, 110, HEAD_H)
            ))
            y -= HEAD_H + 8

            if entry['mode'] == 'transcribe':
                th = dims['th']
                card.addSubview_(self._make_label(
                    entry['text'], body_font, near_black,
                    NSMakeRect(IH, y - th, text_w, th)
                ))
                y -= th + 6
                card.addSubview_(self._make_copy_btn(
                    entry['text'],
                    NSMakeRect(card_w - IH - BTN_W, y - BTN_H, BTN_W, BTN_H)
                ))

            else:
                ch  = dims['ch']
                oh  = dims['oh']
                cmd = dims['cmd']
                lv  = (ROW_H - LABEL_H) / 2  # vertical offset to center label in row

                # Voice command section label + copy button
                card.addSubview_(self._make_label(
                    "VOICE COMMAND", label_font, mid_gray,
                    NSMakeRect(IH, y - ROW_H + lv, card_w - 2*IH - BTN_W - 6, LABEL_H)
                ))
                card.addSubview_(self._make_copy_btn(
                    cmd, NSMakeRect(card_w - IH - BTN_W, y - ROW_H, BTN_W, BTN_H)
                ))
                y -= ROW_H + 4
                card.addSubview_(self._make_label(
                    cmd, body_font, near_black,
                    NSMakeRect(IH, y - ch, text_w, ch)
                ))
                y -= ch

                # Divider
                card.addSubview_(
                    _DividerView.alloc().initWithFrame_(
                        NSMakeRect(IH, y - DIV_TOTAL // 2, card_w - 2*IH, 1)
                    )
                )
                y -= DIV_TOTAL

                # Output section label + copy button
                card.addSubview_(self._make_label(
                    "OUTPUT", label_font, mid_gray,
                    NSMakeRect(IH, y - ROW_H + lv, card_w - 2*IH - BTN_W - 6, LABEL_H)
                ))
                card.addSubview_(self._make_copy_btn(
                    entry['text'], NSMakeRect(card_w - IH - BTN_W, y - ROW_H, BTN_W, BTN_H)
                ))
                y -= ROW_H + 4
                card.addSubview_(self._make_label(
                    entry['text'], body_font, near_black,
                    NSMakeRect(IH, y - oh, text_w, oh)
                ))

            y_cursor = card_y - GAP

        # Scroll to top (newest entry)
        self._content_view.scrollRectToVisible_(NSMakeRect(0, total_h - 1, W, 1))


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

        history_ctrl = HistoryWindowController.alloc().init()
        history_ctrl.setup()
        self._history_ctrl = history_ctrl

        pill_panel = self._panel
        def _toggle_history():
            history_ctrl.toggle_near_frame(pill_panel.frame())
        _toggle_history_callback[0] = _toggle_history


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

        set_widget('active', 'Formating...')
        _gemma_in.put(('transcribe', transcript))
        formated = _gemma_out.get()
        print(f"Formated:  {formated}")

        _transcription_history.append({'time': datetime.datetime.now(), 'mode': 'transcribe', 'text': formated})
        paste_at_cursor(formated)
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

        _transcription_history.append({'time': datetime.datetime.now(), 'mode': 'command', 'command': command, 'text': output})
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
                        "Return the formated text and nothing else.\n\n" + task[1]
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
