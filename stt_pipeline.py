import os

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
    CGEventTapCreate, CGEventTapEnable, kCGSessionEventTap, kCGHeadInsertEventTap,
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
_history_open            = [False] # True while the history panel is visible


def request_accessibility():
    opts = NSDictionary.dictionaryWithObject_forKey_(True, "AXTrustedCheckOptionPrompt")
    if not AXIsProcessTrustedWithOptions(opts):
        print("Accessibility permission required. Grant it in System Settings and restart.")
        import sys; sys.exit(1)


# ── Floating pill widget ──────────────────────────────────────────────────────

class PillView(NSView):
    _state       = 'idle'
    _label       = ''
    _anim_tick   = 0
    _pill_scale  = 0.0   # 0 = small idle pill, 1 = full recording pill
    _hovering    = False
    _hover_scale = 0.0   # 0 = base idle size, 1 = fully expanded hover state

    _CLOSE_R = 9.0

    def updateTrackingAreas(self):
        for area in list(self.trackingAreas()):
            self.removeTrackingArea_(area)
        bw  = self.bounds().size.width
        hs  = self._hover_scale
        pw  = 45  + (160 - 45) * hs
        ph  = 8   + (28  - 8)  * hs
        px  = (bw - pw) / 2
        PAD = 3
        track_rect = NSMakeRect(px - PAD, 8 - PAD, pw + PAD * 2, ph + PAD * 2)
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
        hs = self._hover_scale
        pw = 45  + (160 - 45) * hs
        ph = 8   + (28  - 8)  * hs
        py = 8
        px = (bw - pw) / 2
        # Close button (only when hovering)
        if self._hovering:
            cx = px + pw + 15
            cy = py + ph / 2
            dx = point.x - cx
            dy = point.y - cy
            if dx * dx + dy * dy <= self._CLOSE_R ** 2:
                return self
        # Pill body with small padding
        if px - 6 <= point.x <= px + pw + 6 and py - 4 <= point.y <= py + ph + 4:
            return self
        return None

    def mouseDown_(self, event):
        if self._state != 'idle':
            return
        loc = self.convertPoint_fromView_(event.locationInWindow(), None)
        bw  = self.bounds().size.width
        hs  = self._hover_scale
        pw  = 45  + (160 - 45) * hs
        ph  = 8   + (28  - 8)  * hs
        cx  = (bw - pw) / 2 + pw + 15
        cy  = 8 + ph / 2
        # Close button — let mouseUp_ handle it, don't start a drag
        if self._hovering:
            dx = loc.x - cx
            dy = loc.y - cy
            if dx * dx + dy * dy <= self._CLOSE_R ** 2:
                return
        # Block here while the window is being dragged
        before = self.window().frame().origin
        self.window().performWindowDragWithEvent_(event)
        after  = self.window().frame().origin
        moved  = (after.x - before.x) ** 2 + (after.y - before.y) ** 2
        if moved < 16:
            # Tiny movement = click → toggle history
            if _toggle_history_callback[0]:
                _toggle_history_callback[0]()
        elif _history_open[0] and _toggle_history_callback[0]:
            # Dragged to new spot → close history (now misaligned)
            _toggle_history_callback[0]()

    def mouseUp_(self, event):
        if self._state != 'idle':
            return
        loc = self.convertPoint_fromView_(event.locationInWindow(), None)
        bw  = self.bounds().size.width
        hs  = self._hover_scale
        pw  = 45  + (160 - 45) * hs
        ph  = 8   + (28  - 8)  * hs
        cx  = (bw - pw) / 2 + pw + 15
        cy  = 8 + ph / 2
        dx  = loc.x - cx
        dy  = loc.y - cy
        if self._hovering and dx * dx + dy * dy <= self._CLOSE_R ** 2:
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
            hs   = self._hover_scale
            pw   = 45  + (160 - 45) * hs
            ph   = 8   + (28  - 8)  * hs
            py   = 8
            px   = (bw - pw) / 2
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(px, py, pw, ph), ph / 2, ph / 2
            )
            NSColor.colorWithRed_green_blue_alpha_(0.08, 0.08, 0.08, 1.0).set()
            path.fill()
            path.setLineWidth_(1.0)
            NSColor.colorWithRed_green_blue_alpha_(0.50, 0.50, 0.52, 1.0).set()
            path.stroke()
            if hs > 0.25:
                text_alpha = min(1.0, (hs - 0.25) / 0.35)
                label = "Close history" if _history_open[0] else "History"
                self._text(label, px + pw / 2, py + ph / 2, 12, alpha=text_alpha)
            if self._hovering:
                cx = px + pw + 15
                cy = py + ph / 2
                r  = self._CLOSE_R
                circle = NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(cx - r, cy - r, r * 2, r * 2)
                )
                NSColor.colorWithRed_green_blue_alpha_(0.08, 0.08, 0.08, 1.0).set()
                circle.fill()
                circle.setLineWidth_(1.0)
                NSColor.colorWithRed_green_blue_alpha_(0.50, 0.50, 0.52, 1.0).set()
                circle.stroke()
                self._text('×', cx, cy, 13)

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
    def _text(self, text, x, y, size=13, bold=False, alpha=1.0):
        font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
        color = NSColor.colorWithRed_green_blue_alpha_(1, 1, 1, alpha)
        attrs = NSDictionary.dictionaryWithObjects_forKeys_(
            [color, font],
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

        if self._state == 'idle':
            if self._hovering and self._hover_scale < 1.0:
                self._hover_scale = min(1.0, self._hover_scale + 0.2)
                self.updateTrackingAreas()
                changed = True
            elif not self._hovering and self._hover_scale > 0.0:
                self._hover_scale = max(0.0, self._hover_scale - 0.2)
                self.updateTrackingAreas()
                changed = True
        else:
            if self._hover_scale > 0.0:
                self._hover_scale = 0.0
                self.updateTrackingAreas()
                changed = True

        if changed:
            self.setNeedsDisplay_(True)


class _CopyButton(NSButton):
    _copy_text = ''

    def performCopy_(self, sender):
        subprocess.run(["pbcopy"], input=self._copy_text.encode())
        self.setTitle_("✓")
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.5, self, b'resetCopyTitle:', None, False
        )

    def resetCopyTitle_(self, timer):
        self.setTitle_("⧉")


class _DividerView(NSView):
    def drawRect_(self, rect):
        NSColor.colorWithRed_green_blue_alpha_(0.85, 0.85, 0.87, 1.0).set()
        NSBezierPath.fillRect_(self.bounds())


class _BadgeView(NSView):
    _text  = ''
    _color = None

    def drawRect_(self, rect):
        bw = self.bounds().size.width
        bh = self.bounds().size.height
        if self._color:
            self._color.set()
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(0, 0, bw, bh), bh / 2, bh / 2
        )
        path.fill()
        font  = NSFont.boldSystemFontOfSize_(9)
        attrs = NSDictionary.dictionaryWithObjects_forKeys_(
            [NSColor.whiteColor(), font],
            [NSForegroundColorAttributeName, NSFontAttributeName],
        )
        astr = NSAttributedString.alloc().initWithString_attributes_(self._text, attrs)
        sz   = astr.size()
        astr.drawAtPoint_(NSMakePoint(bw / 2 - sz.width / 2, bh / 2 - sz.height / 2))


class _EntryCardView(NSView):
    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(self.bounds())
        bw   = self.bounds().size.width
        bh   = self.bounds().size.height
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(0.5, 0.5, bw - 1, bh - 1), 10, 10
        )
        NSColor.whiteColor().set()
        path.fill()
        path.setLineWidth_(0.5)
        NSColor.colorWithRed_green_blue_alpha_(0.82, 0.82, 0.85, 1.0).set()
        path.stroke()


class _HistoryContainerView(NSView):
    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(self.bounds())
        bw   = self.bounds().size.width
        bh   = self.bounds().size.height
        # White rounded container
        clip = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(0, 0, bw, bh), 16, 16
        )
        clip.addClip()
        NSColor.colorWithRed_green_blue_alpha_(0.985, 0.985, 0.99, 1.0).set()
        NSBezierPath.fillRect_(NSMakeRect(0, 0, bw, bh))
        # Subtle header tint band at the top
        NSColor.colorWithRed_green_blue_alpha_(0.96, 0.96, 0.975, 1.0).set()
        NSBezierPath.fillRect_(NSMakeRect(0, bh - 52, bw, 52))


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

        PAD    = 16
        HDR_H  = 48  # header zone height

        # Title
        title = NSTextField.alloc().initWithFrame_(
            NSMakeRect(PAD, H - HDR_H + (HDR_H - 20) / 2, W - 2*PAD - 36, 20)
        )
        title.setEditable_(False)
        title.setSelectable_(False)
        title.setBordered_(False)
        title.setBackgroundColor_(NSColor.clearColor())
        title.setStringValue_("Transcription History")
        title.setFont_(NSFont.boldSystemFontOfSize_(14))
        title.setTextColor_(NSColor.colorWithRed_green_blue_alpha_(0.11, 0.11, 0.13, 1.0))
        container.addSubview_(title)

        # Collapse (−) button top-right
        dash_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(W - PAD - 26, H - HDR_H + (HDR_H - 26) / 2, 26, 26)
        )
        dash_btn.setTitle_("−")
        dash_btn.setBezelStyle_(12)
        dash_btn.setButtonType_(0)
        dash_btn.setFont_(NSFont.boldSystemFontOfSize_(16))
        dash_btn.setTarget_(self)
        dash_btn.setAction_(b'collapseHistory:')
        container.addSubview_(dash_btn)

        # Divider below header
        div_y = H - HDR_H
        container.addSubview_(
            _DividerView.alloc().initWithFrame_(NSMakeRect(0, div_y, W, 1))
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

    def collapseHistory_(self, sender):
        self._panel.orderOut_(None)
        self._visible = False
        _history_open[0] = False

    @objc.python_method
    def toggle_near_frame(self, pill_frame):
        if self._visible:
            self._panel.orderOut_(None)
            self._visible = False
            _history_open[0] = False
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
        _history_open[0] = True

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
        btn.setTitle_("⧉")
        btn.setBezelStyle_(12)
        btn.setButtonType_(0)
        btn.setFont_(NSFont.systemFontOfSize_(15))
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
        IV        = 14    # card vertical inner padding (top & bottom)
        IH        = 14    # card horizontal inner padding
        GAP       = 10    # gap between cards
        BTN_W     = 28    # copy button — square symbol button
        BTN_H     = 26
        HEAD_H    = 18    # time + badge header row
        BADGE_H   = 16    # badge pill height
        ROW_H     = BTN_H # label + copy-button row height
        LABEL_H   = 13
        DIV_TOTAL = 20    # space reserved for section divider

        card_w = W - 2 * CARD_X
        text_w = card_w - 2 * IH

        body_font  = NSFont.systemFontOfSize_(13)
        meta_font  = NSFont.systemFontOfSize_(11)
        label_font = NSFont.boldSystemFontOfSize_(10)

        near_black = NSColor.colorWithRed_green_blue_alpha_(0.10, 0.10, 0.12, 1.0)
        mid_gray   = NSColor.colorWithRed_green_blue_alpha_(0.55, 0.55, 0.58, 1.0)
        blue_bg    = NSColor.colorWithRed_green_blue_alpha_(0.20, 0.47, 0.96, 1.0)
        purple_bg  = NSColor.colorWithRed_green_blue_alpha_(0.52, 0.28, 0.90, 1.0)
        blue_text  = NSColor.colorWithRed_green_blue_alpha_(0.14, 0.38, 0.82, 1.0)
        purple_text= NSColor.colorWithRed_green_blue_alpha_(0.44, 0.20, 0.78, 1.0)

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
                card_h = IV + HEAD_H + 10 + th + 8 + BTN_H + IV
                items.append((entry, card_h, {'th': th}))
            else:
                cmd = entry.get('command', '')
                ch  = self._text_height(cmd, body_font, text_w)
                oh  = self._text_height(entry['text'], body_font, text_w)
                card_h = IV + HEAD_H + 10 + ROW_H + 4 + ch + DIV_TOTAL + ROW_H + 4 + oh + IV
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

            # Header: time (gray) + colored mode badge pill
            time_str  = entry['time'].strftime("%-I:%M %p")
            is_trans  = entry['mode'] == 'transcribe'
            badge_txt = "TRANSCRIBE" if is_trans else "COMMAND"
            badge_bg  = blue_bg if is_trans else purple_bg
            badge_w   = 72 if is_trans else 66
            card.addSubview_(self._make_label(
                time_str, meta_font, mid_gray,
                NSMakeRect(IH, y - HEAD_H + (HEAD_H - 13) / 2, 60, 13)
            ))
            badge = _BadgeView.alloc().initWithFrame_(
                NSMakeRect(IH + 66, y - HEAD_H + (HEAD_H - BADGE_H) / 2, badge_w, BADGE_H)
            )
            badge._text  = badge_txt
            badge._color = badge_bg
            card.addSubview_(badge)
            y -= HEAD_H + 10

            if entry['mode'] == 'transcribe':
                th = dims['th']
                card.addSubview_(self._make_label(
                    entry['text'], body_font, near_black,
                    NSMakeRect(IH, y - th, text_w, th)
                ))
                y -= th + 8
                card.addSubview_(self._make_copy_btn(
                    entry['text'],
                    NSMakeRect(card_w - IH - BTN_W, y - BTN_H, BTN_W, BTN_H)
                ))

            else:
                ch  = dims['ch']
                oh  = dims['oh']
                cmd = dims['cmd']
                lv  = (ROW_H - LABEL_H) / 2

                # Voice command section
                card.addSubview_(self._make_label(
                    "VOICE COMMAND", label_font, purple_text,
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

                # Section divider
                card.addSubview_(
                    _DividerView.alloc().initWithFrame_(
                        NSMakeRect(IH, y - DIV_TOTAL // 2, card_w - 2*IH, 1)
                    )
                )
                y -= DIV_TOTAL

                # Output section
                card.addSubview_(self._make_label(
                    "OUTPUT", label_font, blue_text,
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

_WHISPER_ID = "openai/whisper-large-v3-turbo"


def _prompt_hf_token_if_needed():
    """Prompt for an HF token when either model isn't cached yet."""
    from pathlib import Path
    cache = Path.home() / ".cache" / "huggingface" / "hub"

    def is_cached(model_id):
        return (cache / ("models--" + model_id.replace("/", "--"))).exists()

    whisper_ok = is_cached(_WHISPER_ID)
    gemma_ok   = is_cached(GEMMA_MODEL_ID)

    if whisper_ok and gemma_ok:
        return  # nothing to download

    missing = []
    if not whisper_ok:
        missing.append(f"  • {_WHISPER_ID}")
    if not gemma_ok:
        missing.append(f"  • {GEMMA_MODEL_ID}")

    print("\nThe following models are not cached and need to be downloaded:")
    for m in missing:
        print(m)
    print()
    print("A Hugging Face token (huggingface.co/settings/tokens) speeds up the download.")
    print("If you skip this, the download will be very slow (no rate-limit bypass).")
    print()
    token = input("HF token [press Enter to skip]: ").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        print()
    else:
        print("⚠  Warning: downloading without a token will be very slow.\n")


def main():
    request_accessibility()
    _prompt_hf_token_if_needed()

    recording    = False
    frames       = []
    stream       = None
    lock         = threading.Lock()
    fn_held      = [False]
    shift_seen   = [False]
    models_ready = [False]
    _pipe        = [None]          # set by _load_models
    tap_ref      = [None]          # filled after CGEventTap is created
    record_start = [0.0]           # monotonic time when recording began
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

        set_widget('active', 'Formatting...')
        _gemma_in.put(('transcribe', transcript))
        formatted = _gemma_out.get()
        print(f"Formatted:  {formatted}")

        _transcription_history.append({'time': datetime.datetime.now(), 'mode': 'transcribe', 'text': formatted})
        paste_at_cursor(formatted)
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
            record_start[0] = time.monotonic()
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
        # macOS disables the tap when it times out; re-enable immediately so we
        # never miss an Fn-release event and leave recording stuck.
        if event_type == 0xFFFFFFFE:  # kCGEventTapDisabledByTimeout
            if tap_ref[0]:
                CGEventTapEnable(tap_ref[0], True)
            return event

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
    tap_ref[0] = tap
    CFRunLoopAddSource(
        CFRunLoopGetMain(),
        Cocoa.CFMachPortCreateRunLoopSource(None, tap, 0),
        kCFRunLoopCommonModes,
    )

    def _watchdog():
        MAX_RECORD_SECS = 120  # force-stop if stuck recording for 2+ minutes
        while True:
            time.sleep(3)
            if recording and (time.monotonic() - record_start[0]) > MAX_RECORD_SECS:
                print("Watchdog: recording stuck — forcing stop.")
                fn_held[0]    = False
                shift_seen[0] = False
                threading.Thread(target=stop_recording, args=('transcribe',), daemon=True).start()

    threading.Thread(target=_watchdog, daemon=True).start()

    def _load_models():
        set_widget('active', 'Loading models...')
        whisper_done = threading.Event()
        gemma_done   = threading.Event()

        def _load_whisper():
            print("Loading Whisper...")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype  = torch.float16 if device == "mps" else torch.float32
            w_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                _WHISPER_ID, dtype=dtype,
                low_cpu_mem_usage=True, use_safetensors=True,
            )
            w_model.to(device)
            w_processor = AutoProcessor.from_pretrained(_WHISPER_ID)
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
                    sys_prompt = (
                        "You are a transcription editor for spoken English. "
                        "Your only job is to fix punctuation and sentence flow. "
                        "You must not change, add, or remove any words.\n\n"
                        "Rules:\n"
                        "- When two phrases are part of the same continuous thought, join them with a comma, "
                        "em-dash, semicolon, or conjunction — never a full stop\n"
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
                else:  # command
                    command, selected = task[1], task[2]
                    sys_prompt = (
                        "You are a precise voice command executor. "
                        "Execute the user's command exactly and return only the result — "
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
