import math
import subprocess
import queue

import objc
from AppKit import (
    NSView, NSColor, NSBezierPath, NSFont, NSAttributedString,
    NSForegroundColorAttributeName, NSFontAttributeName,
    NSTrackingArea, NSTrackingMouseEnteredAndExited, NSTrackingActiveAlways,
    NSTextField, NSButton,
)
from Foundation import NSObject, NSTimer, NSMakeRect, NSMakePoint, NSDictionary

from .. import state
from ..config import WIN_W, WIN_H

import logging
logging.basicConfig(filename='/tmp/fiveflow_debug.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')


class PillView(NSView):
    _state       = 'idle'
    _label       = ''
    _anim_tick   = 0
    _pill_scale  = 0.0
    _hovering    = False
    _hover_scale = 0.0
    _did_drag    = False

    _CLOSE_R = 9.0

    def acceptsFirstMouse_(self, event):
        return True

    def updateTrackingAreas(self):
        for area in list(self.trackingAreas()):
            self.removeTrackingArea_(area)
        bw  = self.bounds().size.width
        hs  = self._hover_scale
        pw  = 45  + (160 - 45) * hs
        ph  = 8   + (28  - 8)  * hs
        px  = (bw - pw) / 2
        PAD = 3
        extra = (15 + self._CLOSE_R + PAD) * hs
        track_rect = NSMakeRect(px - PAD, 8 - PAD, pw + PAD * 2 + extra, ph + PAD * 2)
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
        if self._hovering:
            cx = px + pw + 15
            cy = py + ph / 2
            dx = point.x - cx
            dy = point.y - cy
            if dx * dx + dy * dy <= self._CLOSE_R ** 2:
                return self
        if px - 6 <= point.x <= px + pw + 6 and py - 4 <= point.y <= py + ph + 4:
            return self
        return None

    def mouseDown_(self, event):
        self._did_drag = False

    def mouseDragged_(self, event):
        if self._state != 'idle':
            return
        self._did_drag = True
        origin = self.window().frame().origin
        self.window().setFrameOrigin_(NSMakePoint(
            origin.x + event.deltaX(),
            origin.y - event.deltaY(),
        ))

    def mouseUp_(self, event):
        logging.debug(f"PillView mouseUp_: state={self._state}, did_drag={self._did_drag}")
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
            logging.debug("PillView mouseUp_: clicked close button")
            from AppKit import NSApplication
            NSApplication.sharedApplication().terminate_(None)
            return
        
        logging.debug(f"PillView mouseUp_: checking toggle callback. callback exists: {state.toggle_history_callback[0] is not None}")
        
        if self._did_drag:
            if state.history_open[0] and state.toggle_history_callback[0]:
                logging.debug("PillView mouseUp_: invoking toggle callback after drag")
                state.toggle_history_callback[0]()
        else:
            if state.toggle_history_callback[0]:
                logging.debug("PillView mouseUp_: invoking toggle callback")
                state.toggle_history_callback[0]()
        self._did_drag = False

    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(self.bounds())
        bw = self.bounds().size.width
        bh = self.bounds().size.height
        dark = NSColor.colorWithRed_green_blue_alpha_(0.11, 0.11, 0.11, 1.0)
        gray = NSColor.colorWithRed_green_blue_alpha_(0.557, 0.557, 0.576, 1.0)
        s    = self._pill_scale

        if self._state == 'recording' or s > 0:
            pw = 60  + (160 - 60)  * s
            ph = 6   + (38  - 6)   * s
            px = (bw - pw) / 2
            py = 8   + ((bh - 38) / 2 - 8) * s

            self._pill(px, py, pw, ph, dark if s > 0.4 else gray)

            if s > 0.4:
                bar_alpha = min(1.0, (s - 0.4) / 0.3)
                level     = state.audio_level
                n, bar_w, gap = 7, 3, 5
                total_w = n * bar_w + (n - 1) * gap
                x0      = px + (pw - total_w) / 2
                max_h   = (38 - 14) * s
                min_h   = 4
                NSColor.colorWithRed_green_blue_alpha_(1, 1, 1, bar_alpha).set()
                for i in range(n):
                    phase  = self._anim_tick * 0.18 + i * 0.75
                    amp    = 0.15 + 0.85 * level
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
                label = "Close history" if state.history_open[0] else "History"
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
                self._text('x', cx, cy, 13)

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
                self._state, self._label = state.state_queue.get_nowait()
                changed = True
        except queue.Empty:
            pass

        if self._state == 'recording':
            self._pill_scale = min(1.0, self._pill_scale + 0.12)
            self._anim_tick += 1
            changed = True
        elif self._pill_scale > 0:
            self._pill_scale = max(0.0, self._pill_scale - 0.12)
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


class CopyButton(NSButton):
    _copy_text = ''

    def performCopy_(self, sender):
        subprocess.run(["pbcopy"], input=self._copy_text.encode())
        self.setTitle_("\u2713")
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.5, self, b'resetCopyTitle:', None, False
        )

    def resetCopyTitle_(self, timer):
        self.setTitle_("\u29c9")


class DividerView(NSView):
    def drawRect_(self, rect):
        NSColor.colorWithRed_green_blue_alpha_(0.85, 0.85, 0.87, 1.0).set()
        NSBezierPath.fillRect_(self.bounds())


class BadgeView(NSView):
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


class EntryCardView(NSView):
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


class HistoryContainerView(NSView):
    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(self.bounds())
        bw   = self.bounds().size.width
        bh   = self.bounds().size.height
        clip = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(0, 0, bw, bh), 16, 16
        )
        clip.addClip()
        NSColor.colorWithRed_green_blue_alpha_(0.985, 0.985, 0.99, 1.0).set()
        NSBezierPath.fillRect_(NSMakeRect(0, 0, bw, bh))
        NSColor.colorWithRed_green_blue_alpha_(0.96, 0.96, 0.975, 1.0).set()
        NSBezierPath.fillRect_(NSMakeRect(0, bh - 52, bw, 52))


def set_widget(state_str, label=''):
    state.state_queue.put((state_str, label))
