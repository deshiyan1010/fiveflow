import math

import objc
from AppKit import (
    NSPanel, NSView, NSColor, NSBezierPath, NSFont, NSAttributedString,
    NSForegroundColorAttributeName, NSFontAttributeName,
    NSScreen, NSStatusWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorTransient,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSScrollView, NSTextField, NSButton,
)
from Foundation import NSObject, NSMakeRect, NSMakeSize, NSDictionary

from .. import state
from ..config import NSBorderlessWindowMask, NSNonactivatingPanelMask, NSBackingStoreBuffered
from .widgets import (
    HistoryContainerView, DividerView, BadgeView, EntryCardView, CopyButton,
)


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

        container = HistoryContainerView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        panel.setContentView_(container)

        PAD    = 16
        HDR_H  = 48

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

        dash_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(W - PAD - 26, H - HDR_H + (HDR_H - 26) / 2, 26, 26)
        )
        dash_btn.setTitle_("\u2212")
        dash_btn.setBezelStyle_(12)
        dash_btn.setButtonType_(0)
        dash_btn.setFont_(NSFont.boldSystemFontOfSize_(16))
        dash_btn.setTarget_(self)
        dash_btn.setAction_(b'collapseHistory:')
        container.addSubview_(dash_btn)

        div_y = H - HDR_H
        container.addSubview_(
            DividerView.alloc().initWithFrame_(NSMakeRect(0, div_y, W, 1))
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
        state.history_open[0] = False

    @objc.python_method
    def toggle_near_frame(self, pill_frame):
        if self._visible:
            self._panel.orderOut_(None)
            self._visible = False
            state.history_open[0] = False
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
        state.history_open[0] = True

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
        btn = CopyButton.alloc().initWithFrame_(frame)
        btn._copy_text = copy_text
        btn.setTitle_("\u29c9")
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

        history = list(state.transcription_history)
        W = self._content_W

        CARD_X    = 4
        IV        = 14
        IH        = 14
        GAP       = 10
        BTN_W     = 28
        BTN_H     = 26
        HEAD_H    = 18
        BADGE_H   = 16
        ROW_H     = BTN_H
        LABEL_H   = 13
        DIV_TOTAL = 20

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

        y_cursor = total_h - GAP

        for entry, card_h, dims in items:
            card_y = y_cursor - card_h
            card = EntryCardView.alloc().initWithFrame_(
                NSMakeRect(CARD_X, card_y, card_w, card_h)
            )
            self._content_view.addSubview_(card)

            y = card_h - IV

            time_str  = entry['time'].strftime("%-I:%M %p")
            is_trans  = entry['mode'] == 'transcribe'
            badge_txt = "TRANSCRIBE" if is_trans else "COMMAND"
            badge_bg  = blue_bg if is_trans else purple_bg
            badge_w   = 72 if is_trans else 66
            card.addSubview_(self._make_label(
                time_str, meta_font, mid_gray,
                NSMakeRect(IH, y - HEAD_H + (HEAD_H - 13) / 2, 60, 13)
            ))
            badge = BadgeView.alloc().initWithFrame_(
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

                card.addSubview_(
                    DividerView.alloc().initWithFrame_(
                        NSMakeRect(IH, y - DIV_TOTAL // 2, card_w - 2*IH, 1)
                    )
                )
                y -= DIV_TOTAL

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

        self._content_view.scrollRectToVisible_(NSMakeRect(0, total_h - 1, W, 1))
