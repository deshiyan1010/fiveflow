import subprocess

from AppKit import (
    NSApplication, NSPanel, NSScreen, NSColor, NSStatusWindowLevel,
    NSApplicationActivationPolicyAccessory,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorTransient,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
)
from Foundation import NSObject, NSTimer, NSMakeRect

from .. import state
from ..config import WIN_W, WIN_H, NSBorderlessWindowMask, NSNonactivatingPanelMask, NSBackingStoreBuffered
from .widgets import PillView, set_widget
from .history import HistoryWindowController


class AppDelegate(NSObject):
    def applicationWillTerminate_(self, notification):
        subprocess.run(["afplay", "/System/Library/Sounds/Bottle.aiff"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

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
        state.toggle_history_callback[0] = _toggle_history
