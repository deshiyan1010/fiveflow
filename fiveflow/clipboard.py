"""Temporary clipboard access that preserves every item and representation."""
from contextlib import contextmanager
import threading
import time
import uuid

_lock = threading.RLock()


def _native():
    from AppKit import NSPasteboard, NSPasteboardItem, NSPasteboardTypeString
    return NSPasteboard.generalPasteboard(), NSPasteboardItem, NSPasteboardTypeString


def _write_text(board, text, text_type):
    board.clearContents()
    if not board.setString_forType_(text, text_type):
        raise RuntimeError('Could not write to the clipboard')
    return board.changeCount()


@contextmanager
def temporary_text(text):
    with _lock:
        board, item_class, text_type = _native()
        snapshot = []
        before = board.changeCount()
        for item in board.pasteboardItems() or []:
            clone = item_class.alloc().init()
            for kind in item.types():
                data = item.dataForType_(kind)
                if data is None or not clone.setData_forType_(data, kind):
                    raise RuntimeError('Could not preserve clipboard data')
            snapshot.append(clone)
        if board.changeCount() != before:
            raise RuntimeError('Clipboard changed while being preserved; retry')
        transaction = _Transaction(board, text_type)
        try:
            board.clearContents()
            transaction.expected = board.changeCount()
            if not board.setString_forType_(text, text_type):
                transaction.expected = board.changeCount()
                raise RuntimeError('Could not write to the clipboard')
            transaction.expected = board.changeCount()
            yield transaction
        finally:
            # Never replace a newer copy made by the user or another app.
            if board.changeCount() == transaction.expected:
                board.clearContents()
                if snapshot and not board.writeObjects_(snapshot):
                    raise RuntimeError('Could not restore the clipboard')


class _Transaction:
    def __init__(self, board, text_type):
        self.board = board
        self.text_type = text_type
        self.expected = board.changeCount()

    def read_changed_text(self):
        count = self.board.changeCount()
        if count == self.expected:
            return None
        text = self.board.stringForType_(self.text_type)
        if self.board.changeCount() != count:
            raise RuntimeError('Clipboard changed while reading the selection')
        self.expected = count
        return text or ''


def selected_text(copy_action, timeout=0.75):
    # A unique marker detects copies even when the selection equals the old text.
    with temporary_text(f'fiveflow-selection-{uuid.uuid4()}') as transaction:
        copy_action()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = transaction.read_changed_text()
            if text is not None:
                return text
            time.sleep(0.01)
        return ''


def accessibility_selection():
    # Some editors copy the whole line on Cmd+C with no selection. Prefer the
    # actual selected-text attribute; None means this app doesn't expose it.
    from ApplicationServices import (
        AXUIElementCreateSystemWide, AXUIElementCopyAttributeValue,
        kAXFocusedUIElementAttribute, kAXSelectedTextAttribute,
    )
    error, focused = AXUIElementCopyAttributeValue(
        AXUIElementCreateSystemWide(), kAXFocusedUIElementAttribute, None)
    if error or focused is None:
        return None
    error, text = AXUIElementCopyAttributeValue(focused, kAXSelectedTextAttribute, None)
    return text if not error and isinstance(text, str) else None


def paste_text(text, paste_action):
    with temporary_text(text):
        time.sleep(0.05)
        paste_action()
        time.sleep(0.2)


def copy_text(text):
    with _lock:
        board, _, text_type = _native()
        _write_text(board, text, text_type)
