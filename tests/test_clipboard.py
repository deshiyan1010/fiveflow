import unittest
from unittest import mock

from fiveflow import clipboard


class Item:
    def __init__(self, values=None):
        self.values = dict(values or {})

    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        return self

    def types(self):
        return list(self.values)

    def dataForType_(self, kind):
        return self.values[kind]

    def setData_forType_(self, data, kind):
        self.values[kind] = data
        return True


class Board:
    def __init__(self, items):
        self.items = items
        self.count = 1
        self.fail_write = False

    def changeCount(self):
        return self.count

    def pasteboardItems(self):
        return self.items

    def clearContents(self):
        self.items = []
        self.count += 1

    def setString_forType_(self, text, kind):
        if self.fail_write:
            return False
        self.items = [Item({kind: text.encode()})]
        return True

    def stringForType_(self, kind):
        return self.items[0].values.get(kind, b'').decode() if self.items else None

    def writeObjects_(self, items):
        self.items = items
        return True

    def copy(self, text):
        self.clearContents()
        self.setString_forType_(text, 'text')


class ClipboardTests(unittest.TestCase):
    def setUp(self):
        self.original = [{'text': b'old', 'rtf': b'{rich text}'}, {'png': b'\x89PNG'}]
        self.board = Board([Item(values) for values in self.original])
        self.enterContext(mock.patch.object(clipboard, '_native', return_value=(self.board, Item, 'text')))

    def values(self):
        return [item.values for item in self.board.items]

    def test_restores_all_items_and_formats_after_paste(self):
        def paste():
            self.assertEqual(self.board.stringForType_('text'), 'new')
        with mock.patch.object(clipboard.time, 'sleep'):
            clipboard.paste_text('new', paste)
        self.assertEqual(self.values(), self.original)

    def test_restores_rich_clipboard_when_keyboard_injection_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'keyboard'):
            with clipboard.temporary_text('new'):
                raise RuntimeError('keyboard')
        self.assertEqual(self.values(), self.original)

    def test_does_not_overwrite_user_copy_during_paste(self):
        with clipboard.temporary_text('new'):
            self.board.copy('user copy')
        self.assertEqual(self.values(), [{'text': b'user copy'}])

    def test_selection_equal_to_original_clipboard_is_detected(self):
        self.assertEqual(clipboard.selected_text(lambda: self.board.copy('old')), 'old')
        self.assertEqual(self.values(), self.original)

    def test_selection_preserves_indentation_and_trailing_newlines(self):
        text = '    code()\n\n'
        self.assertEqual(clipboard.selected_text(lambda: self.board.copy(text)), text)
        self.assertEqual(self.values(), self.original)

    def test_no_selection_restores_clipboard(self):
        self.assertEqual(clipboard.selected_text(lambda: None, timeout=0), '')
        self.assertEqual(self.values(), self.original)

    def test_failed_temporary_write_still_restores_original(self):
        self.board.fail_write = True
        with self.assertRaises(RuntimeError):
            with clipboard.temporary_text('new'):
                self.fail('write should fail')
        self.assertEqual(self.values(), self.original)

    def test_new_user_copy_after_selection_read_is_preserved(self):
        with clipboard.temporary_text('marker') as transaction:
            self.board.copy('selected')
            self.assertEqual(transaction.read_changed_text(), 'selected')
            self.board.copy('user copied this later')
        self.assertEqual(self.values(), [{'text': b'user copied this later'}])

    def test_empty_clipboard_stays_empty_after_temporary_paste(self):
        self.board.items = []
        with clipboard.temporary_text('new'):
            pass
        self.assertEqual(self.values(), [])

    def test_copy_button_leaves_copied_text_in_clipboard(self):
        clipboard.copy_text('history')
        self.assertEqual(self.values(), [{'text': b'history'}])

    def test_accessibility_selection_preserves_whitespace(self):
        import sys
        services = mock.MagicMock()
        services.AXUIElementCopyAttributeValue.side_effect = [(0, object()), (0, '  selected\n')]
        with mock.patch.dict(sys.modules, {'ApplicationServices': services}):
            self.assertEqual(clipboard.accessibility_selection(), '  selected\n')

    def test_accessibility_unavailable_returns_fallback_signal(self):
        import sys
        services = mock.MagicMock()
        services.AXUIElementCopyAttributeValue.return_value = (-25211, None)
        with mock.patch.dict(sys.modules, {'ApplicationServices': services}):
            self.assertIsNone(clipboard.accessibility_selection())
