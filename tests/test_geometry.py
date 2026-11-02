import unittest
from fiveflow.ui.geometry import history_position, pill_contains


class GeometryTests(unittest.TestCase):
    def test_history_stays_on_pills_secondary_screen(self):
        screens = [(0, 0, 1920, 1080), (-1280, 0, 1280, 900)]
        x, y = history_position((-800, 20, 280, 100), screens, (440, 480))
        self.assertGreaterEqual(x, -1270)
        self.assertLessEqual(x + 440, -10)
        self.assertGreaterEqual(y, 10)

    def test_history_opens_below_pill_near_screen_top(self):
        x, y = history_position((600, 900, 280, 100), [(0, 0, 1920, 1080)], (440, 480))
        self.assertLess(y + 480, 900)

    def test_pill_hover_includes_gap_to_close_button(self):
        self.assertTrue(pill_contains(280, 1, True, 234, 20))
        self.assertFalse(pill_contains(280, 0, False, 10, 20))

    def test_history_selects_nearest_visible_screen_when_pill_is_below_dock(self):
        x, y = history_position((-800, 0, 280, 100), [(0, 80, 1920, 1000), (-1280, 80, 1280, 820)], (440, 480))
        self.assertLess(x + 440, 0)
        self.assertGreaterEqual(y, 90)
