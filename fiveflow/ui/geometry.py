"""Pure geometry shared by the macOS views and layout regression tests."""

def pill_contains(width, scale, hovering, x, y):
    pill_width = 45 + (160 - 45) * scale
    pill_height = 8 + (28 - 8) * scale
    left = (width - pill_width) / 2
    extra = 24 if hovering else 0
    return left - 6 <= x <= left + pill_width + 6 + extra and 4 <= y <= 12 + pill_height


def history_position(pill, screens, panel):
    x, y, w, h = pill
    width, height = panel
    center = x + w / 2, y + h / 2
    screen = min(screens, key=lambda s:
                 max(s[0] - center[0], 0, center[0] - s[0] - s[2]) ** 2
                 + max(s[1] - center[1], 0, center[1] - s[1] - s[3]) ** 2)
    sx, sy, sw, sh = screen
    left = max(sx + 10, min(x + (w - width) / 2, sx + sw - width - 10))
    top = y + h + 8
    if top + height > sy + sh - 10:
        top = y - height - 8
    bottom = max(sy + 10, min(top, sy + sh - height - 10))
    return left, bottom
