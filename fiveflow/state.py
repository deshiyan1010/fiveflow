import queue

state_queue  = queue.Queue()
ui_callbacks = queue.Queue()
audio_level  = 0.0

transcription_history   = []
toggle_history_callback = [None]
history_open            = [False]

transcribe_keycode      = 63
listening_for_key       = False
on_key_assigned = None
on_history_changed = None


def add_history(entry):
    transcription_history.append(entry)
    if callable(on_history_changed):
        ui_callbacks.put((on_history_changed, None))
