import queue

state_queue  = queue.Queue()
audio_level  = 0.0

transcription_history   = []
toggle_history_callback = [None]
history_open            = [False]
