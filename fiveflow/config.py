import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

SAMPLE_RATE  = 16000
WIN_W, WIN_H = 280, 100

NSBorderlessWindowMask   = 0
NSNonactivatingPanelMask = 1 << 7
NSBackingStoreBuffered   = 2

kCGKeyboardEventKeycode     = 9
kCGEventFlagMaskSecondaryFn = 0x00800000
kCGEventFlagMaskShift       = 0x00020000

WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
GEMMA_MODEL_ID   = "mlx-community/gemma-4-e2b-it-4bit"
