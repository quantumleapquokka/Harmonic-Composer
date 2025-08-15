from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "midi"       # expects subfolders per-genre with .mid files
ARTIFACTS = ROOT / "outputs"
PROC_DIR = ARTIFACTS / "processed"
MODEL_DIR = ARTIFACTS / "models"
SAMPLE_DIR = ARTIFACTS / "samples"

# Ensure folders exist
for d in [ARTIFACTS, PROC_DIR, MODEL_DIR, SAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Preprocessing/encoding
TIME_RESOLUTION = 4   # steps per quarter note (4 = 16th notes)
MAX_SEQ_LEN = 256     # cap sequence len for training
MIN_NOTES_PER_FILE = 32

# Special tokens
PAD = 0
START = 1
END = 2
REST = 3
PITCH_OFFSET = 4       # actual MIDI pitches will be offset by this to avoid clashing with tokens

# Training
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
EMBED_DIM = 128
LSTM_UNITS = 256
GENRE_EMBED_DIM = 16
