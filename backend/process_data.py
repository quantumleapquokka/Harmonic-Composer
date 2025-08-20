# File 1: process_data.py (Modified to Transcribe Audio Files)
# Purpose: Transcribes raw audio files (.au) to MIDI, then parses the MIDI files.

import numpy as np
import json
from pathlib import Path
from music21 import converter, note, chord, interval, pitch
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from chords_tools import normalize_chord
from tokenizer import build_vocab, save_vocab, encode
import re


# NEW: Import the transcription library
from basic_pitch.inference import predict

# --- Configuration ---
# Path to your folder containing genre subfolders with .au files
AUDIO_DATASET_PATH = Path(__file__).parent / 'genres' 
# Path where the NEW MIDI files will be saved after transcription
TRANSCRIBED_MIDI_PATH = Path(__file__).parent / 'genres_midi'
# Path for the final processed data for the AI
PROCESSED_DATA_PATH = Path(__file__).parent / 'processed_data'

SEQUENCE_LENGTH = 100 

def parse_genres_dataset(audio_path: Path, midi_path: Path):
    """
    First, transcribes .au files to MIDI, then parses the MIDI files.
    """
    # --- STAGE 1: AUDIO TRANSCRIPTION ---
    print("🎤 Stage 1: Transcribing audio files to MIDI...")
    if not audio_path.exists():
        print(f"❌ Error: Audio dataset directory not found at '{audio_path}'")
        return None
    
    audio_files = list(audio_path.glob('**/*.au'))
    if not audio_files:
        print(f"❌ Error: No .au audio files found in '{audio_path}'.")
        return None

    for audio_file in tqdm(audio_files, desc="Transcribing Audio"):
        try:
            # Create a corresponding output path for the MIDI file
            relative_path = audio_file.relative_to(audio_path)
            output_file_path = midi_path / relative_path.with_suffix('.mid')
            
            # Skip if the MIDI file already exists
            if output_file_path.exists():
                continue
                
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use basic-pitch to predict and save the MIDI file
            model_output, midi_data, note_events = predict(str(audio_file))
            midi_data.write(str(output_file_path))

        except Exception as e:
            print(f"\nCould not process {audio_file.name}. Error: {e}")
    print("✅ Transcription complete!")

    # --- STAGE 2: MIDI PARSING ---
    print("\n🎼 Stage 2: Parsing transcribed MIDI files...")
    all_notes = []
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    genre_folders = [f for f in midi_path.iterdir() if f.is_dir()]
    print(f"Found {len(genre_folders)} genre folders in the transcribed directory.")

    for genre_folder in tqdm(genre_folders, desc="Processing Genres"):
        genre_name = genre_folder.name.lower()
        genre_token = f"<{genre_name}>"
        
        for midi_file in tqdm(list(genre_folder.glob('**/*.mid*')), desc=f"Parsing {genre_folder.name}", leave=False):
            try:
                piece_notes = [genre_token] 
                score = converter.parse(midi_file)
                key = score.analyze('key')
                i = interval.Interval(key.tonic, pitch.Pitch('C' if key.mode == 'major' else 'A'))
                score = score.transpose(i)
                
                for element in score.flat.notes:
                    if isinstance(element, note.Note):
                        piece_notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        piece_notes.append('.'.join(str(n) for n in element.normalOrder))
                
                all_notes.extend(piece_notes)
            except Exception as e:
                print(f"\nCould not parse MIDI file {midi_file}: {e}")
            
    with open(PROCESSED_DATA_PATH / 'parsed_notes.json', 'w') as f:
        json.dump(all_notes, f)
    return all_notes

def prepare_sequences_for_training(notes, sequence_length):
    """Creates and saves training sequences from the parsed notes, plus chord conditioning."""

    # ---- 0) Force-include a tiny, universal chord set (12 major + 12 minor)
    roots = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    basic_chords = [normalize_chord(r + "maj") for r in roots] + \
                   [normalize_chord(r + "min") for r in roots]

    # ---- 1) Build deterministic vocab with specials; include chord tokens
    vocab = build_vocab([notes], min_freq=1, add_specials=True, extra_tokens=basic_chords)
    save_vocab(PROCESSED_DATA_PATH / "vocab_map.json", vocab)
    note_to_int = vocab.token_to_id

    # ---- 2) Create melody input/output (teacher-forced shift by 1)
    network_input, network_output = [], []
    for i in tqdm(range(len(notes) - sequence_length), desc="Creating sequences"):
        seq_in  = notes[i : i + sequence_length]
        seq_out = notes[i + 1 : i + sequence_length + 1]
        network_input.append([note_to_int[t] for t in seq_in])
        network_output.append([note_to_int[t] for t in seq_out])

    # ---- 3) Infer a simple chord per window and build X_chords (repeat across the window)
    # Heuristic: pick major/minor whose triad pcs best match the window's pitch-class histogram.
    # This keeps alignment trivial (same length as seq_in).
    note_re = re.compile(r"^[A-Ga-g][#b]?\d+$")  # e.g., C4, F#3
    pc_map  = {"C":0,"C#":1,"D":2,"D#":3,"E":4,"F":5,"F#":6,"G":7,"G#":8,"A":9,"A#":10,"B":11}
    major_triad = [0,4,7]
    minor_triad = [0,3,7]

    def name_to_pc(name: str) -> int:
        # strip octave to pitch class name, normalize flats to sharps
        n = name[:-1].upper() if name[-1].isdigit() else name.upper()
        n = n.replace("DB","C#").replace("EB","D#").replace("GB","F#").replace("AB","G#").replace("BB","A#")
        return pc_map.get(n, None)

    def best_chord_for_window(tok_window):
        # collect pitch classes present in the window
        pcs = []
        for t in tok_window:
            if note_re.match(t):
                pc = name_to_pc(t)
                if pc is not None:
                    pcs.append(pc)
        if not pcs:
            return "N.C."  # no chord info; neutral conditioning

        hist = [0]*12
        for p in pcs:
            hist[p] += 1

        best_name, best_score = "N.C.", -1
        for rname, rpc in pc_map.items():
            # score major
            maj_pcs = [(rpc + x) % 12 for x in major_triad]
            score_maj = sum(hist[x] for x in maj_pcs)
            if score_maj > best_score:
                best_name, best_score = normalize_chord(rname + "maj"), score_maj
            # score minor
            min_pcs = [(rpc + x) % 12 for x in minor_triad]
            score_min = sum(hist[x] for x in min_pcs)
            if score_min > best_score:
                best_name, best_score = normalize_chord(rname + "min"), score_min
        return best_name

    X_chords = []
    for i in range(len(notes) - sequence_length):
        seq_in_tokens = notes[i : i + sequence_length]                  # original string tokens
        chosen_chord  = best_chord_for_window(seq_in_tokens)            # e.g., "Cmaj" / "Amin" / "N.C."
        # ensure it's normalized and in vocab (we force-included majors/minors)
        chosen_chord  = normalize_chord(chosen_chord)
        # encode returns a list of ids; repeat across the sequence length
        try:
            chord_id = encode([chosen_chord], vocab, add_bos=False, add_eos=False, use_unk=True)[0]
        except Exception:
            # fallback to <unk> if something unexpected sneaks in
            chord_id = vocab.token_to_id.get("<unk>", 0)
        X_chords.append([chord_id] * sequence_length)

    # ---- 4) Shape arrays and one-hot targets
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    from tensorflow.keras.utils import to_categorical
    network_output = to_categorical(network_output, num_classes=vocab.size)

    # ---- 5) Save arrays
    print("\n💾 Saving final processed data to disk...")
    np.save(PROCESSED_DATA_PATH / "sequences.npy", network_input)
    np.save(PROCESSED_DATA_PATH / "targets.npy", network_output)
    np.save(PROCESSED_DATA_PATH / "X_chords.npy", np.array(X_chords, dtype=np.int32))
    print("✅ Saved sequences.npy, targets.npy, X_chords.npy, and vocab_map.json")




if __name__ == '__main__':
    # The function now takes the audio and MIDI paths as arguments
    parsed = parse_genres_dataset(AUDIO_DATASET_PATH, TRANSCRIBED_MIDI_PATH)
    if parsed:
        prepare_sequences_for_training(parsed, SEQUENCE_LENGTH)
