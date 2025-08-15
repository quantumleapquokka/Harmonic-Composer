# File 1: process_data.py (Modified to Transcribe Audio Files)
# Purpose: Transcribes raw audio files (.au) to MIDI, then parses the MIDI files.

import numpy as np
import json
from pathlib import Path
from music21 import converter, note, chord, interval, pitch
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

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
        #try:
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

        #except Exception as e:
        #    print(f"\nCould not process {audio_file.name}. Error: {e}")
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
    """Creates and saves training sequences from the parsed notes."""
    pitchnames = sorted(list(set(notes)))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    
    with open(PROCESSED_DATA_PATH / 'vocab_map.json', 'w') as f:
        json.dump(note_to_int, f)

    network_input, network_output = [], []
    for i in tqdm(range(len(notes) - sequence_length), desc="Creating sequences"):
        # --- This is the NEW, CORRECTED code ---
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + 1 : i + sequence_length + 1] # Get the "next note" for each note in the input
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append([note_to_int[char] for char in sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    network_output = to_categorical(network_output, num_classes=len(pitchnames))
    
    print("\n💾 Saving final processed data to disk...")
    np.save(PROCESSED_DATA_PATH / 'sequences.npy', network_input)
    np.save(PROCESSED_DATA_PATH / 'targets.npy', network_output)
    print("✅ All data processing is complete!")


if __name__ == '__main__':
    # The function now takes the audio and MIDI paths as arguments
    parsed = parse_genres_dataset(AUDIO_DATASET_PATH, TRANSCRIBED_MIDI_PATH)
    if parsed:
        prepare_sequences_for_training(parsed, SEQUENCE_LENGTH)