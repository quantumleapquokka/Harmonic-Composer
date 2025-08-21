# File: app.py (Final Corrected Version)
# Purpose: A single Flask server that loads the model and handles generation requests.

import numpy as np
import json
import glob
import shutil
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
from music21 import instrument, note, chord, stream
import os, time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from midi_to_musicalxml import render_all_versions

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = Path(__file__).parent / 'processed_data'
MODEL_CHECKPOINT_PATH = Path(__file__).parent / 'model_advanced_weights/weights-best.hdf5'
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# --- MODEL PARAMETERS (Must match the trained model) ---
VOCAB_SIZE = None 
MAX_LEN = 100 
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 512
NUM_TRANSFORMER_BLOCKS = 4

# --- GENERATION PARAMETERS ---
GENERATION_TEMP = 1.0
GENERATION_TOP_P = 0.9

#====================================================================================
# PART 1: MODEL ARCHITECTURE AND GENERATION FUNCTIONS
#====================================================================================

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.self_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.cross_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
    def call(self, inputs, encoder_outputs, training=False):
        self_attn_output = self.self_att(inputs, inputs, use_causal_mask=True)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.layernorm1(inputs + self_attn_output)
        cross_attn_output = self.cross_att(out1, encoder_outputs)
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        out2 = self.layernorm2(out1 + cross_attn_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

def create_conditional_transformer():
    chord_inputs = layers.Input(shape=(None,), name="chord_inputs")
    melody_inputs = layers.Input(shape=(MAX_LEN,), name="melody_inputs")
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)
    encoded_chords = embedding_layer(chord_inputs)
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        encoded_chords = TransformerEncoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(encoded_chords)
    decoded_melody = embedding_layer(melody_inputs)
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        decoded_melody = TransformerDecoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(decoded_melody, encoded_chords)
    outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(decoded_melody)
    model = Model(inputs=[chord_inputs, melody_inputs], outputs=outputs)
    return model

def top_p_sampling(logits, p=0.9, temp=1.0):
    logits = logits / temp
    sorted_indices = tf.argsort(logits, direction="DESCENDING")
    sorted_logits = tf.gather(logits, sorted_indices, batch_dims=1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices_to_remove = cumulative_probs > p
    indices_to_remove = tf.roll(indices_to_remove, 1, axis=-1)
    indices_to_remove = tf.concat([tf.zeros_like(indices_to_remove[:,:1]), indices_to_remove[:,1:]], axis=-1)
    logits_for_sampling = tf.where(indices_to_remove, -float('inf'), sorted_logits)
    sampled_indices = tf.random.categorical(logits_for_sampling, 1)
    final_index = tf.gather(sorted_indices, sampled_indices, batch_dims=1)
    return final_index[0, 0]

def generate_music_with_chords(model, vocab_map, int_to_note, chord_progression, start_token="<classical>", max_tokens=500):
    chord_tokens = np.array([[vocab_map[c] for c in chord_progression if c in vocab_map]])
    prompt_tokens = [vocab_map.get(start_token, 0)]
    for _ in range(max_tokens):
        prompt_for_model = prompt_tokens[-MAX_LEN:]
        padded_prompt = tf.keras.preprocessing.sequence.pad_sequences([prompt_for_model], maxlen=MAX_LEN, padding='post')
        logits = model.predict([chord_tokens, padded_prompt], verbose=0)[0]
        next_token_logits = logits[len(prompt_for_model) - 1]
        next_token = top_p_sampling(tf.expand_dims(next_token_logits, 0), p=GENERATION_TOP_P, temp=GENERATION_TEMP)
        prompt_tokens.append(int(next_token))
    generated_notes = [int_to_note[i] for i in prompt_tokens]
    return generated_notes

def save_as_midi(note_sequence, output_path):
    output_notes = []
    offset = 0
    for item in note_sequence:
        if item.startswith('<') and item.endswith('>'): continue
        if ('.' in item) or item.isdigit():
            notes_in_chord = item.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(item)
            new_note.offset = offset
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=str(output_path))


#====================================================================================
# PART 2: FLASK SERVER SETUP AND GLOBAL MODEL LOADING
#====================================================================================

app = Flask(__name__)
CORS(app)

# --- Global variables to hold the loaded model and vocabulary ---
loaded_model = None
vocab_map = None
int_to_note = None

def load_model_and_vocab():
    """Load the AI model and vocabulary into memory."""
    global loaded_model, vocab_map, int_to_note, VOCAB_SIZE
    
    print("Loading vocabulary...")
    with open(PROCESSED_DATA_PATH / 'vocab_map.json', 'r') as f:
        vocab_map = json.load(f)
    int_to_note = {i: n for n, i in vocab_map.items()}
    VOCAB_SIZE = len(vocab_map)

    print("Rebuilding model architecture...")
    loaded_model = create_conditional_transformer()

    print(f"Loading trained weights from {MODEL_CHECKPOINT_PATH}...")
    loaded_model.load_weights(str(MODEL_CHECKPOINT_PATH))
    print("✅ Model loaded successfully and is ready to generate music!")

@app.route('/generate', methods=['POST'])
def generate():
    # data = request.get_json()
    # genre = data['genre']
    # chords = data['chords']
    # genre_token = f"<{genre}>"
    # user_chords = [genre_token] + chords
    # print(f"Generating music for chord progression: {user_chords}...")
    # music_sequence = generate_music_with_chords(loaded_model, vocab_map, int_to_note, chord_progression=user_chords, start_token=genre_token)
    # timestamp = int(time.time())
    # output_filename = f"composition_{genre}_{timestamp}.mid"
    # output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
    # save_as_midi(music_sequence, output_path)
    # print(f"Music saved to {output_path}")
    # download_url = f"/download/{output_filename}"
    # return jsonify({'message': 'Music generated successfully!', 'download_url': download_url})
    data = request.get_json()
    genre = data['genre']
    chords = data['chords']
    instrument_choice = data.get('instrument', 'piano')  # piano or guitar

    genre_token = f"<{genre}>"
    user_chords = [genre_token] + chords
    print(f"Generating music for chord progression: {user_chords}...")

    # Step 1: Generate base MIDI
    music_sequence = generate_music_with_chords(
        loaded_model, vocab_map, int_to_note,
        chord_progression=user_chords, start_token=genre_token
    )
    timestamp = int(time.time())
    base_filename = f"composition_{genre}_{timestamp}"
    midi_filename = f"{base_filename}.mid"
    midi_path = os.path.join(OUTPUT_DIRECTORY, midi_filename)
    save_as_midi(music_sequence, midi_path)
    print(f"Music saved to {midi_path}")

    # Step 2: Convert to MusicXML + MIDI for both piano & guitar
    render_all_versions(midi_path, out_dir=OUTPUT_DIRECTORY, basename=base_filename)
    import shutil

    # copy the chosen XML to frontend/public/latest.xml
    frontend_latest_path = os.path.join('../frontend/public', f'latest_{instrument_choice}.musicxml')
    shutil.copy(os.path.join(OUTPUT_DIRECTORY, xml_file), frontend_latest_path)

    # Step 3: Choose instrument-specific files to return
    if instrument_choice == "guitar":
        xml_file = f"{base_filename}_guitar.musicxml"
        midi_file = f"{base_filename}_guitar.mid"
    else:  # default to piano
        xml_file = f"{base_filename}_piano.musicxml"
        midi_file = f"{base_filename}_piano.mid"

    xml_url = f"/download/{xml_file}"
    midi_url = f"/download/{midi_file}"

    return jsonify({
        'message': 'Music generated successfully!',
        'xml_url': xml_url,
        'midi_url': midi_url
    })

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(OUTPUT_DIRECTORY, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found on server.'}), 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500

# reads most recent XML file in output folder
@app.route('/latest-xml/<instrument>')
def latest_xml(instrument='piano'):
    xml_files = sorted(
        glob.glob(os.path.join(OUTPUT_DIRECTORY, f"*_{instrument}.musicxml")),
        key=os.path.getmtime,
        reverse=True
    )
    if not xml_files:
        return jsonify({'error': 'No compositions found.'}), 404

    latest_file = xml_files[0]
    with open(latest_file, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    return xml_content

# Load the model and vocab ONCE when the server process starts.
load_model_and_vocab()
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

if __name__ == '__main__':
    # The app.run call is all that's needed here when running directly
    app.run(debug=True)