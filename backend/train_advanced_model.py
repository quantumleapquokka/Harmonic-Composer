# File 2: train_advanced_model.py (Modified for Genre Folders)
# Purpose: Builds, trains, and uses a state-of-the-art Transformer model for music generation.

import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
from music21 import instrument, note, chord, stream

# --- Configuration ---
PROCESSED_DATA_PATH = Path(__file__).parent / 'processed_data'
MODEL_CHECKPOINT_PATH = Path(__file__).parent / 'model_advanced_weights/weights-best.hdf5'
OUTPUT_MIDI_PATH = Path(__file__).parent / 'output/ai_transformer_composition.mid'

# --- Transformer Model Parameters ---
VOCAB_SIZE = None # Determined from data
MAX_LEN = 100 # Must match SEQUENCE_LENGTH in data processing
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 512
NUM_TRANSFORMER_BLOCKS = 6

# --- Generation Parameters ---
GENERATION_TEMP = 1.0 # Higher values mean more random/creative
GENERATION_TOP_P = 0.9 # Nucleus sampling probability

class TransformerBlock(layers.Layer):
    """A single block of the Transformer Decoder architecture."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, use_causal_mask=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    """Combines token embedding with positional embedding."""
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

def create_transformer_model():
    """Builds the full Transformer model."""
    inputs = layers.Input(shape=(MAX_LEN,))
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)
    x = embedding_layer(inputs)
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(x)
    outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.summary()
    return model

def top_p_sampling(logits, p=0.9, temp=1.0):
    """Performs nucleus sampling on the model's output logits."""
    logits = logits / temp
    sorted_logits = tf.sort(logits, direction="DESCENDING")
    sorted_indices = tf.argsort(logits, direction="DESCENDING")
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1)
    sorted_indices_to_remove = tf.concat([tf.zeros_like(sorted_indices_to_remove[:,:1]), sorted_indices_to_remove[:,1:]], axis=-1)
    indices_to_remove = tf.scatter_nd(tf.expand_dims(sorted_indices, axis=-1), tf.cast(sorted_indices_to_remove, dtype=tf.int32), logits.shape)
    logits = tf.where(indices_to_remove, -np.inf, logits)
    return tf.random.categorical(logits, 1)[0, 0]

def generate_music(model, vocab_map, int_to_note, start_prompt="<classical>", max_tokens=500):
    """Generates music using the trained model with Top-p sampling."""
    print(f"🎼 Generating music with starting prompt: {start_prompt}...")
    
    if start_prompt not in vocab_map:
        print(f"Error: Start prompt '{start_prompt}' not in vocabulary. Please choose from available genres.")
        return []

    num_tokens_generated = 0
    prompt_tokens = [vocab_map[start_prompt]]
    
    while num_tokens_generated < max_tokens:
        pad_len = MAX_LEN - len(prompt_tokens)
        sample_index = len(prompt_tokens) - 1
        if pad_len < 0:
            prompt_tokens = prompt_tokens[-MAX_LEN:]
            pad_len = 0
            sample_index = -1
        padded_prompt = tf.keras.preprocessing.sequence.pad_sequences([prompt_tokens], maxlen=MAX_LEN, padding='post')
        logits = model.predict(padded_prompt, verbose=0)[0]
        next_token_logits = logits[sample_index]
        next_token = top_p_sampling(tf.expand_dims(next_token_logits, 0), p=GENERATION_TOP_P, temp=GENERATION_TEMP)
        prompt_tokens.append(int(next_token))
        num_tokens_generated += 1
    
    generated_notes = [int_to_note[i] for i in prompt_tokens]
    return generated_notes

def save_as_midi(note_sequence, output_path):
    """Saves a sequence of notes as a MIDI file."""
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_stream.write('midi', fp=str(output_path))
    print(f"✅ Composition complete! Saved to '{output_path}'")


if __name__ == '__main__':
    X = np.load(PROCESSED_DATA_PATH / 'sequences.npy')
    y = np.load(PROCESSED_DATA_PATH / 'targets.npy')
    with open(PROCESSED_DATA_PATH / 'vocab_map.json', 'r') as f:
        vocab_map = json.load(f)
    int_to_note = {i: n for n, i in vocab_map.items()}
    VOCAB_SIZE = len(vocab_map)

    transformer = create_transformer_model()

    if MODEL_CHECKPOINT_PATH.exists():
        print(f"Found existing weights. Loading from {MODEL_CHECKPOINT_PATH}...")
        transformer.load_weights(str(MODEL_CHECKPOINT_PATH))
    else:
        print("No weights found. Starting new training session...")
        MODEL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(filepath=str(MODEL_CHECKPOINT_PATH), save_weights_only=True, save_best_only=True, monitor='loss', verbose=1)
        transformer.fit(X, y, epochs=100, batch_size=64, callbacks=[checkpoint])

    # === CHOOSE YOUR GENRE HERE! ===
    # Change this to '<jazz>', '<pop>', '<country>', etc.
    music_prompt = "<jazz>" 
    
    music = generate_music(transformer, vocab_map, int_to_note, start_prompt=music_prompt)
    if music:
        save_as_midi(music, OUTPUT_MIDI_PATH)