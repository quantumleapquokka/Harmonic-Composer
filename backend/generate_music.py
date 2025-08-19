# File: generate_music.py (Corrected Version)
# Purpose: Loads a pre-trained Transformer model and uses it to generate music.

import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path
from music21 import instrument, note, chord, stream

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = Path(__file__).parent / 'processed_data'
MODEL_CHECKPOINT_PATH = Path(__file__).parent / 'model_advanced_weights/weights-best.hdf5'
OUTPUT_MIDI_PATH = Path(__file__).parent / 'output/new_composition.mid'

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
# PART 1: REBUILD THE MODEL ARCHITECTURE (No changes here)
#====================================================================================

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

class TransformerEncoderBlock(layers.Layer):
    """A single block of the Transformer Encoder."""
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
    """A single block of the Transformer Decoder with cross-attention."""
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
    """Builds the full Encoder-Decoder Transformer model."""
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

#====================================================================================
# PART 2: MUSIC GENERATION AND SAVING FUNCTIONS (with corrected loop)
#====================================================================================

def top_p_sampling(logits, p=0.9, temp=1.0):
    """A robust implementation of top-p (nucleus) sampling."""
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
    """Generates music conditioned on a specific chord progression."""
    print(f"🎼 Generating music for chord progression: {chord_progression}...")
    chord_tokens = np.array([[vocab_map[c] for c in chord_progression if c in vocab_map]])
    prompt_tokens = [vocab_map.get(start_token, 0)]
    
    # --- THIS IS THE CORRECTED GENERATION LOOP ---
    num_tokens_generated = 0
    while num_tokens_generated < max_tokens:
        # 1. Truncate the prompt to the model's max length
        prompt_for_model = prompt_tokens[-MAX_LEN:]
        
        # 2. Pad the sequence
        padded_prompt = tf.keras.preprocessing.sequence.pad_sequences([prompt_for_model], maxlen=MAX_LEN, padding='post')
        
        # 3. Get the model's predictions
        logits = model.predict([chord_tokens, padded_prompt], verbose=0)[0]
        
        # 4. We only care about the prediction for the very last token in our input sequence
        next_token_logits = logits[len(prompt_for_model) - 1]
        
        # 5. Sample the next token
        next_token = top_p_sampling(tf.expand_dims(next_token_logits, 0), p=GENERATION_TOP_P, temp=GENERATION_TEMP)
        
        # 6. Add the new token to our full sequence and continue
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

#====================================================================================
# PART 3: MAIN EXECUTION BLOCK (No changes here)
#====================================================================================
if __name__ == '__main__':
    print("Loading vocabulary...")
    with open(PROCESSED_DATA_PATH / 'vocab_map.json', 'r') as f:
        vocab_map = json.load(f)
    int_to_note = {i: n for n, i in vocab_map.items()}
    VOCAB_SIZE = len(vocab_map)

    print("Rebuilding model architecture...")
    model = create_conditional_transformer()

    print(f"Loading trained weights from {MODEL_CHECKPOINT_PATH}...")
    model.load_weights(str(MODEL_CHECKPOINT_PATH))

    # Define your chord progression and generate music!
    user_chords = ["<jazz>", "Cminor", "Fmajor", "A#major", "D#major"] 
    music = generate_music_with_chords(model, vocab_map, int_to_note, chord_progression=user_chords)
    
    if music:
        save_as_midi(music, OUTPUT_MIDI_PATH)