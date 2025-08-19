# File 2: train_advanced_model.py (Upgraded with Chord Conditioning)
# Purpose: Builds and trains an Encoder-Decoder Transformer for chord-conditioned music generation.

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
VOCAB_SIZE = None 
MAX_LEN = 100
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 512
NUM_TRANSFORMER_BLOCKS = 4 # Adjusted for a deeper Encoder/Decoder model

# --- Generation Parameters ---
GENERATION_TEMP = 1.0
GENERATION_TOP_P = 0.9

#====================================================================================
# MODIFICATION 1: Splitting the Transformer into Encoder and Decoder Blocks
# The Encoder processes the conditioning signal (chords).
# The Decoder generates the output (melody) while paying attention to the Encoder.
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
        # This is the new layer that connects the decoder to the encoder (melody to chords)
        self.cross_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, inputs, encoder_outputs, training=False):
        # Causal self-attention for the melody-so-far
        self_attn_output = self.self_att(inputs, inputs, use_causal_mask=True)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.layernorm1(inputs + self_attn_output)

        # Cross-attention: melody "looks at" the chord progression
        cross_attn_output = self.cross_att(out1, encoder_outputs)
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        out2 = self.layernorm2(out1 + cross_attn_output)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

def create_conditional_transformer():
    """Builds the full Encoder-Decoder Transformer model."""
    # Two inputs: one for chords (encoder), one for melody (decoder)
    chord_inputs = layers.Input(shape=(None,), name="chord_inputs")
    melody_inputs = layers.Input(shape=(MAX_LEN,), name="melody_inputs")

    # Shared embedding layer for a common musical vocabulary
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)
    
    # Encoder branch for processing chords
    encoded_chords = embedding_layer(chord_inputs)
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        encoded_chords = TransformerEncoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(encoded_chords)

    # Decoder branch for generating melody
    decoded_melody = embedding_layer(melody_inputs)
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        decoded_melody = TransformerDecoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(decoded_melody, encoded_chords)

    outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(decoded_melody)
    model = Model(inputs=[chord_inputs, melody_inputs], outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.summary()
    return model

#====================================================================================
# MODIFICATION 2: Updating the Generation Function to Use Chord Conditioning
#====================================================================================
def generate_music_with_chords(model, vocab_map, int_to_note, chord_progression, start_token="<classical>", max_tokens=500):
    """Generates music conditioned on a specific chord progression."""
    print(f"🎼 Generating music for chord progression: {chord_progression}...")
    
    # Convert chord progression and start token to integer tokens
    chord_tokens = np.array([[vocab_map[c] for c in chord_progression]])
    prompt_tokens = [vocab_map[start_token]]
    
    num_tokens_generated = 0
    while num_tokens_generated < max_tokens:
        pad_len = MAX_LEN - len(prompt_tokens)
        sample_index = len(prompt_tokens) - 1
        if pad_len < 0:
            prompt_tokens = prompt_tokens[-MAX_LEN:]
            pad_len = 0
            sample_index = -1
            
        padded_prompt = tf.keras.preprocessing.sequence.pad_sequences([prompt_tokens], maxlen=MAX_LEN, padding='post')
        
        # Model now takes two inputs
        logits = model.predict([chord_tokens, padded_prompt], verbose=0)[0]
        next_token_logits = logits[sample_index]
        
        # Using top_p_sampling from your original file
        next_token = top_p_sampling(tf.expand_dims(next_token_logits, 0), p=GENERATION_TOP_P, temp=GENERATION_TEMP)
        prompt_tokens.append(int(next_token))
        num_tokens_generated += 1
    
    generated_notes = [int_to_note[i] for i in prompt_tokens]
    return generated_notes


# Find this block at the end of train_advanced_model.py
if __name__ == '__main__':
    # --- UNCOMMENT THE DATA LOADING LINES ---
    X_chords = np.load(PROCESSED_DATA_PATH / 'X_chords.npy')
    X_melody = np.load(PROCESSED_DATA_PATH / 'X_melody.npy')
    y = np.load(PROCESSED_DATA_PATH / 'y_melody.npy')

    with open(PROCESSED_DATA_PATH / 'vocab_map.json', 'r') as f:
        vocab_map = json.load(f)
    int_to_note = {i: n for n, i in vocab_map.items()}
    VOCAB_SIZE = len(vocab_map)

    transformer = create_conditional_transformer()

    # --- UNCOMMENT THE TRAINING CALL ---
    checkpoint = ModelCheckpoint(filepath=str(MODEL_CHECKPOINT_PATH), save_weights_only=True, save_best_only=True, monitor='loss', verbose=1)
    transformer.fit([X_chords, X_melody], y, epochs=10, batch_size=32, callbacks=[checkpoint])

    # --- UNCOMMENT THE GENERATION CALL (after training) ---
    user_chords = ["<classical>", "Cmajor", "Gmajor", "Aminor", "Fmajor"] 
    music = generate_music_with_chords(transformer, vocab_map, int_to_note, chord_progression=user_chords)
    if music:
        save_as_midi(music, OUTPUT_MIDI_PATH)