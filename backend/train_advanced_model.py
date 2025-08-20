# File: backend/train_advanced_model.py
# Purpose: Builds and trains an Encoder-Decoder Transformer for chord-conditioned music generation.

import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
from music21 import instrument, note, chord, stream

# ADDED: import sampling + MIDI saver from your generator (so NameError won't occur)
from generate_music import top_p_sampling, save_as_midi  # CHANGED/ADDED

# ADDED: correct module name and the specific helpers we actually use
from chords_tools import normalize_chord, progression_to_grid  # CHANGED/ADDED

# --- Configuration ---
PROCESSED_DATA_PATH = Path(__file__).parent / 'processed_data'
MODEL_CHECKPOINT_PATH = Path(__file__).parent / 'model_advanced_weights' / 'weights-best.hdf5'
OUTPUT_MIDI_PATH = Path(__file__).parent.parent / 'output' / 'ai_transformer_composition.mid'

# --- Transformer Model Parameters ---
VOCAB_SIZE = None            # will be set after we load vocab_map.json
MAX_LEN = 100
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 512
NUM_TRANSFORMER_BLOCKS = 4   # deeper encoder/decoder

# --- Generation Parameters ---
GENERATION_TEMP = 1.0
GENERATION_TOP_P = 0.9

#====================================================================================
# Encoder/Decoder building blocks
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
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
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
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, inputs, encoder_outputs, training=False):
        # causal self-attention for melody-so-far
        self_attn_output = self.self_att(inputs, inputs, use_causal_mask=True)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.layernorm1(inputs + self_attn_output)

        # cross-attention: melody looks at chords
        cross_attn_output = self.cross_att(out1, encoder_outputs)
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        out2 = self.layernorm2(out1 + cross_attn_output)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

def create_conditional_transformer():
    """Build the full Encoder-Decoder Transformer model."""
    # Two inputs: chords (encoder) + melody (decoder)
    chord_inputs = layers.Input(shape=(None,), name="chord_inputs")
    melody_inputs = layers.Input(shape=(MAX_LEN,), name="melody_inputs")

    # Shared embedding over same vocab
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)

    # Encoder
    encoded_chords = embedding_layer(chord_inputs)
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        encoded_chords = TransformerEncoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(encoded_chords)

    # Decoder
    decoded_melody = embedding_layer(melody_inputs)
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        decoded_melody = TransformerDecoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(decoded_melody, encoded_chords)

    outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(decoded_melody)
    model = Model(inputs=[chord_inputs, melody_inputs], outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.summary()
    return model

#====================================================================================
# Generation helper that uses chord conditioning
#====================================================================================
def generate_music_with_chords(model, vocab_map, int_to_note, chord_progression,
                               start_token="<classical>", max_tokens=500):
    """Generates music conditioned on a specific chord progression."""
    print(f"🎼 Generating music for chord progression: {chord_progression}...")

    # CHANGED: normalize + grid chords so they match your vocab (Cmaj/Amin etc.)
    norm = [normalize_chord(c) for c in chord_progression]  # ADDED
    grid = progression_to_grid(norm, bars=25, beats_per_bar=4, chords_per_bar=1)[:MAX_LEN]  # ADDED

    # CHANGED: map with safe fallback to <unk> so we never KeyError
    unk = vocab_map.get("<unk>", 0)  # ADDED
    chord_tokens = np.array([[vocab_map.get(c, unk) for c in grid]])  # CHANGED

    # Prompt init
    if start_token not in vocab_map:
        # if user passes a tag not in vocab, fall back to first token id (usually <bos> or similar)
        start_id = next(iter(vocab_map.values()))
    else:
        start_id = vocab_map[start_token]

    prompt_tokens = [start_id]

    num_tokens_generated = 0
    while num_tokens_generated < max_tokens:
        pad_len = MAX_LEN - len(prompt_tokens)
        sample_index = len(prompt_tokens) - 1
        if pad_len < 0:
            prompt_tokens = prompt_tokens[-MAX_LEN:]
            pad_len = 0
            sample_index = -1

        padded_prompt = tf.keras.preprocessing.sequence.pad_sequences(
            [prompt_tokens], maxlen=MAX_LEN, padding='post'
        )

        # model takes [X_chords, X_melody]
        logits = model.predict([chord_tokens, padded_prompt], verbose=0)[0]
        next_token_logits = logits[sample_index]

        # top-p sampling (already imported)
        next_token = top_p_sampling(tf.expand_dims(next_token_logits, 0),
                                    p=GENERATION_TOP_P, temp=GENERATION_TEMP)
        prompt_tokens.append(int(next_token))
        num_tokens_generated += 1

    generated_notes = [int_to_note[i] for i in prompt_tokens if i in int_to_note]
    return generated_notes


#====================================================================================
# Training script
#====================================================================================
if __name__ == '__main__':
    # Load arrays produced by process_data.py
    X_melody = np.load(PROCESSED_DATA_PATH / "sequences.npy")   # (N, T)
    y        = np.load(PROCESSED_DATA_PATH / "targets.npy")     # (N, T, V)
    X_chords = np.load(PROCESSED_DATA_PATH / "X_chords.npy")    # (N, T)

    # CHANGED: read tokenizer-format vocab ({"token_to_id": {...}})
    data = json.load(open(PROCESSED_DATA_PATH / 'vocab_map.json', 'r', encoding='utf-8'))  # CHANGED
    vocab_map = data["token_to_id"]                                                                  # CHANGED
    int_to_note = {i: n for n, i in vocab_map.items()}
    VOCAB_SIZE = len(vocab_map)                                                                      # CHANGED

    # Build model with correct vocab size
    transformer = create_conditional_transformer()

    # Train
    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_CHECKPOINT_PATH),
        save_weights_only=True,
        save_best_only=True,
        monitor='loss',
        verbose=1
    )
    transformer.fit([X_chords, X_melody], y, epochs=10, batch_size=32, callbacks=[checkpoint])

    # Quick demo generation after training
    # CHANGED: these can be any user chords; normalizer will map "Cmajor" -> "Cmaj" etc.
    user_chords = ["<classical>", "Cmajor", "Gmajor", "Aminor", "Fmajor"]  # will be normalized
    music = generate_music_with_chords(transformer, vocab_map, int_to_note, chord_progression=user_chords)
    if music:
        save_as_midi(music, OUTPUT_MIDI_PATH)

