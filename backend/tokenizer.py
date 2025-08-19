"""
Lightweight tokenizer for symbolic-music tokens (notes/chords/genre tags).
- Deterministic vocab building
- Robust save/load
- Encode/Decode helpers
- Padding utilities
- Mild token normalization (unicode accidentals, whitespace, case)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Sequence, Optional
from pathlib import Path
import json
import math
import re
from collections import Counter

# -------------------------
# Special tokens (canonical)
# -------------------------
PAD = "<pad>"
BOS = "<bos>"     # beginning of sequence
EOS = "<eos>"     # end of sequence
UNK = "<unk>"     # unknown / out-of-vocab
REST = "<rest>"   # musical rest (if you represent rests as tokens)
SEP = "<sep>"     # optional separator for segments/spans
MASK = "<mask>"   # optional for future masked training

SPECIAL_TOKENS: Tuple[str, ...] = (PAD, BOS, EOS, UNK, REST, SEP, MASK)

# Regex for light token cleanup (collapse inner whitespace)
_ws_re = re.compile(r"\s+")

# -------------------------
# Dataclass: Vocab container
# -------------------------
@dataclass(frozen=True)
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def get(self, token: str, default: Optional[int] = None) -> Optional[int]:
        return self.token_to_id.get(token, default)

    def token(self, idx: int, default: Optional[str] = None) -> Optional[str]:
        return self.id_to_token.get(idx, default)


# -------------------------
# Normalization helpers
# -------------------------
def normalize_token(tok: str) -> str:
    """
    Normalize a symbolic token (note/chord/tag):
    - strip & collapse whitespace
    - unify unicode ♯♭ to #/b
    - upper-case leading pitch class (C, D#, Fm7, <GENRE> stays as-is)
    - keep angle-bracket tags (<classical>) untouched
    """
    if tok is None:
        return ""

    # Trim and collapse internal whitespace
    t = _ws_re.sub(" ", str(tok).strip())

    # If this looks like a tag token, return as-is
    if t.startswith("<") and t.endswith(">"):
        return t

    # Replace common unicode accidentals
    t = (t.replace("♯", "#").replace("♭", "b").replace("♮", ""))
    # Uniform hyphen
    t = t.replace("–", "-").replace("—", "-")

    # Heuristic uppercasing for pitch-class start
    # e.g., "c#m7" -> "C#m7", "f min" -> "F min"
    if t:
        # Uppercase first char if alphabetical (avoid uppercasing <bos>, etc.)
        if t[0].isalpha():
            t = t[0].upper() + t[1:]

    return t


# -------------------------
# Vocab building / IO
# -------------------------
def build_vocab(
    corpora: Iterable[Iterable[str]],
    min_freq: int = 1,
    add_specials: bool = True,
    extra_tokens: Optional[Iterable[str]] = None,
) -> Vocab:
    """
    Build a deterministic vocab from one or more token sequences.
    - min_freq: keep tokens with frequency >= min_freq
    - add_specials: include SPECIAL_TOKENS in the front
    - extra_tokens: optional set/list of known tokens to force-include

    Ordering:
      1) specials (fixed order)
      2) by decreasing frequency
      3) ties broken alphabetically
    """
    counter: Counter = Counter()

    # Count normalized tokens
    for seq in corpora:
        for tok in seq:
            t = normalize_token(tok)
            if t:
                counter[t] += 1

    # Apply min_freq
    items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]

    # Force-include extra tokens with freq = +inf to pin them near front (after specials)
    if extra_tokens:
        for tok in extra_tokens:
            t = normalize_token(tok)
            if t and t not in counter:
                items.append((t, math.inf))

    # Sort deterministically: freq desc, then token asc
    items.sort(key=lambda x: (-x[1], x[0]))

    token_to_id: Dict[str, int] = {}
    id_to_token: Dict[int, str] = {}

    next_id = 0
    if add_specials:
        for sp in SPECIAL_TOKENS:
            token_to_id[sp] = next_id
            id_to_token[next_id] = sp
            next_id += 1

    for tok, _freq in items:
        # Skip if already present (dup via extra_tokens)
        if tok in token_to_id:
            continue
        token_to_id[tok] = next_id
        id_to_token[next_id] = tok
        next_id += 1

    return Vocab(token_to_id, id_to_token)


def save_vocab(path: Path | str, vocab: Vocab) -> None:
    """
    Save vocab to JSON:
      {
        "token_to_id": { "<pad>":0, "C": 10, ... }
      }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"token_to_id": vocab.token_to_id}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_vocab(path: Path | str) -> Vocab:
    """
    Load vocab from JSON file produced by save_vocab.
    Rebuild id_to_token for safety.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    token_to_id = {str(k): int(v) for k, v in data["token_to_id"].items()}
    id_to_token = {v: k for k, v in token_to_id.items()}
    return Vocab(token_to_id, id_to_token)


# -------------------------
# Encode / Decode
# -------------------------
def encode(
    tokens: Sequence[str],
    vocab: Vocab,
    add_bos: bool = False,
    add_eos: bool = False,
    use_unk: bool = True,
) -> List[int]:
    """
    Encode a list of string tokens into ids.
    Unknown tokens map to <unk> if available and use_unk=True, else raise KeyError.
    """
    ids: List[int] = []
    if add_bos and BOS in vocab.token_to_id:
        ids.append(vocab.token_to_id[BOS])

    for tok in tokens:
        t = normalize_token(tok)
        if t in vocab.token_to_id:
            ids.append(vocab.token_to_id[t])
        else:
            if use_unk and UNK in vocab.token_to_id:
                ids.append(vocab.token_to_id[UNK])
            else:
                raise KeyError(f"Token not in vocab and use_unk=False: {tok!r}")

    if add_eos and EOS in vocab.token_to_id:
        ids.append(vocab.token_to_id[EOS])

    return ids


def decode(
    ids: Sequence[int],
    vocab: Vocab,
    skip_specials: bool = False,
) -> List[str]:
    """
    Decode a list of ids back into tokens.
    If skip_specials=True, drop PAD/BOS/EOS/UNK/MASK/SEP/REST.
    """
    out: List[str] = []
    specials = set(SPECIAL_TOKENS) if skip_specials else set()
    for i in ids:
        tok = vocab.id_to_token.get(int(i), UNK)
        if skip_specials and tok in specials:
            continue
        out.append(tok)
    return out


# -------------------------
# Padding utilities
# -------------------------
def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_id: int,
    max_len: Optional[int] = None,
    pad: str = "post",   # "pre" | "post"
    trunc: str = "post", # "pre" | "post"
) -> Tuple[List[List[int]], List[int]]:
    """
    Pad (and optionally truncate) a batch of sequences to the same length.
    Returns:
      padded: List[List[int]] with shape [B, T]
      lengths: original lengths (clipped to max_len if truncated)
    """
    assert pad in ("pre", "post")
    assert trunc in ("pre", "post")

    lengths = [len(s) for s in sequences]
    if max_len is None:
        max_len = max(lengths) if lengths else 0

    padded: List[List[int]] = []
    for seq in sequences:
        s = list(seq)
        # Truncate
        if len(s) > max_len:
            s = s[-max_len:] if trunc == "pre" else s[:max_len]
        # Pad
        pad_amt = max_len - len(s)
        if pad_amt > 0:
            pad_vec = [pad_id] * pad_amt
            s = (pad_vec + s) if pad == "pre" else (s + pad_vec)
        padded.append(s)

    # Clip reported lengths when truncation occurs
    clipped_lengths = [min(l, max_len) for l in lengths]
    return padded, clipped_lengths


# -------------------------
# Convenience helpers
# -------------------------
def ensure_special_tokens(vocab: Vocab) -> Vocab:
    """
    Ensure SPECIAL_TOKENS exist; if any are missing, append them with new ids.
    Returns a (possibly) new Vocab (immutability of dataclass).
    """
    if all(sp in vocab.token_to_id for sp in SPECIAL_TOKENS):
        return vocab

    token_to_id = dict(vocab.token_to_id)
    id_to_token = dict(vocab.id_to_token)
    next_id = max(id_to_token.keys(), default=-1) + 1
    changed = False
    for sp in SPECIAL_TOKENS:
        if sp not in token_to_id:
            token_to_id[sp] = next_id
            id_to_token[next_id] = sp
            next_id += 1
            changed = True
    return Vocab(token_to_id, id_to_token) if changed else vocab


def make_int_maps(vocab: Vocab) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Alias generator for (token_to_id, id_to_token).
    Useful when other modules expect separate dicts.
    """
    return vocab.token_to_id, vocab.id_to_token


def corpus_from_files(paths: Iterable[Path | str]) -> Iterable[List[str]]:
    """
    Utility to load line-delimited token files where each line is a whitespace-separated sequence.
    Example line:
        <classical> C4 E4 G4 <sep> Am Dm G C <eos>
    """
    for p in paths:
        text = Path(p).read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            yield [normalize_token(tok) for tok in line.split() if tok.strip()]
