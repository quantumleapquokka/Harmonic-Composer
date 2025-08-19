# backend/chord_tools.py
"""
Chord utilities for symbolic music backends (no external deps).
Safe to add now; integrate later by importing the functions you need.

Features
--------
- normalize_chord: canonicalize chord spelling & aliases (e.g., Cmaj7, F#m7, Bb7/F)
- parse_chord: break into root, quality, extensions/alterations, bass (slash) note
- transpose_chord / transpose_progression: move chords by semitones
- chord_to_pitches: quick MIDI pitch spelling for triads/7ths (for simple accompaniment)
- progression utilities:
    - fill_or_trim_progression: repeat/trim to target length
    - progression_to_grid: map a list of chords to a fixed step grid (e.g., bars*beats)
- roman numeral conversions (basic diatonic set): roman_to_chord, chord_to_roman
- detect_key_from_progression (very rough heuristic)

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable
import re

# ----------------------------
# Pitch class maps (12-TET)
# ----------------------------
_PITCHES_SHARP = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
_ENHARMONIC_FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B", "Fb": "E",
}
# Allow unicode accidentals
_ACC_REPL = {
    "♯": "#",
    "♭": "b",
    "♮": "",
    "–": "-",
    "—": "-",
}

# ----------------------------
# Quality aliases & patterns
# ----------------------------
_QUAL_ALIASES = {
    "": "maj", "M": "maj", "maj": "maj", "Δ": "maj", "MAJ": "maj",
    "m": "min", "min": "min", "-": "min",
    "dim": "dim", "o": "dim",
    "aug": "aug", "+": "aug",
    "sus": "sus", "sus2": "sus2", "sus4": "sus4",
    "7": "7",
    "maj7": "maj7", "M7": "maj7", "Δ7": "maj7",
    "min7": "min7", "m7": "min7",
    "ø": "m7b5", "m7b5": "m7b5",
    "o7": "dim7", "dim7": "dim7",
}

# triad intervals from root (in semitones)
_TRIADS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "sus": [0, 5, 7],
}
# seventh extensions
_SEVENTH_TOP = {
    "7": 10,      # b7
    "maj7": 11,   # M7
    "min7": 10,
    "m7b5": 10,   # half-diminished
    "dim7": 9,
}

# Parsing regex: root ([A-G][#|b]?), optional quality chunk, optional extensions, optional slash bass
# Examples: Cmaj7, F#m7b5, Bb7sus4, Gadd9, Dm7/G
_CHORD_RE = re.compile(
    r"""
    ^\s*
    (?P<root>[A-Ga-g][#b♯♭]?)           # Root
    (?P<qual>(maj7|M7|Δ7|maj|M|Δ|min7|m7|m|dim7|o7|dim|o|aug|\+|sus4|sus2|sus|7|-)?)
    (?P<rest>(?:add\d+|[#b]\d+|\(\s*[#b]?\d+\s*\)|\d+|sus4|sus2|sus|no3|no5|[a-zA-Z0-9#b]+)*)?
    (?:/(?P<bass>[A-Ga-g][#b♯♭]?))?     # optional slash bass
    \s*$
    """,
    re.VERBOSE,
)

@dataclass
class ParsedChord:
    root: str              # e.g., "C#"
    quality: str           # canonical: maj/min/dim/aug/sus2/sus4/sus/7/maj7/min7/m7b5/dim7
    suffix: str            # raw leftover, e.g., "add9", "b9#11"
    bass: Optional[str]    # slash bass like "G#"
    symbol: str            # normalized chord symbol

# ----------------------------
# Helpers
# ----------------------------
def _canon_note_name(n: str) -> str:
    """Canonicalize note name to sharps; keep 'Bb' -> 'A#' etc."""
    if not n:
        return ""
    n = n.strip()
    for k, v in _ACC_REPL.items():
        n = n.replace(k, v)
    # Uppercase pitch letter
    if n[0].isalpha():
        n = n[0].upper() + n[1:]
    # Replace flats with sharps via mapping
    if len(n) >= 2 and n[1] == "b":
        n = _ENHARMONIC_FLAT_TO_SHARP.get(n, n)
    # C, C#, D, etc.
    if n not in _PITCHES_SHARP:
        # Try stripping naturals or extraneous symbols
        n = n.replace("♮", "")
    return n

def _pc_index(note: str) -> int:
    """Pitch-class index 0..11 for canonical sharp name."""
    note = _canon_note_name(note)
    return _PITCHES_SHARP.index(note)

def _transpose_note(note: str, semitones: int) -> str:
    i = _pc_index(note)
    j = (i + semitones) % 12
    return _PITCHES_SHARP[j]

def _canon_quality(q: str) -> str:
    q = q or ""
    q = q.strip()
    for k, v in _ACC_REPL.items():
        q = q.replace(k, v)
    return _QUAL_ALIASES.get(q, q or "maj")

def normalize_chord(symbol: str) -> str:
    """
    Normalize chord symbol to a canonical sharp-based form.
    Examples:
      "bbmaj7" -> "A#maj7"
      "Fmin7"  -> "Fmin7"
      "G-7"    -> "Gmin7"
      "CΔ7"    -> "Cmaj7"
      "Dm7/G"  -> "Dm7/G"
    """
    if not symbol or symbol.strip().upper() in {"N.C.", "NC", "NOCHORD"}:
        return "N.C."
    s = symbol.strip()
    for k, v in _ACC_REPL.items():
        s = s.replace(k, v)

    m = _CHORD_RE.match(s)
    if not m:
        # fallback: try to canonicalize just the root and return as-is
        # This keeps the string but cleans the accidental usage.
        root_guess = _canon_note_name(s.split("/")[0][:2])
        return root_guess if root_guess else s

    root = _canon_note_name(m.group("root"))
    qual = _canon_quality(m.group("qual"))
    rest = (m.group("rest") or "").strip()
    bass = m.group("bass")
    bass = _canon_note_name(bass) if bass else None

    # Convert minor alias "-" to "min"
    if qual == "-":
        qual = "min"

    # Build the canonical symbol (root + quality + rest + /bass)
    # Special case: bare "7" means dominant 7 on top of major triad; keep as "7"
    if qual in {"7", "maj7", "min7", "m7b5", "dim7"}:
        body = f"{root}{qual}"
    elif qual in {"maj", "min", "dim", "aug", "sus", "sus2", "sus4"}:
        body = f"{root}{qual}"
    else:
        # Unknown quality tokens: leave as-is but still attach to root
        body = f"{root}{qual}" if qual else f"{root}maj"

    if rest:
        body += rest

    if bass:
        body += f"/{bass}"

    return body

def parse_chord(symbol: str) -> ParsedChord:
    """Parse and return structured chord info (with a normalized symbol)."""
    norm = normalize_chord(symbol)
    if norm == "N.C.":
        return ParsedChord(root="N.C.", quality="maj", suffix="", bass=None, symbol="N.C.")
    m = _CHORD_RE.match(norm)
    if not m:
        # Try minimal recovery: treat whole as root
        r = _canon_note_name(norm)
        return ParsedChord(root=r or "C", quality="maj", suffix="", bass=None, symbol=r or "Cmaj")

    root = _canon_note_name(m.group("root"))
    qual = _canon_quality(m.group("qual"))
    rest = (m.group("rest") or "").strip()
    bass = m.group("bass")
    bass = _canon_note_name(bass) if bass else None

    return ParsedChord(root=root, quality=qual, suffix=rest, bass=bass, symbol=norm)

# ----------------------------
# Chord to pitches (simple)
# ----------------------------
def chord_to_pitches(symbol: str, octave: int = 4, include_seventh: bool = True) -> List[int]:
    """
    Returns a basic MIDI pitch list representing the chord (root position).
    Useful for simple LH patterns. Does not voice-lead or handle complex alterations.
    """
    p = parse_chord(symbol)
    if p.symbol == "N.C.":
        return []
    root_pc = _pc_index(p.root)
    triad = _TRIADS.get(p.quality, _TRIADS["maj"])

    pcs = [(root_pc + iv) % 12 for iv in triad]

    # Add seventh if quality demands (or include_seventh=True and we have mapping)
    if include_seventh:
        if p.quality in _SEVENTH_TOP:
            pcs.append((root_pc + _SEVENTH_TOP[p.quality]) % 12)
        elif p.quality == "7":  # dominant 7
            pcs.append((root_pc + 10) % 12)

    # Map to MIDI by adding octave
    base = 12 * (octave + 1)  # MIDI C4 = 60; octave=4 -> base 60
    # Choose nearest spelling for each pc in the target octave
    pitches = [base + _PITCHES_SHARP.index(_PITCHES_SHARP[pc]) for pc in pcs]

    # Add slash bass if present (force below the chord)
    if p.bass:
        bass_pc = _pc_index(p.bass)
        bass_pitch = 12 * (octave) + bass_pc  # one octave below base
        pitches = [bass_pitch] + [n for n in pitches if n != bass_pitch]
    return sorted(set(pitches))

# ----------------------------
# Transposition
# ----------------------------
def transpose_chord(symbol: str, semitones: int) -> str:
    p = parse_chord(symbol)
    if p.symbol == "N.C.":
        return "N.C."
    new_root = _transpose_note(p.root, semitones)
    new_bass = _transpose_note(p.bass, semitones) if p.bass else None
    body = f"{new_root}{p.quality}{p.suffix}" if p.quality else f"{new_root}maj{p.suffix}"
    if new_bass:
        body += f"/{new_bass}"
    return normalize_chord(body)

def transpose_progression(chords: Iterable[str], semitones: int) -> List[str]:
    return [transpose_chord(c, semitones) for c in chords]

# ----------------------------
# Progression shaping
# ----------------------------
def fill_or_trim_progression(chords: List[str], target_len: int, fill_token: str = "N.C.") -> List[str]:
    """
    Repeat or trim chords to reach target_len (useful for fixed sequence lengths).
    """
    if target_len <= 0:
        return []
    if not chords:
        return [fill_token] * target_len
    out = []
    i = 0
    while len(out) < target_len:
        out.append(chords[i % len(chords)])
        i += 1
    return out[:target_len]

def progression_to_grid(
    chords: List[str],
    bars: int,
    beats_per_bar: int = 4,
    chords_per_bar: int = 1,
    fill_token: str = "N.C.",
) -> List[str]:
    """
    Expand a chord list to a step grid of length bars*beats_per_bar.
    - chords_per_bar: how many distinct chord slots you want per bar (1, 2, or 4 are common)
    The returned list's length is bars*beats_per_bar; chords are repeated within each slot.
    """
    total_steps = bars * beats_per_bar
    if chords_per_bar not in (1, 2, 4):
        chords_per_bar = 1

    slots_per_bar = chords_per_bar
    steps_per_slot = beats_per_bar // slots_per_bar
    if steps_per_slot == 0:
        steps_per_slot = 1

    # Prepare slots: repeat/trim input chords to bars*slots_per_bar
    needed_slots = bars * slots_per_bar
    slot_chords = fill_or_trim_progression([normalize_chord(c) for c in chords], needed_slots, fill_token)

    # Expand each slot to steps
    out: List[str] = []
    for sc in slot_chords:
        out.extend([sc] * steps_per_slot)

    # Pad/truncate to exact length
    if len(out) < total_steps:
        out += [fill_token] * (total_steps - len(out))
    return out[:total_steps]

# ----------------------------
# Key guessing & roman numerals (basic)
# ----------------------------
_SCALE_SHARP = {
    "C":  [0,2,4,5,7,9,11],
    "G":  [7,9,11,0,2,4,6],
    "D":  [2,4,6,7,9,11,1],
    "A":  [9,11,1,2,4,6,8],
    "E":  [4,6,8,9,11,1,3],
    "B":  [11,1,3,4,6,8,10],
    "F#": [6,8,10,11,1,3,5],
    "C#": [1,3,5,6,8,10,0],
    "F":  [5,7,9,10,0,2,4],
    "Bb": [10,0,2,3,5,7,9],  # left as Bb for readability
    "Eb": [3,5,7,8,10,0,2],
    "Ab": [8,10,0,1,3,5,7],
}

_ROMAN_MAJOR = ["I","ii","iii","IV","V","vi","vii°"]  # diatonic triads quality
_ROMAN_MINOR = ["i","ii°","III","iv","v","VI","VII"]

def detect_key_from_progression(chords: Iterable[str]) -> str:
    """
    Super rough heuristic: pick the major key whose diatonic pitch classes best match roots.
    Returns a key name like 'C' or 'G'. Defaults to 'C' on ties/empty.
    """
    roots = []
    for c in chords:
        p = parse_chord(c)
        if p.symbol != "N.C.":
            roots.append(_pc_index(p.root))
    if not roots:
        return "C"
    best_key = "C"
    best_score = -1
    for key, pcs in _SCale_SHARP_safe().items():
        score = sum(1 for r in roots if r in pcs)
        if score > best_score:
            best_key, best_score = key, score
    return best_key

def _SCale_SHARP_safe():
    # helper to avoid accidental mutation
    return {k: list(v) for k, v in _SCALE_SHARP.items()}

def chord_to_roman(chord: str, key: str = "C", minor: bool = False) -> str:
    """
    Very basic roman numeral mapping by chord root relative to key tonic.
    Ignores many alterations; intended for labeling, not analysis.
    """
    key = _canon_note_name(key)
    pcs = _SCale_SHARP_safe().get(key, _SCALE_SHARP["C"])
    p = parse_chord(chord)
    if p.symbol == "N.C.":
        return "N.C."
    degree = (_pc_index(p.root) - pcs[0]) % 12
    # Find which scale degree matches (approximate)
    try:
        idx = pcs.index((_pc_index(p.root)) % 12)
    except ValueError:
        return "?"  # non-diatonic
    table = _ROMAN_MINOR if minor else _ROMAN_MAJOR
    rn = table[idx]
    # Append simple 7th markers
    if p.quality in {"7", "min7", "maj7", "m7b5", "dim7"}:
        rn += "7"
    return rn

def roman_to_chord(roman: str, key: str = "C") -> str:
    """
    Basic roman->chord root mapping for major keys (triads). Returns canonical symbol.
    Accepts forms like 'ii', 'V', 'vi', 'vii°'. Seventh markers are ignored.
    """
    key = _canon_note_name(key)
    pcs = _SCale_SHARP_safe().get(key, _SCALE_SHARP["C"])
    base = roman.strip().replace("7", "")
    if not base:
        return "N.C."
    # Determine degree index and quality from common forms
    mapping = {
        "I":  (0, "maj"), "ii": (1, "min"), "iii": (2, "min"),
        "IV": (3, "maj"), "V":  (4, "maj"), "vi":  (5, "min"),
        "vii°": (6, "dim"), "viiº": (6, "dim"),
        "i":  (0, "min"), "II": (1, "maj"), "III": (2, "maj"),
        "iv": (3, "min"), "v":  (4, "min"), "VI": (5, "maj"), "VII": (6, "maj"),
    }
    if base not in mapping:
        return "N.C."
    deg, qual = mapping[base]
    root_pc = pcs[deg]
    root = _PITCHES_SHARP[root_pc]
    return normalize_chord(f"{root}{qual}")

# ----------------------------
# Demo usage (only when run directly)
# ----------------------------
if __name__ == "__main__":
    demo = ["bbmaj7", "Fmin7", "G7", "CΔ7", "Dm7/G", "N.C.", "Ebsus4", "A#m7b5", "C+7"]
    print("Normalize:")
    for s in demo:
        print(f"  {s:10s} -> {normalize_chord(s)}")

    print("\nParse & pitches:")
    for s in demo:
        print(f"  {s:10s} -> {parse_chord(s)} -> {chord_to_pitches(s)}")

    print("\nTranspose +2:")
    print([transpose_chord(s, 2) for s in demo])

    print("\nProgression to grid (4 bars, 4/4, 1 chord/bar):")
    prog = ["Cmaj7", "Am7", "Dm7", "G7"]
    print(progression_to_grid(prog, bars=4, beats_per_bar=4, chords_per_bar=1))

    print("\nRoman:")
    print(chord_to_roman("Am7", key="C"), chord_to_roman("G7", key="C"))
    print(roman_to_chord("ii", key="C"), roman_to_chord("V", key="C"))
