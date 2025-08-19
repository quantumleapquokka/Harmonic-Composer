# midi_to_musicxml.py
# Takes a MIDI file and exports:
#   - piano MusicXML + MIDI
#   - guitar MusicXML + MIDI
#
# Usage (standalone):
#   python midi_to_musicxml.py path/to/file.mid
#
# Usage (as a library):
#   from midi_to_musicxml import render_all_versions
#   render_all_versions("output/ai_transformer_composition.mid")

from pathlib import Path
from typing import Optional
from music21 import converter, instrument, stream
import copy

def _assign_instrument(score: stream.Score, inst_cls) -> stream.Score:
    sc = copy.deepcopy(score)
    parts = sc.parts if sc.parts else [sc]
    for p in parts:
        # wipe any existing instrument markings
        for i in list(p.recurse().getElementsByClass(instrument.Instrument)):
            try:
                i.activeSite.remove(i)
            except Exception:
                pass
        # add ours at time 0
        p.insert(0, inst_cls())
    return sc

def render_all_versions(midi_path: str, out_dir: Optional[str] = None, basename: Optional[str] = None):
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")

    # derive names
    if basename is None:
        basename = midi_path.stem
    if out_dir is None:
        out_dir = str(midi_path.parent)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # read MIDI
    base_score = converter.parse(str(midi_path))

    # piano
    piano_score = _assign_instrument(base_score, instrument.Piano)
    (out_dir / f"{basename}_piano.musicxml").write_text("")  # ensure path exists on some systems
    piano_score.write("musicxml", fp=str(out_dir / f"{basename}_piano.musicxml"))
    piano_score.write("midi",     fp=str(out_dir / f"{basename}_piano.mid"))

    # guitar
    guitar_score = _assign_instrument(base_score, instrument.AcousticGuitar)
    (out_dir / f"{basename}_guitar.musicxml").write_text("")
    guitar_score.write("musicxml", fp=str(out_dir / f"{basename}_guitar.musicxml"))
    guitar_score.write("midi",     fp=str(out_dir / f"{basename}_guitar.mid"))

    print("Exported:")
    print(out_dir / f"{basename}_piano.musicxml")
    print(out_dir / f"{basename}_piano.mid")
    print(out_dir / f"{basename}_guitar.musicxml")
    print(out_dir / f"{basename}_guitar.mid")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python midi_to_musicxml.py path/to/file.mid")
        sys.exit(1)
    render_all_versions(sys.argv[1])
