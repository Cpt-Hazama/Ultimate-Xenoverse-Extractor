# Ultimate Xenoverse Extractor — Phase 1

A functional first-stage GUI and asset indexer for Xenoverse 2 extraction workflows.

## Current implementation

- First-run extracted-assets directory picker with persistent settings.
- Scans either `data`, `data/chara`, or a direct folder of character-code directories.
- Parses the bundled HTML ID lists without external HTML libraries.
- Resolves character display names and retains original codes, e.g. `Goku [GOK]`.
- Blue/red/gray availability coloring from the ID list.
- Searchable category library.
- Per-character asset classification for EMD, ESK, EMM, EMB/DYT, EAN, camera EAN, face/lip EAN, BCS, BAC, BCM, BDM, and BAI.
- Automatic main-rig and same-name physics-rig assignment preview.
- Model, animation, and extraction inspector UI.
- The bundled example `XBF` folder loads automatically if the first directory dialog is cancelled.

## Run

```bat
py -m pip install -r requirements.txt
py run_extractor.py
```

Or run `install_and_run.bat` on Windows.

## Architecture

- `id_lists.py`: HTML table parser and ID indexes.
- `scanner.py`: library discovery, file classification, and automatic skeleton pairing.
- `models.py`: neutral data structures shared by future native parsers and exporters.
- `app.py`: PySide6 GUI.

The next stage should add native binary readers under `ultimate_xenoverse_extractor/formats/`, beginning with ESK and EMD. The Source exporter should consume the neutral structures, not raw file offsets, so coordinate conversion is applied exactly once.
