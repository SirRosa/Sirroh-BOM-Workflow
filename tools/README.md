# Atlas – podział, nazwy i ponowne składanie

Dwa niezależne skrypty Python (Pillow) komunikują się przez **`atlas_manifest.json`**.

1. `tools/split_atlas.py` wycina kafelki 1:1 z atlasu, zapisuje każdy jako `tiles/rRR_cCC.png` oraz manifest wszystkich miejsc w siatce.
2. `tools/assemble_atlas.py` czyta manifest i składa atlas na podstawie **stałych `id`, `row`, `column`, `x`, `y`** – nigdy na podstawie kolejności plików ani wpisów JSON.

Manifest automatycznie ma pola `name` i `prefab` dla każdej pozycji, np.:

```json
{
  "id": "r07_c02",
  "name": "r07_c02",
  "prefab": "",
  "row": 7,
  "column": 2,
  "x": 96,
  "y": 576,
  "width": 96,
  "height": 96,
  "file": "tiles/r07_c02.png",
  "empty": false,
  "sha256_rgba": "..."
}
```

**Do przyszłego nazywania prefabów zmieniaj wyłącznie `name` i/lub `prefab`.** `id`, `file`, współrzędne oraz `sha256_rgba` pozostają bez zmian. Rozdzielacz zachowuje ręcznie nadane nazwy nawet po ponownym uruchomieniu. Jeden przezroczysty kafelek też jest zapisany, żeby odtworzenie było kompletne (łącznie 72 miejsc).

## Weryfikacja

```bash
python tools/split_atlas.py --input output/Valheim_Compass_Atlas_BOM.png --dest atlas_work/Valheim_Compass_Atlas_BOM
python tools/assemble_atlas.py --manifest atlas_work/Valheim_Compass_Atlas_BOM/atlas_manifest.json --output atlas_work/Valheim_Compass_Atlas_BOM/reassembled.png --verify-original output/Valheim_Compass_Atlas_BOM.png
```

Składacz porównuje **każdy piksel RGBA** i SHA-256. Raport `reassembled.verification.json` podaje również, czy identyczne są **bajty PNG** (może się różnić przy innych wersjach kompresji). Domyślnie sprawdza integralność wszystkich sprite'ów. Przy świadomych zmianach grafiki dostępna jest opcja `--allow-modified`, ale wtedy wynik oczywiście przestaje być identyczny z pierwowzorem.

Workflow `.github/workflows/atlas-roundtrip.yml` uruchamia testy, podział, składanie, weryfikację oraz publikuje wynik w `atlas_work/Valheim_Compass_Atlas_BOM/`. Nie zmienia `backups/1.0/` ani istniejącego skryptu BOM.
