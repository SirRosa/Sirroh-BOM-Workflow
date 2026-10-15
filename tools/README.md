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


## Podgląd nazw

`tools/preview_atlas_labels.py` generuje **atlas testowy tylko do podglądu**.
Nie zmienia kafelków ani manifestu – odczytuje `atlas_manifest.json`, ładuje istniejące pliki `tiles/*.png` i rysuje siatkę z podpisami.

Przykład:

```bash
python tools/preview_atlas_labels.py \
  --manifest atlas_work/Valheim_Compass_Atlas_BOM/atlas_manifest.json \
  --output atlas_work/Valheim_Compass_Atlas_BOM/preview_named_atlas.png \
  --label-mode name
```

Dostępne tryby podpisów:
- `name` – pokazuje `tiles[].name`
- `prefab` – pokazuje `tiles[].prefab` (lub `name`, gdy prefab pusty)
- `both` – pokazuje `name`, a pod nim `[prefab]`, jeśli różnią się od siebie


## Nazwy prefabów na podstawie Internal ID

Nazwy w `atlas_manifest.json` pochodzą z wiki Valheim: należy wybierać **Internal ID** danego obiektu, a nie wariant wejścia, o ile jest to inny identyfikator niż podany w Wiki.

W przypadku lochów rozróżniamy generator instancji i prefab wejścia, np. `DG_ForestCrypt` (Burial Chambers) versus `Crypt2`, `Crypt3`, `Crypt4` (warianty lokacji). To **nie są synonimy**. Szczegóły i linki źródłowe są w `prefab_mapping_review.json`. Niepewne graficznie symbole to `custom_*`, a nie zgadnięte identyfikatory.

Przy zmianie nazw odpal test podziału i składania oraz odśwież podgląd: 

```bash
python tools/preview_atlas_labels.py --manifest atlas_work/Valheim_Compass_Atlas_BOM/atlas_manifest.json --output atlas_work/Valheim_Compass_Atlas_BOM/preview_named_atlas.png
```

Podgląd z podpisami jest **wyłącznie do oceny**, nie może zastępować atlasu gry.
