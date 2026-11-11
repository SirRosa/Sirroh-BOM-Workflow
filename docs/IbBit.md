# IbBit v1 — ograniczona paleta 3 tonów na rodzinę koloru

Niezależny, niedestrukcyjny skrypt graficzny do stylizacji tekstur 2D. Oryginały pozostają bez zmian.

**Kolejność:** RGBA → dopasowanie do 256×256 z zachowaniem proporcji → bilateralne filtrowanie zachowujące krawędzie → przypisanie do rodzin barw (w tym osobno brąz i pomarańcz) → 3 odcienie w LAB dla każdej obecnej rodziny → usunięcie drobnych wysp tonów w obrębie tej samej rodziny → PNG RGBA.

Model kolorów: każda rodzina ma wyłącznie trzy możliwości `dark`, `neutral` (średni), `light`. Ton jest wybierany na podstawie lokalnej jasności. Paleta jest adaptacyjna wobec wejścia (nie ma jednego globalnego brązu lub pomarańczu). Czarny/szary/biały należy do rodziny `neutral`. Alfa jest po przeskalowaniu zachowywana bez zmian. Filtr łączy małe wyspy tonów (≤8 px) tylko w tej samej rodzinie; nie wycina strukturalnych szczelin i dłuższych konturów.

Wymagania: Pillow, numpy, opencv-python-headless (z istniejącego `requirements.txt`).

```bash
python src/ibbit.py --input-dir /sciezka/original_png --output-dir /sciezka/ibbit_output
python -m unittest discover -s tests -p test_ibbit.py -v
```

Opcjonalne parametry: `--tone-gap 17`, `--size 256`, `--no-smoothing`. Katalog docelowy nie może znajdować się wewnątrz źródłowego.

**Uwaga:** IbBit jest stylizacją inspirowaną ograniczonym, plamowym cieniowaniem gier retro; nie jest wierną rekonstrukcją konkretnego stylu pixel-art. Pierwsza wersja jest przeznaczona do oceny wizualnej. Wygładzenie plam nie oznacza gwarantowanej eliminacji artefaktów dla każdej grafiki.