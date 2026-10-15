# Sirroh B&om Workflow

Automatyczny B&om dla atlasów ikon. Przetwarzanie jest **niegeneratywne**, lokalne dla każdego kafelka i odtwarzalne. Oryginały pozostają w `input/`.

## Struktura

```text
input/                     Oryginalne atlasy PNG
output/                    Atlasy po przetworzeniu (*_BOM.png) i raporty JSON
src/bom_icons.py           Silnik przetwarzający
.github/workflows/bom.yml  GitHub Actions: testy, konwersja, publikacja
requirements.txt           Wymagane biblioteki Python
 tests/                     Testy automatyczne
```

## Zasada działania

1. Analiza przezroczystych przerw w obrazie wykrywa **liczbę rzędów i kolumn** bez sztywnego schematu 8×9; w razie niejednoznaczności program zgłasza błąd zamiast ciąć atlas przypadkowo.
2. Ikony są wyodrębniane jako osobne obiekty (również gdy wykraczają poza źródłowe granice kafelka), skalowane **proporcjonalnie** do kafelków 96×96 px i pozycjonowane: **poziomo na środku, pionowo przy dolnej krawędzi z marginesem 2 px**. Ostatni pusty kafelek pozostaje pusty.
3. Zachowano BOM: usuwanie ukrytego RGB, oczyszczanie maski, 3× bilateral denoise (d=7, sigmaColor=28, sigmaSpace=5), redukcję palety rzeczywistych kolorów przez k-means bez ditheringu, cleanup 5×5 / 1 pass / próg 0.46 oraz hard alpha 0/255.
4. **Każdy kafelek posiada własną paletę** maks. 6 kolorów (domyślny limit B&om dla większości kategorii). Można porównawczo ustawić `--colors 4`.
5. Raport JSON zawiera liczbę kafelków, rozmiary, granice wykrytej siatki i statystyki palet.

## GitHub Actions

Workflow startuje automatycznie po zmianie plików w `input/`, `src/`, `tests/`, `requirements.txt` lub workflow. Można też ręcznie uruchomić go w zakładce **Actions → BOM - Przetwarzanie atlasow → Run workflow**.

Wyniki trafiają do `output/` w repozytorium **oraz** do artefaktu przebiegu Actions. Folder `output/` nie wyzwala kolejnego workflow, więc nie ma pętli commitów.

## Lokalne uruchamianie

```bash
pip install -r requirements.txt
python src/bom_icons.py --input-dir input --output-dir output
python src/bom_icons.py --input input/atlas.png --output output/test_BOM.png
```

Dla atlasów nietypowych, bez jednoznacznych przezroczystych odstępów:

```bash
python src/bom_icons.py --input input/atlas.png --output output/test_BOM.png --cols 8 --rows 9
```

**Ograniczenie:** autodetekcja wymaga siatki zbliżonej do kwadratowych kafelków i widocznych odstępów między ikonami. Nie interpretuje dowolnie chaotycznego kolażu jako atlasu.
