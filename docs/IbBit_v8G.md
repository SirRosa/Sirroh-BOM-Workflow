# IbBit v8G — wariant glebowy

**Oddzielny, niedestrukcyjny wariant IbBit v8**, używający tylko trzech konkretnych kolorów z uniwersalnej palety v8. Nie zmienia głównego skryptu IbBit.

## Dokładna paleta

| Przedział jasności źródła | Odcień z IbBit v8 | RGB | HEX |
|---|---|---|---|
| Najciemniejsze ~12% | Neutral dark | 34, 37, 34 | `#222522` |
| Środkowe ~76% | Brown dark | 58, 41, 26 | `#3A291A` |
| Najjaśniejsze ~12% | Neutral mid | 72, 82, 70 | `#485246` |

Niezależnie od materiału piksele są mapowane na trzy tony powyżej. Nie ma klasyfikacji do innych rodzin barw.

## Przetwarzanie

1. **Najpierw** skalowanie do 256×256 z zachowaniem proporcji i kanału alfa.
2. Wygładzanie jasności filtrem bilateralnym, zachowując krawędzie.
3. Jeden **wspólny próg globalny dla całej paczki** — zapobiega różnicom tonalnym pomiędzy podobnymi wariantami tekstur.
4. Mapowanie najciemniejszych, średnich i najjaśniejszych wartości na stałą paletę RGB.
5. Usuwanie plam koloru **mniejszych niż 16 px**: wybór sąsiedniego tonu z największą liczbą kontaktów 8-sąsiedztwa. Plamy ≥16 px są zachowane.
6. PNG RGBA, raport JSON, brak zmian w plikach wejściowych.

## Uruchomienie

```bash
pip install -r requirements.txt
python src/ibbit_v8g.py --input-dir ./gleba_input --output-dir ./gleba_v8g
```

Opcje: `--size 256`, `--shadow-fraction 0.12`, `--light-fraction 0.12`, `--min-area 16`.

### Ograniczenia

Mapowanie jest tonalne, nie semantyczne. Nie rozróżnia automatycznie kamieni, korzeni, desek ani gleby. Wspólna paleta zmniejsza ryzyko rozjazdu kolorystycznego, ale nie gwarantuje identyczności pikseli na przeciwległych krawędziach ani pełnego seamless blendingu.