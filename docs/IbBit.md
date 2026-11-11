# IbBit v7 — uniwersalna paleta bagiennego grimdark

Wersja 7 zachowuje **algorytm v6**, włącznie z 12 rodzinami, trzema tonami na kolor, neutral-first, proporcjonalnym resize do 256×256 przed obróbką, zachowaniem alfa i scalaniem sąsiadujących plam mniejszych niż 16 pikseli. **Nie ma profili materiałów**.

## Zmiana

Wspólna, stała paleta 36 barw została przyciemniona i odbarwiona: kamień ma popielato-bagienny charakter, drewno otrzymało wilgotny, ziemisty brąz, pozostałe barwy akcentowe są ograniczone i zgaszone. Wybrana przez użytkownika „czerń” `#222522` pozostaje najciemniejszym kolorem całej palety. Wszystkie rodziny posiadają dokładnie 3 rosnące jasnością tony.

Dla kontrolnej kolekcji 30 assetów średnia luminancja widocznych pikseli zmniejszyła się z 84,64 (v6) do 68,90 (v7), czyli o 18,59%. Kanał alfa był identyczny w 30/30 plików. Uwaga: ponieważ przy remisach czyszczenia małych plam algorytm rozstrzyga podobieństwem jasności barw palety, podmiana palety zmieniła przypisania w 0,136% widocznych pikseli. Algorytm czyszczenia nie był zmieniany.

## Użycie

```bash
python src/ibbit.py --input-dir input --output-dir output
```

Wymagania: `Pillow`, `numpy`, `opencv-python-headless`. Wszystkie 36 barw znajdują się w słowniku `PALETTE` w `src/ibbit.py` i są wspólne dla każdego przetwarzanego obrazu.