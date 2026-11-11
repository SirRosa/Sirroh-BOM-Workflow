# IbBit v6 — uniwersalna paleta grim (3 tony na kolor)

IbBit nie ma profili dla kamienia, drewna, ścian, mostów ani żadnych innych obiektów.
Wszystkie obrazy są przetwarzane tą samą ścieżką:

1. Skalowanie z zachowaniem proporcji i przezroczystości do **256×256 px**, **przed** analizą.
2. Wykrywanie rodzin barw tylko dla dostatecznie nasyconych źródłowych kolorów. Niemal szare piksele pozostają `neutral`; barwy drewniane często należą do `brown`; reszta jest dostępna dla prawdziwych akcentów.
3. Wspólna, **stała paleta 12 rodzin × 3 tony** (`dark`, `mid`, `light`). Wszystkie 36 RGB są zdefiniowane w `PALETTE` w `src/ibbit.py`, nie na podstawie plików wejściowych.
4. Klasyfikacja jasności przez dwa progi kwantylowe dla każdej obecnej rodziny (1/3 i 2/3). Ciemne obrazy nadal używają trzech tonów, o ile ich wejściowa jasność ma rzeczywistą zmienność. Nie ma sztywnych progów, które wymuszają `dark` na całej ścianie.
5. Ograniczenie szumu bilateralnego (z zachowaniem krawędzi), a następnie łączenie spójnych obszarów jednego koloru o powierzchni **mniejszej niż 16 pikseli**. Wybór koloru zastępczego zależy od największej liczby kontaktów z 8 sąsiadami. Kolory są wyłącznie z palety; alfa jest zachowana.

**Kolor zastępujący czerń:** `neutral dark` = RGB `(34, 37, 34)` / `#222522` — zgodny z propozycją użytkownika. Walidator pilnuje, aby każdy inny odcień każdej rodziny był od niego jaśniejszy.

**Istotne ograniczenie:** jeśli obiekt zawiera tylko jednolitą plamę źródłową, skrypt nie wymyśla detali tylko po to, aby wykorzystać wszystkie trzy tony. Pojedyncze wyizolowane plamy mniejsze niż 16 px, które nie stykają się z żadnym innym widocznym kolorem (otoczone transparentnymi pikselami), pozostają bez zmian, ponieważ skrypt nie niszczy kształtu ani nie dokleja do nich obcych kolorów.

### Uruchomienie

```bash
pip install -r requirements.txt
python src/ibbit.py --input-dir katalog_wejsciowy --output-dir katalog_wynikowy
```

Ustawienia: `--size 256` (domyślnie), `--min-area 16` (domyślnie), `--no-smoothing` (debug).

### Testy

```bash
python -m unittest discover -s tests -p test_ibbit.py -v
# Dodatkowy test wszystkich 30 obrazów lokalnych:
IBBIT_TEST_ASSET_DIR=/sciezka/do/30/png python -m unittest discover -s tests -p test_ibbit.py -v
```

W v5 problemem były sztywne nadpisania tonów `dark` po podziale kwantylowym, czego skutkiem była utrata odcienia `mid` w ścianach. V6 usuwa tę ścieżkę. Nowy silnik pozostaje deterministyczny i niedestrukcyjny — nie modyfikuje plików źródłowych.

W v6 dodatkowo istnieje **uniwersalne, wspólne dla wszystkich rodzin ograniczenie jasności źródła**: progi `dark/mid` i `mid/light` nie mogą spaść poniżej odpowiednio 12 i 21 jednostek CIE L*. Zapobiega to rozjaśnianiu prawie czarnych fragmentów (np. szpar między deskami) tylko dlatego, że były one relatywnie najjaśniejszą jedną trzecią bardzo ciemnego materiału. Nie jest to profil drewna ani kamienia.