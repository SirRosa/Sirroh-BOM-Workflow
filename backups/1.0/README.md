# Sirroh BOM Workflow — kopia referencyjna 1.0

**STATUS: ZACHOWAĆ / NIE NADPISYWAĆ.**

To pełna kopia referencyjna wersji **BOM G**, zaakceptowanej przez użytkownika jako najlepszy dotychczasowy wynik. Nazwa backupu: **1.0**. Wersja silnika w skrypcie pozostaje `BOM-ICONS-2026-10-15-G` celowo, aby zachować identyczność z pierwotnym działającym rozwiązaniem.

**W RAZIE BŁĘDÓW, REGRESJI LUB POGORSZENIA GRAFIKI NALEŻY WZOROWAĆ SIĘ NA ROZWIĄZANIACH Z TEJ WERSJI 1.0.** Najpierw porównaj nowy kod i wynik z tym backupem; odtwórz sprawdzone algorytmy, zamiast dopisywać korekty w ciemno.

## Zweryfikowany stan

- Oryginalny atlas: `input/Valheim_Compass_Atlas.png` (nieprzetworzony).
- Przetworzony atlas: `output/Valheim_Compass_Atlas_BOM.png`, wraz z raportem JSON.
- Silnik i testy: `src/bom_icons.py`, `tests/test_bom.py`.
- Workflow i zależności: `.github/workflows/bom.yml`, `requirements.txt`.
- Automatyczne wykrywanie siatki: 8×9, 71 ikon + 1 pusty kafelek.
- Rozmiar: 96×96 px/ikona, poziomo wyśrodkowane, pionowo do dołu z odstępem 2 px.
- Maksymalnie 6 kolorów na ikonę, przezroczystość 0/255.
- Zabezpieczenia barw: brak zielonych przebarwień nordyckiej maliny (rząd 4, kolumna 8).
- Zabezpieczenia jasności: brak jasnych plamek na ciemnym porożu czaszki jelenia (rząd 7, kolumna 2).
- GitHub Actions: 8 testów przeszło; wynik zbudowany i zapisany.

## Nienaruszalny punkt odniesienia

- Oryginalny commit wersji G (po wygenerowaniu output): `667b213190d94a3c1a921545ec3a1045a7dd94ae`.
- Osobna gałąź Git: `backup/1.0`, wskazuje oryginalny commit.
- Backup w folderze `backups/1.0/` zawiera kopie plików; następne wersje rozwijaj poza tym folderem.
- Git blob wynikowego PNG: `b9e27e54dd8fa2e7ce5577efdb742c343a8ba180`.

**Ważne:** nie usuwać ani nie modyfikować `backups/1.0` i `backup/1.0` podczas zwykłych iteracji. Nowe eksperymenty powinny bazować na tej wersji i być porównywane z zapisanym PNG. W razie regresji przywróć rozwiązania z 1.0, zachowując testy regresyjne.
