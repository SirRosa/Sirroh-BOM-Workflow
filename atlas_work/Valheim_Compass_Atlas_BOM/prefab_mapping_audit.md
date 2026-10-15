# Weryfikacja Internal ID — Valheim Wiki (2026-10-16)

**Zasada:** dla rozpoznawalnych obiektów stosuj dokładny `Internal ID` wskazany na Valheim Wiki. Nie podstawiaj arbitralnie wariantu lokacji ani przedmiotu. Dla ikon symbolicznych stosuj `custom_*`.

**Ochrona danych:** tylko `name` i `prefab` w `atlas_manifest.json` są edytowane. Zawsze zachowuj `id`, `file`, `row`, `column`, współrzędne i `sha256_rgba`.

## Wynik

- Prefaby/ID: **45**
- Ikony opisowe `custom_*`: **26**
- Puste miejsca: **1**
- Podział i składanie: **100% RGBA i identyczny bajtowo PNG** (weryfikacja w `reassembled.verification.json`).

## Istotne rozróżnienie: dungeon generator ≠ prefab wejścia

| Komórka | Co przedstawia | Główne ID w manifeście | Generator / ID wnętrza | Prefaby wejścia w świecie |
|---|---|---|---|---|
| r09_c01 | Burial Chambers | `DG_ForestCrypt` | `DG_ForestCrypt` | `Crypt2`, `Crypt3`, `Crypt4` |
| r09_c02 | Sunken Crypt | `SunkenCrypt4` | `DG_SunkenCrypt` | `SunkenCrypt4` (inne warianty istnieją) |
| r09_c03 | Troll Cave | `DG_Cave` | `DG_Cave` | `TrollCave02` |
| r09_c04 | Frost Cave | `MountainCave02` | brak osobno udokumentowanego ID | `MountainCave02` |
| r09_c05 | Howling Caverns | `Hildir_cave` | `DG_Hildir_Cave` | `Hildir_cave` |
| r09_c06 | Infested Mine | `Mistlands_DvergrTownEntrance1` | `DG_DvergrTown` | warianty `1` i `2` |
| r09_c07 | Putrid Hole | `MorgenHole1` | `DG_Hole` | warianty `1`, `2`, `3` |

Źródła: [Burial Chambers, Internal ID](https://valheim.fandom.com/wiki/Burial_Chambers), [Sunken Crypt, Internal ID](https://valheim.fandom.com/wiki/Sunken_Crypt), [Infested Mine, Internal ID](https://valheim.fandom.com/wiki/Infested_Mine), [Putrid Hole, Internal ID](https://valheim.fandom.com/wiki/Putrid_Hole), [Jötunn prefab IDs](https://valheim-modding.github.io/Jotunn/data/prefabs/prefab-list.html), [Valheim 1.0 location prefabs](https://valheimhub.wiki/db/locations/burial-chamber/).

**Uwaga:** w odniesieniu do Cave generatora i prefabu wejścia to dwa różne identyfikatory. `DG_Cave` i `TrollCave02` nie są synonimami; wybór zależy od tego, czy mod odczytuje instancję lochu czy lokację świata.

## Poprawione pozycje

| Komórka | Nowe ID / custom | Powód |
|---|---|---|
| `r01_c01` | `custom_runestone_zigzag` | Runa nie określa biomu ani konkretnego wariantu kamienia; dawniej arbitralnie Runestone_Meadows |
| `r01_c08` | `custom_stone_tower` | Stylizowana wieża bez cech pozwalających zidentyfikować StoneTower1 |
| `r02_c01` | `custom_wooden_house` | Symbol chaty nie określa żadnego z 13 wariantów WoodHouse1–13 |
| `r06_c04` | `custom_pickaxe` | Symbol ogólnego kilofa: nie wskazuje materiału, więc PickaxeIron było zgadywaniem |
| `r06_c05` | `custom_sword` | Symbol ogólnego miecza: nie wskazuje materiału, więc SwordIron było zgadywaniem |
| `r06_c08` | `custom_fire_runestone` | Stylizowana runa ognia nie identyfikuje runestone konkretnego biomu |
| `r07_c01` | `custom_horned_runestone` | Rogaty kamień rytualny nie jest jednoznacznie ołtarzem Eikthyra |
| `r07_c05` | `piece_cauldron` | Ikona przedstawia wiszący kociołek nad ogniem, nie ruszt stacji pieczenia |
| `r07_c07` | `custom_fish` | Sylwetka nie identyfikuje gatunku Fish1 (okoń) |
| `r08_c06` | `custom_world_tree` | Symbol drzewa świata nie przedstawia konkretnego odrostu YggaShoot1 |
| `r09_c01` | `DG_ForestCrypt` | Burial Chambers: wiki podaje DG_ForestCrypt jako Internal ID; Crypt2 było tylko wariantem wejścia |
| `r09_c03` | `DG_Cave` | Generator jaskini trolla: DG_Cave; wariant wejścia w świecie to TrollCave02 |

## Pliki do oceny

- `atlas_manifest.json`: rzeczywiste nazwy, prefab i pozycje.
- `prefab_mapping_review.json`: źródła, warianty, uwagi o pewności.
- `preview_named_atlas.png`: podgląd z nazwami — **nie nadaje się do użycia w grze**.
- `reassembled_after_wiki_review.png`: rekonstrukcja tożsama z oryginałem (lokalny test).

**Nie zmieniać**: `backups/1.0/`.
