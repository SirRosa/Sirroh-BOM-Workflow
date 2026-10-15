# Audyt mapowania prefabów — stan wznowionej pracy

Data ponownej kontroli: 2026-10-16. Plik nie zmienia manifestu ani ikon, stanowi kolejkę do zatwierdzenia.

## Co jest zapisane

- `atlas_manifest.json`: 72 pozycje, 54 prefabów, 17 nazw `custom_*`, 1 puste miejsce.
- `prefab_mapping_review.json`: opis wszystkich ikon, `confidence`, `looks_like` i uwagi o niejednoznacznościach.
- Nazwy plików `tiles/rRR_cCC.png` oraz identyfikatory i pozycje są stałe: zmieniać wyłącznie `name` i `prefab` w manifeście.
- Backup `backups/1.0/` jest niezależny i pozostał bez zmian.

## Test odtworzenia po nazwaniu

- Atlas 8 × 9, kafelki 96 × 96, łącznie 71 niepustych.
- Ponownie uruchomiono `tools/assemble_atlas.py` z nazwanym manifestem oraz `tools/test_roundtrip.py`: 4/4 testy przeszły.
- Zgodność obrazu RGBA: **100%**; zgodność zakodowanych bajtów PNG: **100%**.
- SHA-256 oryginalnego PNG oraz rekonstrukcji: `cd58639cd50b84c939d29dfb61172f27f9a0cbf8424d8837d47bf644330fedab`.

## Źródła prefabu

- Jötunn — [automatycznie wygenerowana lista prefabów Valheim 1.0.7](https://valheim-modding.github.io/Jotunn/data/prefabs/prefab-list.html).
- [Lista lokacji Valheim 1.0.16](https://asharppen.github.io/Valheim.SpawnThat/configs/data/locations-by-biome.html).
- To, że identyfikator istnieje, **nie dowodzi**, że reprezentuje dokładnie rysunek danej ikony. Wpisy medium wymagają rozstrzygnięcia użytkownika.

## Pozycje do ręcznego zatwierdzenia — 21

| Kafelek | Przypisany prefab | Co przedstawia ikona | Powód niepewności |
|---|---|---|---|
| `r01_c01` | `Runestone_Meadows` | Kamień runiczny z czerwonym zygzakiem | Runa nie wskazuje konkretnego biomu; przypisany reprezentatywny istniejący kamień runiczny |
| `r01_c08` | `StoneTower1` | Kamienna wieża | Wiele wariantów wież w Valheim; wybrano istniejący reprezentatywny wariant |
| `r02_c01` | `WoodHouse1` | Drewniana chata | Uproszczona ikona budynku; wybrano reprezentatywny prefab chaty |
| `r02_c06` | `TrophySkeleton` | Ludzka czaszka | Prefab trofeum szkieletu, graficzna stylizacja czaszki |
| `r03_c01` | `Stone` | Kamienie | Ikona przedstawia skupisko głazów, nazwa rzeczywistego materiału |
| `r06_c01` | `Beech_Sapling` | Młoda sadzonka drzewa | Reprezentatywna sadzonka buka |
| `r06_c03` | `Beech1` | Zielone drzewo liściaste | Reprezentatywny buk; możliwe inne drzewa liściaste |
| `r06_c04` | `PickaxeIron` | Kilof | Uproszczona ikona, wybrano żelazny typ |
| `r06_c05` | `SwordIron` | Miecz | Uproszczona ikona, wybrano żelazny typ |
| `r06_c06` | `ShieldWood` | Okrągła tarcza malowana | Drewniana tarcza pozwala na malowane wzory |
| `r06_c08` | `Runestone_Ashlands` | Kamień z ognistą runą | Runiczna ikona jest symboliczna; wybrano istniejący prefab rune z Popiołów |
| `r07_c01` | `Eikthyrnir` | Rytualny kamień z rogami | Przypisanie do lokalizacji ołtarza Eikthyra, nie do samej konstrukcji kamienia |
| `r07_c02` | `TrophyDeer` | Czaszka jelenia z porożem | Trofeum jelenia w grze nie musi być dokładnie kościane |
| `r07_c03` | `Player_tombstone` | Kamienny nagrobek z krzyżem | Uproszczona wersja wizualna nagrobka gracza |
| `r07_c05` | `piece_cookingstation` | Trójnóg nad ogniskiem | Stacja gotowania nad ogniem, uproszczona forma |
| `r07_c07` | `Fish1` | Ryba | Reprezentatywny gatunek: okoń; rysunek nie rozróżnia gatunku |
| `r08_c04` | `Troll` | Duży humanoid / troll | Uproszczona sylwetka, najbardziej odpowiada trollowi |
| `r08_c06` | `YggaShoot1` | Drzewo świata z korzeniami | Reprezentatywna realna roślina Yggdrasil; rysunek jest symboliczny |
| `r09_c01` | `Crypt2` | Kamienna komora grobowa / dolmen | Jeden z wariantów wejść do Burial Chamber; wygląd może przypominać Dolmen01 |
| `r09_c06` | `Mistlands_DvergrTownEntrance1` | Zainfekowana kopalnia | Dwa możliwe prefaby wejścia; reprezentatywny wariant 1 |
| `r09_c07` | `MorgenHole1` | Jaskinia Morgena w Popiołach | Trzy podobne warianty, wybrano pierwszy |

### Najważniejsze niejednoznaczności

- `r09_c01 / Crypt2`: pierwszy obraz w rzędzie jaskiń przypomina również dolmen (`Dolmen01`). Obecnie pozostaje `Crypt2`, bo w atlasie ma reprezentować wejście do Burial Chambers; zweryfikować przed zmianą.
- `r07_c01 / Eikthyrnir`: prawdziwy prefab lokacji ołtarza, ale rysunek symbolizuje kamień rytualny.
- `r08_c06 / YggaShoot1`: istniejąca roślina/odrost Yggdrasil; ikona raczej symbolizuje drzewo świata.
- `r09_c06` i `r09_c07`: poprawne rodziny wejść, lecz dokładny wariant prefabu (`...1`, `...2`, `...3`) nie wynika z ilustracji.

## Pozycje custom do ewentualnego dopasowania — 17

- `r01_c02` → `custom_stone_diamond` — Kamienny romb.
- `r01_c03` → `custom_diamond_rune` — Romb z wyciętym geometrycznym symbolem.
- `r01_c06` → `custom_tent` — Namiot lub schronienie.
- `r02_c03` → `custom_wagon_wheel` — Drewniane koło wozu.
- `r02_c04` → `custom_anchor` — Kotwica statku.
- `r02_c05` → `custom_spiral_rune` — Kamienna spirala.
- `r02_c07` → `custom_stone_cross` — Kamienny krzyż.
- `r02_c08` → `custom_horned_helmet` — Hełm z rogami.
- `r03_c04` → `custom_stone_pedestal` — Okrągły głaz na postumencie.
- `r03_c07` → `custom_loot_sack` — Worek na łup.
- `r03_c08` → `custom_checkmark` — Znak zatwierdzenia.
- `r04_c01` → `custom_red_cross` — Czerwony znak X.
- `r04_c02` → `custom_exclamation_mark` — Czerwony wykrzyknik.
- `r05_c03` → `custom_white_radish` — Biała rzodkiew z zielonymi liśćmi.
- `r05_c08` → `custom_white_flowers` — Białe kwiaty na zielonej łodydze.
- `r07_c08` → `custom_paw_print` — Odcisk zwierzęcej łapy.
- `r08_c07` → `custom_rune_k` — Kamień z czerwonym znakiem przypominającym runę K.

**Dalej:** wspólnie potwierdzać niepewne pozycje na podstawie atlasu, bez zgadywania konkretnego prefabu na podstawie ogólnego symbolu. Korekty tylko nazw/prefabów w manifeście; po każdej zmianie ponowić assembly i SHA.
