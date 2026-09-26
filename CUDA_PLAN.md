# CUDA-Plan für INDIGO (`indigo_libs`, `indigo_server`)

Stand: 2026-09-26 · Zielplattformen: NVIDIA Jetson und Jetson Orin (JetPack 7.2)

## 1. Ergebnis der Untersuchung

`indigo_server` enthält keinen rechenintensiven Code (Start der Treiber, Netzwerk, Ressourcen). Beschleunigen lässt sich nur Bildverarbeitung in `indigo_libs`.

Baseline auf dem Entwicklungsgerät (Jetson Nano, 2 CPU-Kerne sichtbar, CUDA 10.2, sm_53). Testbild: 4144×2822 (11,7 MP), 16 Bit, RGGB, mit Vignettierung, Rauschen, 600 Sternen und Hotpixeln (`indigo_test/cuda/test_image.h`). Median aus 5 Läufen, reproduzierbar mit `make -C indigo_test/cuda bench`. Es sind weiterhin synthetische Bilder, echte Kameraframes fehlen noch.

| Funktion | Zeit | Aufrufer |
|---|---|---|
| Stretch-Parameter 16 Bit RGGB | 17 ms (vorher 78 ms, siehe 3) | `indigo_raw_to_jpeg` |
| `indigo_stretch_16_rggb` (Debayer + Stretch) | 459 ms | `indigo_raw_to_jpeg` (jedes Preview-Bild) |
| `indigo_stretch_16` (Mono) | 173 ms | `indigo_raw_to_jpeg` |
| **libjpeg-Kompression RGB8, Qualität 90** | **1190–1430 ms** (schwankt) | `indigo_raw_to_jpeg` |
| `indigo_contrast` / `indigo_stddev_*` | 116 ms | Autofokus (Imager-Agent) |
| **`indigo_find_stars_precise`** (100 Sterne) | **3341 ms** (nach Phase 1.1: 114 ms) | Guider- und Imager-Agent |
| `indigo_binarize` | 54 ms | Bahtinov |
| `indigo_equalize_bayer_channels` | 48 ms | Guider- und Imager-Agent |
| `indigo_donuts_frame_digest`, Region 256×256 / Vollbild | 20 ms / 3282 ms | Guider-Agent |
| `base64_encode` (23 MB) | 57 ms | Blob-Transfer |

**Zwei Befunde bestimmen die Reihenfolge:**
1. Die JPEG-Kompression (1180 ms) kostet mehr als doppelt so viel wie Stretch und Debayer (459 ms). Ein GPU-Stretch allein würde die Preview-Zeit nur um etwa ein Viertel senken. Das gebündelte libjpeg ist die IJG-Version 9.2 ohne SIMD.
2. `indigo_find_stars_precise` scannt für jeden gefundenen Stern das ganze Bild neu (`while (lmax > threshold)` mit Vollbild-Schleife, also O(Sterne × Pixel)). Das ist ein Algorithmusproblem der CPU-Version, mit CUDA würde man es nur verstecken.

### Bewertung

**Gut geeignet**
1. Stretch und Debayer (`indigo_stretch.cpp`): reine Pixel-Parallelität, heute nur `std::thread` auf 2 Kernen.
2. Stretch-Parameter: Median und MAD über zwei `nth_element`-Läufe. Bei 16 Bit geht das exakt über ein 65536-Bin-Histogramm, auf CPU wie GPU.
3. Ganzbild-Reduktionen in `indigo_raw_utils.c`: `indigo_contrast` / `indigo_stddev_*`, Vorstufe von `indigo_find_stars_precise`, `indigo_binarize`, `indigo_update_saturation_mask`.

**Bedingt / zuerst auf der CPU verbessern**
- Hough-Transformation (Bahtinov): `cos`/`sin` werden pro Pixel und Theta (3000 Winkel) neu berechnet. Eine Sinus-Tabelle bringt auf der CPU schon viel. `indigo_skeletonize` ist iterativ und GPU-tauglich.
- Donuts: Die Zeit stammt aus `clear_hot_pixel_*` pro Pixel (zwei Vollbild-Durchläufe, 3,3 s; mit der üblichen Region 20 ms). Die FFTs sind 1D mit höchstens einigen tausend Punkten (cuFFT lohnt nicht). Beim Guiding ist die Region klein, dann überwiegt der Launch-Overhead.
- `indigo_make_psf_map`: nicht gemessen.

**Nicht sinnvoll**
Base64/MD5 (speicher- bzw. netzwerkgebunden), Bus, XML/JSON, TCP, Timer, FITS/TIFF/SER/AVI-I/O, Platesolver, libraw, `indigo_align.c`, `indigo_polynomial_fit.c`, `indigocat` (nur grob angesehen).

**Engpass Preview:** Die libjpeg-Kompression (siehe oben) bleibt nach einem GPU-Stretch auf der CPU und wird zum Hauptposten.

## 2. Randbedingungen

- **Toolchain-Spanne:** Das Entwicklungsgerät läuft mit JetPack 4.x (CUDA 10.2, Ubuntu 18.04). Das Zielsystem ist JetPack 7.2 (Ubuntu 24.04, neuere CUDA-Version). Der `.cu`-Code muss mit beiden Toolchains übersetzen: C++11, keine neuen CUDA-Features, `__shfl_*_sync` statt der alten Warp-Intrinsics.
- **Architekturen:** Nano sm_53, Orin sm_87. Neuere Toolkits (CUDA 13) unterstützen Maxwell/Pascal/Volta nicht mehr. Ein Nano kann daher nur mit einem älteren Toolkit gebaut werden, und die Architektur wird pro Gerät gewählt.
- **Nicht verifiziert:** Ich konnte JetPack 7.2 nicht prüfen (CUDA-Version, Paketnamen, `nvidia-jetpack`-Versionsschema, sm-Nummer von Thor). Die Umgebungsprüfung ist deshalb tolerant ausgelegt (siehe 3) und muss auf einem Orin mit JetPack 7.2 einmal ausgeführt werden.
- **Speicher:** Jetson und Orin haben gemeinsamen CPU/GPU-Speicher (4 GB beim Nano). Kopieren ist billig, aber nicht kostenlos. Zero-Copy oder `cudaMallocManaged` ist zu prüfen. Puffer müssen gepoolt werden, kein `cudaMalloc` pro Frame.
- **Nebenläufigkeit:** Mehrere CCD-Treiber und Agenten rufen die Funktionen gleichzeitig auf. Pro Aufrufer ist ein Stream nötig, Kontext und Pool brauchen Locking.
- **Fallback:** Die CPU-Implementierung bleibt Referenz und Fallback. Die GPU-Version wird nur bei großen Bildern und wenn ein Gerät vorhanden ist verwendet.

## 3. Schritt 1 (erledigt): Build-Schalter und Umgebungsprüfung

### Bedienung

```
make CUDA=yes                  # mit CUDA bauen (Umgebung wird vorher geprüft)
make CUDA=no                   # ohne CUDA (Standard)
make cuda-check                # vollständiger Bericht inkl. Test-Kernel auf der GPU
make CUDA=yes CUDA_ARCH=87     # SM-Architektur explizit setzen
make CUDA=yes CUDA_HOME=/usr/local/cuda-12.6
make CUDA=yes CUDA_SKIP_PKG_CHECK=1   # nicht per dpkg installiertes CUDA
```

Weitere Variablen: `CUDA_STD` (Standard `c++11`).

### Was geändert wurde

- `Makefile`: Schalter `CUDA` (Standard `no`, vorher war CUDA automatisch an, sobald `/usr/local/cuda` existierte und wurde immer fest gelinkt). Bei `CUDA=yes` bricht der Build mit einer Fehlermeldung ab, wenn die Prüfung fehlschlägt. `-gencode` wird aus der erkannten Architektur erzeugt (Maschinencode plus PTX der letzten Architektur). `-std` kommt aus `CUDA_STD`, Include- und Lib-Pfade kommen aus der Prüfung (auch `targets/*-linux/`-Layouts). Ein Stempel `.cuda_config` (in `.gitignore`) sorgt dafür, dass `Makefile.inc` und die Objekte von `indigo_libs` genau dann neu erzeugt werden, wenn CUDA umgeschaltet wird.
- `indigo_libs/Makefile`: `.cu`-Regel nutzt `$(NVCC)` statt festem `nvcc`, Statuszeile zeigt CUDA an.
- `tools/check_cuda_env.sh` (neu): Prüft
  - Plattform (Jetson-SoC aus dem Device Tree, Orin/Nano/Xavier/TX2/Thor) und L4T/JetPack-Generation (Warnung, wenn `nvidia-jetpack` älter als 7.2 ist),
  - `nvcc` (≥ 10.2), Host-Compiler, Unterstützung der Zielarchitektur durch nvcc (`--dryrun`),
  - `cuda_runtime.h`, `libcudart`, `libcuda` und den ldconfig-Cache,
  - Ubuntu-Pakete `cuda-nvcc-X-Y`, `cuda-cudart-X-Y`, `cuda-cudart-dev-X-Y`, `nvidia-l4t-cuda` (Fehler, mit `apt install`-Hinweis), `build-essential`, `nvidia-jetpack` (Warnungen), Gruppe `video`,
  - im Modus `check` zusätzlich Übersetzen, Linken (wie das Makefile: `nvcc -c` und danach `g++ … -lcudart`) und Ausführen eines Test-Kernels.

### Getestet (auf dem Nano, JetPack 4.x, CUDA 10.2)

- Vollständiger Bericht ohne Fehler, Test-Kernel läuft auf sm_53.
- Fehlerfälle brechen mit passenden Hinweisen ab: falsches `CUDA_HOME`, nicht unterstützte Architektur (`CUDA_ARCH=87` mit nvcc 10.2).
- Umschalten `no` → `yes` → `no` regeneriert `Makefile.inc` und baut die Objekte neu, ein zweiter Lauf ohne Änderung regeneriert nichts.
- Eine Test-`.cu` wird mit `$(NVCC)` und dem erzeugten `-gencode` übersetzt.
- `libindigo` baut mit `CUDA=yes` und `CUDA=no`. Bei `CUDA=yes` taucht `libcudart` noch nicht in `ldd` auf, weil bisher kein Code CUDA-Symbole benutzt und der Linker die Bibliothek verwirft.
- **Nicht getestet:** Orin, JetPack 7.2, CUDA ≥ 12.

Beim Umschalten ist ein Fehler aufgefallen und behoben: `ar rv` ersetzt nur Archivmitglieder, ein `CUDA=yes`-Build ließ `indigo_cuda.o` in `libindigo.a` zurück. Das Archiv wird jetzt vor dem Neuaufbau gelöscht.

### Entscheidung: `libcudart` bleibt fest gelinkt

Ein `CUDA=yes`-Build von `libindigo` braucht `libcudart` beim Start. Das ist bewusst so gewählt: CUDA ist ein Opt-in-Build (Standard `CUDA=no`), auf Jetson und Orin ist die Laufzeit mit JetPack immer vorhanden, und Pakete für andere Systeme werden ohne CUDA gebaut. Der gesamte CUDA-Zugriff läuft über `indigo_libs/indigo_cuda.h`, ein späterer Wechsel auf `dlopen` bleibt damit lokal. Erneut prüfen, falls ein einziges Paket für Systeme mit und ohne GPU gebraucht wird.

### Offene Punkte zum Schalter

- `-lcudart` steht weiterhin in den globalen `LDFLAGS` (auch für Treiber). Sinnvoll ist, das später auf `libindigo` zu beschränken.
- `-DINDIGO_CUDA` steht in `CFLAGS`/`CXXFLAGS` für alle Quellen. Der Stempel baut nur `indigo_libs` neu. Sobald Treiber das Makro auswerten, müssen sie mit erfasst werden.

## 4. Plan

**Phase 0: Basis (erledigt)**
- Build-Schalter und Umgebungsprüfung (Abschnitt 3).
- `indigo_libs/indigo_cuda.h` / `indigo_cuda.cu`: Geräteerkennung, Kontext-Pool (Stream und Puffer je Aufrufer, höchstens 8 Kontexte, danach CPU-Fallback), Selbsttest, `INDIGO_CUDA=0` schaltet zur Laufzeit ab. Ohne CUDA sind es Inline-Stubs. Der Header liegt bewusst nicht in `indigo_libs/indigo/`, weil jeder geänderte Header dort einen erzwungenen `make clean` des ganzen Baums auslöst und der Header intern ist.
- `indigo_test/cuda/`: `make test` (Stretch-Parameter gegen die Original-Implementierung, 1200 bitgenaue Vergleiche; CUDA-API mit 12 Threads; im `CUDA=no`-Build prüft der Test die Stubs) und `make bench`. Beide Modi sind auf dem Nano gelaufen.
- Histogramm-Median in `indigo_compute_stretch_params` statt Sortieren: 78 ms → 17 ms auf demselben 12-MP-Bild, Ergebnis bitgenau gleich.
- Noch nicht gemacht: Messung mit echten Kameraframes, `nvprof`/`tegrastats`, Messung auf einem Orin.

**Phase 1: CPU-Engpässe (ohne CUDA, größter Hebel)**
1. **Erledigt:** `indigo_find_stars_precise` (und damit `_clipped`, `_filtered`, `indigo_find_stars`): einmaliger Durchlauf mit nach Helligkeit sortierter Kandidatenliste statt Vollbild-Scan pro Stern (`find_star_peaks()` in `indigo_raw_utils.c`). Bei zu wenig Speicher fällt die Funktion auf den alten Scan zurück. Test `indigo_test/cuda/test_find_stars.c` gegen eine Kopie der Originalfunktion: 1219 Vergleiche ohne Abweichung (8/16 Bit, Radius 0–8, verschiedene `stars_max`, leere, konstante, gesättigte, dichte und Hot-Line-Bilder). Zeit auf dem 12-MP-Bild mit 100 Sternen und Radius 4: 3843 ms → 114 ms. Zusätzlicher Speicher: 8 Byte je Kandidat (in üblichen Bildern einige tausend).
2. **JPEG-Kompression: mit dem gebündelten libjpeg 9.2 gemessen** (`make -C indigo_test/cuda bench`, Programm `bench_jpeg`; libjpeg-turbo und Hardware-Encoder sind bewusst nicht gemessen). Der Encoder wird heute nur mit `jpeg_set_defaults` und der Qualität konfiguriert. Die Basiszeit schwankt zwischen Läufen (1190–1430 ms für RGB, wohl Takt/Temperatur), Unterschiede unter etwa 10 % sind Rauschen.

   Profil (`gprof`, RGB): DCT-Arbeit 67 % (`forward_DCT`, `jpeg_fdct_16x16`, `jpeg_fdct_islow`), Farbumrechnung `rgb_ycc_convert` 17 %, Huffman `encode_mcu_huff` 15 %. Grund für `jpeg_fdct_16x16`: libjpeg 9 verlegt die Chroma-Unterabtastung standardmäßig in die DCT (`do_fancy_downsampling = TRUE`). Bei Graustufen dominiert dagegen Huffman mit 50 %, weil das nach dem Stretch verrauschte Bild 4,3 MB Ausgabe erzeugt.

   | RGB 12 MP, Qualität 90 | Zeit | Größe | PSNR | Faktor |
   |---|---|---|---|---|
   | heute (Standard) | 1424 ms | 800 KB | 43,25 dB | 1,00 |
   | Qualität 75 / 95 | 1328 / 1541 ms | 279 / 1502 KB | 42,58 / 43,73 dB | 1,07 / 0,92 |
   | `JDCT_IFAST` | 1343 ms | 804 KB | 43,20 dB | 1,06 |
   | `JDCT_FLOAT` | 1152 ms | 775 KB | 43,27 dB | **1,24** |
   | optimierte Huffman-Tabellen | 1333 ms | 668 KB | 43,25 dB | 1,07 |
   | Chroma 4:4:4 | 2225 ms | 1261 KB | 44,68 dB | 0,64 |
   | `do_fancy_downsampling = FALSE` | 1263 ms | 801 KB | 43,18 dB | 1,13 |
   | `do_fancy_downsampling = FALSE` + `JDCT_FLOAT` | 873 ms | 774 KB | 43,20 dB | **1,63** |
   | fertige YCbCr-4:2:0-Ebenen (`jpeg_write_raw_data`), ISLOW | 823 ms | 801 KB | 43,19 dB | **1,73** |
   | fertige YCbCr-4:2:0-Ebenen, `JDCT_FLOAT` | 430 ms | 774 KB | 43,21 dB | **3,31** |

   Graustufen (Mono-Kameras): 1039 ms, 4322 KB. `JDCT_IFAST` bringt nur 7 %, die Zeit hängt an der Huffman-Codierung des verrauschten Bildes.

   Schlussfolgerungen:
   - **Umgesetzt:** `do_fancy_downsampling = FALSE` und `dct_method = JDCT_FLOAT` in `indigo_raw_to_jpeg` (1,63× bei gleichem PSNR). Verhaltensänderung: Die JPEG-Bytes ändern sich (klassischer Chroma-Filter, die Gleitkomma-DCT ist plattformabhängig). Abgesichert durch `test_raw_to_jpeg` (ruft `indigo_raw_to_jpeg` auf einem Fake-Gerät auf, 40 Fälle: 8/16 Bit Mono, alle Bayer-Muster, 24/48 Bit RGB, ungerade Größen, Histogramm; dekodiert und prüft PSNR ≥ 36 dB gegen das gestretchte Bild). Das Guard `JPEG_LIB_VERSION >= 80` schützt vor Systembibliotheken ohne `do_fancy_downsampling`.
   - Der größte Hebel ist, Farbumrechnung und Unterabtastung aus libjpeg zu entfernen: 3,3× für den libjpeg-Teil. Die Umrechnung auf der CPU kostet allerdings 511 ms (ein Thread) und würde den Gewinn aufheben. Sinnvoll ist sie nur, wenn der Stretch-Kernel der GPU (Phase 2) direkt Y-, Cb-, Cr-Ebenen (Cb/Cr halbiert) erzeugt. Für Phase 2 heißt das: Ausgabe nicht als RGB8, sondern als YCbCr-4:2:0 planen.
   - Im Raw-Modus erwartet libjpeg 9 Chroma nur dann in halber Auflösung, wenn `do_fancy_downsampling = FALSE` gesetzt ist. Sonst erwartet es volle Auflösung (16×16-DCT). Das gilt nur für diese Bibliotheksversion.
   - Qualität, `JDCT_IFAST` und optimierte Tabellen bringen für die Zeit nichts.
   - Nicht gemessen: Mehrfaden-Kompression (2 CPU-Kerne sichtbar), libjpeg-turbo, Hardware-Encoder.

3. **Erledigt:** Lesefehler in `indigo_compute_stretch_params` behoben. Die Bayer-Ebenen mit Zeilen- oder Spalten-Offset (`buffer + width…`, `buffer + 1`) wurden mit der vollen Bildgröße abgetastet und lasen bei ungerader Höhe bis zu 301 Werte hinter den Puffer, bei bestimmten Breiten (`(width - 1) % (2 * sample_by) == 0`) ein Element dahinter bzw. in die nächste Zeile. Das Template bekommt jetzt den Offset getrennt und begrenzt Zeilen und Spalten der Ebene. **Verhaltensänderung:** Bei diesen Bildgrößen ändern sich die Stretch-Parameter minimal, weil nicht mehr Speicherinhalt hinter dem Bild in den Median eingeht. Bei allen anderen Größen (z. B. gerade Höhe) bleibt das Ergebnis bitgleich. Der Test extrahiert jede Ebene explizit und vergleicht mit der Originalmathematik; die Puffer enden vor einer Guard-Page. Gegenprobe: Mit dem alten Code stürzt der Test ab.

**Phase 2: Stretch-Pipeline auf der GPU** (`indigo_stretch.cu`)
- Kernels für Stretch (8/16 Bit, Mono, RGB24/48) und Debayer + Stretch (RGGB/GBRG/GRBG/BGGR). Die Parameterberechnung bleibt vorerst auf der CPU (17 ms).
- Signaturen aus `indigo_stretch.h` bleiben. Dispatch: GPU, wenn `indigo_cuda_available()` und mindestens `INDIGO_CUDA_MIN_PIXELS` Pixel, sonst CPU.
- Nutzen hängt vom Ergebnis aus Phase 1.2 ab: Mit CPU-JPEG spart Phase 2 nur den Stretch-Anteil von 459 ms. Mit einem Hardware-Encoder, der GPU-Puffer nimmt, entfällt die ganze CPU-Kette.
- Byte-Swap und RGB-Swap in `indigo_process_image` in den Kernel ziehen (ein Upload pro Frame).

**Phase 3: Statistik für die Agenten** (nur für große Bilder, mit Schwelle)
- Reduktionen in `indigo_contrast` / `indigo_stddev_*` (116 ms), `binarize` (54 ms), `equalize_bayer` (48 ms), Saturation-Mask. Absolut kleine Beträge, Nutzen erst nach Phase 1 neu bewerten.

**Phase 4: Bahtinov**
- Zuerst die Sinus-Tabelle auf der CPU, danach Hough mit Atomics und `skeletonize` auf der GPU, falls noch nötig.

**Phase 5: Optional**
- Donuts: `clear_hot_pixel_*` auf der CPU optimieren. Vollbild dauert 3,3 s, mit Region 20 ms, also nur relevant, wenn Aufrufer große Regionen nutzen (nicht geprüft).
- `indigo_make_psf_map` nach einer Messung.

## 5. Offene Entscheidungen

1. Standard bleibt `CUDA=no`. Falls Auto-Erkennung gewünscht ist, kann `CUDA=auto` ergänzt werden.
2. Die Kombination „Jetson und Orin mit JetPack 7.2“ lässt sich für ein Nano nicht herstellen (JetPack 4.x ist dort das Ende). Klären, welche Geräte tatsächlich mit welcher JetPack-Version getestet werden.
3. Freigabe für Phase 1 (`find_stars`-Umbau, JPEG-Messung). Phase 1 ändert bestehende Funktionen der Bibliothek, auch wenn es die Ergebnisse nicht ändern soll.
