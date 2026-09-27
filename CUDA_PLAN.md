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
   - **Umgesetzt:** `do_fancy_downsampling = FALSE` und `dct_method = JDCT_FLOAT` in `indigo_raw_to_jpeg`. Verhaltensänderung: Die JPEG-Bytes ändern sich (klassischer Chroma-Filter, die Gleitkomma-DCT ist plattformabhängig). **Korrektur:** Der erste Commit (`639a57892`) hatte die Float-DCT hinter `#ifdef DCT_FLOAT_SUPPORTED` gesetzt. Dieses Makro steht in `jmorecfg.h` nur für libjpegs eigene Quellen, für Anwendungen ist es nie definiert: Die Float-DCT war damit wirkungslos, das Profil zeigte weiter `jpeg_fdct_islow`. Behoben in Phase 2 (siehe unten). Seitdem pinnt `test_raw_to_jpeg` die Einstellungen byte-genau: Es komprimiert das gestretchte Bild direkt mit libjpeg (Qualität 90, ohne Fancy-Downsampling, Float-DCT) und verlangt exakt dieselben Bytes wie `indigo_raw_to_jpeg`.
   - Der größte Hebel ist, Farbumrechnung und Unterabtastung aus libjpeg zu entfernen: 3,3× für den libjpeg-Teil. Die Umrechnung auf der CPU kostet allerdings 511 ms (ein Thread) und würde den Gewinn aufheben. Sinnvoll ist sie nur, wenn der Stretch-Kernel der GPU (Phase 2) direkt Y-, Cb-, Cr-Ebenen (Cb/Cr halbiert) erzeugt. Für Phase 2 heißt das: Ausgabe nicht als RGB8, sondern als YCbCr-4:2:0 planen.
   - Im Raw-Modus erwartet libjpeg 9 Chroma nur dann in halber Auflösung, wenn `do_fancy_downsampling = FALSE` gesetzt ist. Sonst erwartet es volle Auflösung (16×16-DCT). Das gilt nur für diese Bibliotheksversion.
   - Qualität, `JDCT_IFAST` und optimierte Tabellen bringen für die Zeit nichts.
   - Nicht gemessen: Mehrfaden-Kompression (2 CPU-Kerne sichtbar), libjpeg-turbo, Hardware-Encoder.

3. **Erledigt:** Lesefehler in `indigo_compute_stretch_params` behoben. Die Bayer-Ebenen mit Zeilen- oder Spalten-Offset (`buffer + width…`, `buffer + 1`) wurden mit der vollen Bildgröße abgetastet und lasen bei ungerader Höhe bis zu 301 Werte hinter den Puffer, bei bestimmten Breiten (`(width - 1) % (2 * sample_by) == 0`) ein Element dahinter bzw. in die nächste Zeile. Das Template bekommt jetzt den Offset getrennt und begrenzt Zeilen und Spalten der Ebene. **Verhaltensänderung:** Bei diesen Bildgrößen ändern sich die Stretch-Parameter minimal, weil nicht mehr Speicherinhalt hinter dem Bild in den Median eingeht. Bei allen anderen Größen (z. B. gerade Höhe) bleibt das Ergebnis bitgleich. Der Test extrahiert jede Ebene explizit und vergleicht mit der Originalmathematik; die Puffer enden vor einer Guard-Page. Gegenprobe: Mit dem alten Code stürzt der Test ab.

**Phase 2: Stretch-Pipeline auf der GPU (umgesetzt, Nano gemessen)**

Umsetzung:
- `indigo_libs/indigo_stretch_kernels.h` (intern): `debayer()` und `stretch()` für CPU und GPU gemeinsam (`INDIGO_HD` = `__host__ __device__`), dazu `indigo_make_stretch_channel()` mit denselben Ausdrücken wie der CPU-Code. Der CPU-Code nutzt die gemeinsamen Funktionen; `debayer` rechnet mit `float`-Division statt `double`. Das ist exakt gleich (Summen < 2^24, durch 2 und 4 exakt, für /3 erschöpfend geprüft: 0 Abweichungen über alle Summen), die 64 Ausgabe-Hashes aller CPU-Funktionen sind vor und nach dem Umbau identisch.
- `indigo_libs/indigo_stretch_cuda.cu` (nicht `indigo_stretch.cu`: gleicher Objektname wie `indigo_stretch.cpp`): Kernel für Debayer+Stretch (8/16 Bit, alle vier Bayer-Muster), reines Debayer (8 Bit), Mono-Stretch (8/16 Bit) und interleaved RGB (24/48 Bit). `indigo_stretch.cpp` ruft sie ab `INDIGO_CUDA_MIN_PIXELS` (262143) auf; bei nicht verfügbarer GPU, kleinem Bild oder CUDA-Fehler läuft der unveränderte CPU-Code. Die Stretch-Parameter (Median/MAD, 17 ms) bleiben auf der CPU.
- YCbCr-Pfad für Farbkameras: Der Kernel erzeugt direkt planares YCbCr 4:2:0 (ein Thread je 2×2-Block, keine Warp-Divergenz) mit libjpegs Festkommakonstanten, Chroma-Mittelung (`h2v2_downsample`, bias 1,2,1,2) und Randauffüllung (Spalten vor der Mittelung, Zeilen als letzte Y- bzw. Chroma-Zeile). `indigo_raw_to_jpeg` schreibt die Ebenen mit `jpeg_write_raw_data`. Ohne GPU oder bei Fehler bleibt der bisherige Pfad. Die Float-DCT wird jetzt wirklich gesetzt (`#ifdef` entfernt).
- Laufzeitschalter `indigo_cuda_set_enabled()` und `INDIGO_CUDA=0` (Umgebungsvariable).

Tests (alle im CUDA- und im `CUDA=no`-Build gelaufen):
- `test_cuda_stretch`: 71 Vergleiche GPU gegen CPU, byte-identisch (alle Funktionen und Bayer-Muster, ungerade Größen), prüft über einen Zähler, dass die GPU wirklich benutzt wurde.
- `test_raw_to_jpeg`: 76 Fälle. `indigo_raw_to_jpeg` mit GPU an und aus liefert **byte-identische JPEG-Dateien** (auch für gerade Höhen mit Rest 6 und 10 Zeilen und für die Größe direkt an der GPU-Schwelle), und beide entsprechen exakt libjpeg mit den Vorschau-Einstellungen.
- Beim Testen gefundene Fehler (behoben): Namenskollision `indigo_stretch.o`; falsche Chroma-Auffüllung in den letzten Zeilen bei gerader Höhe (libjpeg füllt in der bereits unterabgetasteten Ebene auf); wirkungsloser `#ifdef` für die Float-DCT.

Messungen (Nano, 12 MP, Median):

| Vorschau-Konvertierung `indigo_raw_to_jpeg` | Zeit |
|---|---|
| Ausgangszustand (RGGB 16 Bit, CPU, ISLOW, Fancy-Downsampling) | etwa 1900 ms |
| CPU mit Float-DCT und klassischem Chroma | 1113 ms |
| **GPU (YCbCr-Pfad)** | **488 ms** |
| Mono 16 Bit: CPU / GPU | 864 / 718 ms |

Aufteilung der GPU-Konvertierung: Stretch-Parameter 18 ms (CPU), GPU-Schritt 60 ms (Upload 13, Kernel etwa 30, Download etwa 10), libjpeg (DCT + Huffman) rund 400 ms. Der erste GPU-Aufruf kostet zusätzlich etwa 130 ms (Kontext, Puffer). Zum Vergleich der frühere RGB-Kernel: 41 ms allein für den Kernel (Kopieren mit 3-Byte-Schreiben 11 ms, drei IEEE-Divisionen 12 ms, Debayer 17 ms); `__restrict__`, andere Blockformen und Schreiben über Shared Memory brachten nichts, gepinnter Speicher auch nicht (Bandbreite etwa 1,9 GB/s in beide Richtungen).

Was bleibt:
- Mono-Kameras: Die Zeit steckt in der Huffman-Codierung des verrauschten Bildes (4,3 MB Ausgabe), die GPU spart dort nur den Stretch (864 → 718 ms). Hebel wären ein Hardware-Encoder oder eine geringere Qualität, nicht mehr CUDA.
- Farbe: Rest sind DCT und Huffman in libjpeg (etwa 400 ms, ein Thread). Mehrfaden-Kompression (2 CPU-Kerne sichtbar) und Hardware-Encoder sind nicht gemessen.
- Kernel weiter optimieren (Divisionen, Speicherzugriffe) lohnt kaum: Die GPU-Schritte machen nur noch etwa 80 von 488 ms aus.
- Nicht getestet: Orin, JetPack 7.2, echte Kameraframes.

**Phase 3: Statistik für die Agenten (teilweise umgesetzt, Nano gemessen)**

Umgesetzt: `indigo_binarize` (alle 6 `indigo_raw_type`-Varianten) und die Skalierungs-Schleife von `indigo_equalize_bayer_channels` (MONO8/16). Beide reduzieren auf exakte Ganzzahlsummen (Reihenfolge der Addition ändert das Ergebnis nicht), daher bit-genau gleich der CPU:
- `indigo_libs/indigo_raw_utils_cuda.cu` (neu): `binarize_reduce_kernel` (Summe/Quadratsumme über alle Pixel, Blockreduktion + `atomicAdd`), `binarize_threshold_kernel` (Schwellwert anwenden, zählt zusätzlich helle Pixel für die 10-%-Prüfung), `scale_bayer_kernel` (wendet die vier vorher auf der CPU berechneten Skalierungsfaktoren an, ein Thread je 2×2-Block wie die CPU-Schleife).
- Beide Funktionen versuchen zuerst die GPU (ab `INDIGO_CUDA_MIN_PIXELS`), sonst unverändert CPU. Die Stichprobennahme und Berechnung der Skalierungsfaktoren von `equalize_bayer_channels` bleiben auf der CPU (günstig, verstreuter Zugriff).

Test `indigo_test/cuda/test_cuda_raw_utils.c`: 132 Vergleiche (6 `raw_type`, 3 Sigma-Werte, mehrere Größen inklusive Grenzfall an `INDIGO_CUDA_MIN_PIXELS`, konstante Bilder), bit-genau, lief mit `CUDA_LAUNCH_BLOCKING=1` fehlerfrei durch (siehe unten, wichtig nach den gefundenen Fehlern). Alle Tests liefen zusätzlich im `CUDA=no`-Build durch.

**Beim Portieren gefundene Fehler (alle behoben):**
1. **Speicherfehler in `scale_bayer_kernel`:** Der Kernel-Aufruf übergab versehentlich den Host-Zeiger `data` statt des hochgeladenen Device-Zeigers `device_data`. Das ist auf Jetson besonders tückisch: Wegen des gemeinsamen physischen Speichers von CPU und GPU lief das für `MONO8` bei mehreren Bildgrößen unbemerkt durch, brach aber bei `MONO16` mit `unspecified launch failure` ab und beschädigte danach den CUDA-Kontext für den Rest des Prozesses (alle folgenden Aufrufe fielen still auf die CPU zurück, was Testergebnisse verfälschte). Gefunden erst mit `CUDA_LAUNCH_BLOCKING=1`, das jeden Kernel-Start synchron macht und den Fehler an der richtigen Stelle meldet. **Lehre für die weiteren Phasen: jeder neue GPU-Test läuft ab jetzt einmal mit `CUDA_LAUNCH_BLOCKING=1`, ein bestandener Test ohne dieses Flag reicht nicht.**
2. **Vorhandener Fehler in `indigo_binarize` (MONO16, RGB48):** `sum_sq += value * value;` rechnet `value * value` in 32 Bit; bei 16-Bit-Werten über 46340 läuft das über (`65535² ≈ 4,3 × 10⁹ > INT32_MAX`) und ergibt durch den Zahlenüberlauf ein falsches, plattformabhängiges Ergebnis. Das ist unabhängig von CUDA und betrifft jedes reale 16-Bit-Bild mit hellen Pixeln. Behoben durch Aufweitung auf `long` vor der Multiplikation (`(long)value * value`), für alle sechs `raw_type`-Zweige einheitlich angewendet.
3. **Rundungsunterschied bei RGB-Schwellwerten:** Für RGB24/RGBA32/ABGR32/RGB48 vergleicht die CPU absichtlich die *Summe* der drei Kanäle mit `Schwelle × 3` (nicht den gerundeten Mittelwert mit der Schwelle), um die Rundung der Ganzzahldivision durch 3 zu vermeiden. Der erste GPU-Kernel nutzte den gerundeten Mittelwert und wich dadurch an Werten nahe der Schwelle ab. Behoben durch eine zweite, dem CPU-Code entsprechende Vergleichsfunktion in der Schwellwert-Kernel.

Messung (Nano, 12 MP):

| Funktion | CPU | GPU |
|---|---|---|
| `indigo_binarize` | 70 ms | 26–35 ms |
| `indigo_equalize_bayer_channels` | 45–49 ms | 30–45 ms |

Nicht umgesetzt (zurückgestellt, mit Begründung):
- **`indigo_contrast` / `indigo_stddev_*`** (8 Varianten: MONO8/16, RGB24/48, je maskiert/unmaskiert): Die zweite Passe summiert `(x - mean)²` als `double` in fester CPU-Reihenfolge. Diese Summe ist *nicht* automatisch exakt wie bei den Ganzzahlsummen oben – eine GPU-Reduktion in anderer Reihenfolge liefert im Allgemeinen ein anderes Ergebnis in den letzten Bits. Eine bit-genaue Portierung bräuchte eine Umformulierung auf `Σx²/N − mean²` (mathematisch gleich, ganzzahlig exakt berechenbar, aber eine numerisch andere Rechenreihenfolge als heute) – das ändert das Verhalten aller 8 CPU-Funktionen und braucht eine eigene Freigabe, ähnlich der Korrektur der Stretch-Parameter in Phase 0.
- **Dabei gefunden, nicht behoben:** `indigo_stddev_rgb48` (unmaskiert) hat in der ersten Passe (Mittelwertberechnung) einen Indexierungsfehler – `i = y * width + x` statt `i = (y * width + x) * 3` wie in Passe 2 und wie bei `indigo_stddev_rgb24`. Das liest an einer falschen Stelle im Puffer und verfälscht `indigo_contrast` für unmaskierte RGB48-Bilder (nur der maskierte Pfad ist betroffen-frei). Unabhängig von CUDA, nicht behoben, da außerhalb des Auftrags „GPU-Beschleunigung“.
- **`indigo_update_saturation_mask`:** Die Erkennung ist bit-exakt möglich (gleiche Bausteine wie `stddev`), aber das Löschen der Maskenbox pro erkanntem gesättigtem Pixel ist auf der CPU bereits O(Erkennungen × Boxfläche) und damit schwer sauber zu parallelisieren; eng gekoppelt an `indigo_contrast` (maskierter Pfad). Zurückgestellt.

Empfehlung: Vor einer GPU-Portierung von `indigo_contrast`/`stddev` zuerst entscheiden, ob (a) die Σx²-Umformulierung mit dokumentierter Verhaltensänderung gewünscht ist und (b) der RGB48-Indexierungsfehler separat behoben werden soll.

**Phase 4: Bahtinov**
- Zuerst die Sinus-Tabelle auf der CPU, danach Hough mit Atomics und `skeletonize` auf der GPU, falls noch nötig.

**Phase 5: Optional**
- Donuts: `clear_hot_pixel_*` auf der CPU optimieren. Vollbild dauert 3,3 s, mit Region 20 ms, also nur relevant, wenn Aufrufer große Regionen nutzen (nicht geprüft).
- `indigo_make_psf_map` nach einer Messung.

## 5. Offene Entscheidungen

1. Standard bleibt `CUDA=no`. Falls Auto-Erkennung gewünscht ist, kann `CUDA=auto` ergänzt werden.
2. Die Kombination „Jetson und Orin mit JetPack 7.2“ lässt sich für ein Nano nicht herstellen (JetPack 4.x ist dort das Ende). Klären, welche Geräte tatsächlich mit welcher JetPack-Version getestet werden.
3. Freigabe für Phase 1 (`find_stars`-Umbau, JPEG-Messung). Phase 1 ändert bestehende Funktionen der Bibliothek, auch wenn es die Ergebnisse nicht ändern soll.
