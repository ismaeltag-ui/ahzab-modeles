# ahzab-modeles

Modèles téléchargés à la demande par l'application **Ahzab** (révision du Coran), qui les
fait tourner **sur l'appareil**. Rien de ce que l'application entend ne quitte le téléphone.

## `fastconformer-quran-ar-juge-v1`

Le « juge » de la reconnaissance depuis la version qui l'embarque : quand le suivi de la
récitation bloque sur un mot, ce modèle relit les dernières secondes de la voix et dit si le
mot a été prononcé en entier. Il remplace `whisper-base-ar-quran-v1` (plus bas), gardé en secours.

| Fichier | Format | Taille | md5 | sha256 |
| --- | --- | --- | --- | --- |
| `fastconformer-quran-ar-juge-int8.onnx` | ONNX (opset 17), int8 dynamique | 175 031 540 octets | `2ad63f9037eabcaa62034a2c34c20dd0` | `0dbb048a055c4f040baa35ac60a6453478fb1e855816d62d5d5714b4fcfa8264` |

Un seul graphe : entrée `audio` `[1, N]` (float32, 16 kHz, mono, entre -1 et 1), sortie `ids`
`[1, T]` (int32) : l'indice du jeton le plus probable de chaque trame de 80 ms, le blanc
valant `1024`. Il reste à fusionner les répétitions, ôter les blancs et recoller les morceaux
de mots (vocabulaire SentencePiece de 1 024 jetons du modèle d'origine, `▁` ouvrant un mot).

### Origine

- **Modèle** : [`mohammed/fastconformer-quran-ar`](https://huggingface.co/mohammed/fastconformer-quran-ar),
  révision `60973918b40c5fcdc0e93310c13a4a4661525142`, fichier `phase3_full_wer0.0014.nemo`
  (sha256 `dc3557eea22b25ab5c7af2d8cf01636efd415c4ed417418156e5895056c705d5`), sous licence
  **CC-BY-4.0** ([LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0)).
- **Modèle de base** : [`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)
  de NVIDIA, révision `7f32349d952f42a28dce979ba73270aa2bbdfa89`, sous licence **CC-BY-4.0**.
- **Modifications** (voir [NOTICE](NOTICE)) : export ONNX de la tête CTC seule (attention
  bilatérale, hors flux) ; quantification int8 dynamique des seuls `MatMul` ; le spectre de
  NeMo (préaccentuation, transformée de Fourier, filtres mel, logarithme, normalisation par
  filtre) recalculé en opérateurs ONNX standard et placé devant l'encodeur ; un `ArgMax` puis
  un `Cast` en int32 à la sortie. Les poids n'ont pas été réentraînés.

### Refaire le fichier

Python 3.12, NeMo 3.0.0, torch 2.14.1 (CPU), onnx 1.23.2, onnxruntime 1.31.0. Les scripts
sont dans [`fastconformer/`](fastconformer/).

```bash
# 1. Export ONNX hors flux, transducteur et tête CTC (poids lus en weights_only)
python -I exporter.py phase3_full_wer0.0014.nemo modeles controle.wav
# 2. Quantification int8 des MatMul de la tête CTC
python -I -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('modeles/hors/ctc.onnx', 'modeles/hors/ctc.int8.onnx', weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'])"
# 3. Le spectre en ONNX, vérifié contre NeMo (écart maximal 3e-5), collé devant l'encodeur
python -I -X utf8 exporter3.py modeles controle.wav
# 4. Sortie en int32
python -I -X utf8 finir.py modeles/juge/juge-matmul.onnx fastconformer-quran-ar-juge-int8.onnx controle.wav
```

## `whisper-base-ar-quran-v1`

Le juge précédent, gardé en secours : Whisper affiné sur le Coran, que l'application fait
tourner quand FastConformer manque.

| Fichier | Format | Taille | md5 | sha256 |
| --- | --- | --- | --- | --- |
| `ggml-tarteel-base-ar-quran-q5_0.bin` | ggml, quantifié q5_0 | 55 295 433 octets | `3cb68fa992c3c44e59c535ed1a3efc37` | `1eb6255686651cb496e7ef52bc15c4f4f3fe197af6eea40eeb60e9f212a96950` |

### Origine

- **Modèle** : [`tarteel-ai/whisper-base-ar-quran`](https://huggingface.co/tarteel-ai/whisper-base-ar-quran),
  révision `5c3c53fdf9272c4f6ee0bee09a1e5a4a615ee25c`, sous licence **Apache 2.0** ([LICENSE](LICENSE)).
- **Modèle de base** : [Whisper](https://github.com/openai/whisper) d'OpenAI (`whisper-base`), sous
  licence **MIT** ([LICENSE-whisper](LICENSE-whisper)).
- **Modifications** : conversion au format de [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
  (`models/convert-h5-to-ggml.py`, v1.9.2, `n_text_ctx` pris dans `max_target_positions`), puis
  quantification en `q5_0` (`whisper-quantize`, v1.9.2). Les poids n'ont pas été réentraînés.
  Voir [NOTICE](NOTICE).

### Refaire le fichier

```bash
# 1. Conversion (Python, transformers, torch ; dépôt openai/whisper pour mel_filters.npz)
python convert-h5-to-ggml.py <snapshot tarteel-ai/whisper-base-ar-quran> <dépôt openai/whisper> <sortie>
# 2. Quantification
whisper-quantize ggml-model.bin ggml-tarteel-base-ar-quran-q5_0.bin q5_0
```

Le fichier `f16` intermédiaire a pour sha256
`aaebca10e2a65a6dda073a6b3ac59900f91887628acf872c565a0198ee850ead`.
