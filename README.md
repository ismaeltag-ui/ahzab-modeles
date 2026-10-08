# ahzab-modeles

Modèles téléchargés à la demande par l'application **Ahzab** (révision du Coran), qui les
fait tourner **sur l'appareil**. Rien de ce que l'application entend ne quitte le téléphone.

## `whisper-base-ar-quran-v1`

Le « juge » de la reconnaissance : quand le suivi de la récitation bloque sur un mot, ce
modèle relit les dernières secondes de la voix et dit si le mot a été prononcé en entier.

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
