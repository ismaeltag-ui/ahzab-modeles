"""
Export ONNX de mohammed/fastconformer-quran-ar (CC-BY-4.0), hors du dépôt.

- ouvre le .nemo en archive, et lit les poids en `weights_only=True` (aucun
  pickle exécuté) ;
- écrit les sorties de contrôle : configuration du préprocesseur, spectre NeMo
  d'un extrait (pour vérifier le spectre du banc Node), transcriptions NeMo ;
- exporte hors flux (attention bilatérale) puis causal [128, 0] (la recette de
  la fiche du modèle), chaque fois le transducteur (encodeur + decoder_joint) et
  la tête CTC (un graphe), en f32 puis int8 dynamique.

Usage : python -I exporter.py <fichier.nemo> <dossier modeles> <wav de contrôle>…
"""
import json
import os
import sys
import tarfile
import time

import torch
from omegaconf import OmegaConf, open_dict

src, out, *wavs = sys.argv[1:]
work = os.path.join(out, "_nemo")
os.makedirs(work, exist_ok=True)
if not os.path.exists(os.path.join(work, "model_config.yaml")):
    with tarfile.open(src) as archive:
        archive.extractall(work, filter="data")
print("archive :", sorted(os.listdir(work)))

cfg = OmegaConf.load(os.path.join(work, "model_config.yaml"))


def local(value):
    if isinstance(value, str) and value.startswith("nemo:"):
        return os.path.join(work, value[len("nemo:"):])
    return value


with open_dict(cfg):
    for key in list(cfg.tokenizer.keys()):
        cfg.tokenizer[key] = local(cfg.tokenizer[key])
    cfg.tokenizer.dir = work
    for key in ("train_ds", "validation_ds", "test_ds"):
        if key in cfg:
            cfg[key] = None
print("tokenizer :", OmegaConf.to_container(cfg.tokenizer))
print("preprocessor :", json.dumps(OmegaConf.to_container(cfg.preprocessor)))
print("encoder :", {k: cfg.encoder.get(k) for k in ("self_attention_model", "att_context_size", "conv_context_size", "subsampling", "subsampling_factor", "n_layers", "d_model")})

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel  # noqa: E402

model = EncDecHybridRNNTCTCBPEModel(cfg=cfg)
weights = [name for name in os.listdir(work) if name.endswith(".ckpt")][0]
state = torch.load(os.path.join(work, weights), map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(state, strict=False)
print("poids : manquants", missing, "inattendus", unexpected)
model.eval()
model.preprocessor.featurizer.dither = 0.0
model.preprocessor.featurizer.pad_to = 0

# Les jetons, au format tokens.txt (id 1024 = blanc).
vocab = model.tokenizer.vocab
os.makedirs(os.path.join(out, "hors"), exist_ok=True)
os.makedirs(os.path.join(out, "causal"), exist_ok=True)
for sub in ("hors", "causal"):
    with open(os.path.join(out, sub, "tokens.txt"), "w", encoding="utf-8") as f:
        for i, piece in enumerate(vocab):
            f.write(f"{piece} {i}\n")
        f.write(f"<blk> {len(vocab)}\n")

controle = {"preprocessor": OmegaConf.to_container(cfg.preprocessor), "clips": []}


def read_wav(path):
    import wave

    import numpy as np

    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32") / 32768
    return torch.from_numpy(data.copy())


def decode(mode):
    """Transcription NeMo directe des wav de contrôle, tête `mode`."""
    texts = []
    for path in wavs:
        audio = read_wav(path)
        with torch.no_grad():
            feats, flen = model.preprocessor(input_signal=audio[None], length=torch.tensor([audio.shape[0]]))
            enc, elen = model.encoder(audio_signal=feats, length=flen)
            if mode == "ctc":
                logp = model.ctc_decoder(encoder_output=enc)
                hyp = model.ctc_decoding.ctc_decoder_predictions_tensor(logp, decoder_lengths=elen, return_hypotheses=False)
            else:
                hyp = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=elen, return_hypotheses=False)
        h = hyp[0] if isinstance(hyp, (list, tuple)) else hyp
        h = h[0] if isinstance(h, (list, tuple)) else h
        texts.append(getattr(h, "text", h))
    return texts


for path in wavs:
    audio = read_wav(path)
    with torch.no_grad():
        feats, flen = model.preprocessor(input_signal=audio[None], length=torch.tensor([audio.shape[0]]))
    controle["clips"].append({"wav": path, "T": int(flen[0]), "feats": feats[0, :, : int(flen[0])].flatten().tolist()})


def export(sub):
    folder = os.path.join(out, sub)
    started = time.time()
    model.set_export_config({"decoder_type": "rnnt", "cache_support": False})
    model.export(os.path.join(folder, "model.onnx"))
    os.replace(os.path.join(folder, "encoder-model.onnx"), os.path.join(folder, "encoder.onnx"))
    os.replace(os.path.join(folder, "decoder_joint-model.onnx"), os.path.join(folder, "decoder_joint.onnx"))
    model.set_export_config({"decoder_type": "ctc", "cache_support": False})
    model.export(os.path.join(folder, "ctc.onnx"))
    model.set_export_config({"decoder_type": "rnnt"})
    print(sub, "exporté en", round(time.time() - started), "s :", sorted(os.listdir(folder)))


def quantize(sub):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    folder = os.path.join(out, sub)
    for name in ("encoder", "decoder_joint", "ctc"):
        source = os.path.join(folder, f"{name}.onnx")
        target = os.path.join(folder, f"{name}.int8.onnx")
        quantize_dynamic(source, target, weight_type=QuantType.QInt8)
        print(" ", target, os.path.getsize(target))


# 1. Hors flux.
controle["hors"] = {"rnnt": decode("rnnt"), "ctc": decode("ctc")}
print("NeMo hors flux :", json.dumps(controle["hors"], ensure_ascii=False))
export("hors")
quantize("hors")

# 2. Causal [128, 0] : le même poids, reconstruit avec des convolutions causales
# (CausalConv1D, posées à la construction) puis l attention locale [128, 0].
with open_dict(cfg):
    cfg.encoder.conv_context_size = "causal"
model = EncDecHybridRNNTCTCBPEModel(cfg=cfg)
missing, unexpected = model.load_state_dict(state, strict=False)
print("poids (causal) : manquants", missing, "inattendus", unexpected)
model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[128, 0])
model.eval()
model.preprocessor.featurizer.dither = 0.0
model.preprocessor.featurizer.pad_to = 0
controle["causal"] = {"rnnt": decode("rnnt"), "ctc": decode("ctc")}
print("NeMo causal :", json.dumps(controle["causal"], ensure_ascii=False))
export("causal")
quantize("causal")

with open(os.path.join(out, "controle.json"), "w", encoding="utf-8") as f:
    json.dump(controle, f, ensure_ascii=False)
print("fini")
