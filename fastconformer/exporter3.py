"""
Troisième passe d'export (2026-10-15) : le modèle du juge, en un seul graphe.

Entrée `audio` [1, N] (float32, 16 kHz, mono, entre -1 et 1) ; sortie `ids`
[1, T/8] (int64) : le jeton le plus probable de chaque trame de 80 ms, blanc
compris (1024). Le reste (fusionner les répétitions, ôter les blancs, recoller
les morceaux de mots) se fait en JavaScript.

Le spectre est dans le graphe : la transformée de Fourier y est une
convolution de 514 filtres (cosinus et sinus fenêtrés), rien qui dépende d'un
opérateur rare — Hermes, sans compilateur à la volée, mettrait des secondes à
la calculer. Il reste en float32 ; seul l'encodeur + la tête CTC sont quantifiés,
« MatMul seul » (la recette par défaut décale l'encodeur de 30 %).

Usage : python -I -X utf8 exporter3.py <dossier modeles> <wav>…
Écrit <dossier>/juge/ : front.onnx, juge-<recette>.onnx, tokens.json, controle3.json.
"""
import json
import os
import sys
import time
import wave

import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import OmegaConf, open_dict

out, *wavs = sys.argv[1:]
work = os.path.join(out, "_nemo")
dest = os.path.join(out, "juge")
os.makedirs(dest, exist_ok=True)

cfg = OmegaConf.load(os.path.join(work, "model_config.yaml"))
with open_dict(cfg):
    for key in list(cfg.tokenizer.keys()):
        value = cfg.tokenizer[key]
        if isinstance(value, str) and value.startswith("nemo:"):
            cfg.tokenizer[key] = os.path.join(work, value[len("nemo:"):])
    cfg.tokenizer.dir = work
    for key in ("train_ds", "validation_ds", "test_ds"):
        if key in cfg:
            cfg[key] = None

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel  # noqa: E402

weights = [name for name in os.listdir(work) if name.endswith(".ckpt")][0]
state = torch.load(os.path.join(work, weights), map_location="cpu", weights_only=True)
model = EncDecHybridRNNTCTCBPEModel(cfg=cfg)
model.load_state_dict(state, strict=True)
model.eval()
feat = model.preprocessor.featurizer
feat.dither = 0.0
feat.pad_to = 0
print("préprocesseur :", {k: getattr(feat, k, None) for k in ("n_fft", "hop_length", "win_length", "preemph", "mag_power", "log_zero_guard_value", "normalize", "exact_pad", "stft_pad_amount")}, flush=True)


def read_wav(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32") / 32768
    return data.copy()


# ---------------------------------------------------------------------------
# 1. LE SPECTRE, EN CONVOLUTION
# ---------------------------------------------------------------------------

N_FFT, HOP = 512, 160


class Front(torch.nn.Module):
    def __init__(self):
        super().__init__()
        window = feat.window.detach().float()  # Hann de 400 points
        full = torch.zeros(N_FFT)
        off = (N_FFT - window.shape[0]) // 2
        full[off : off + window.shape[0]] = window
        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64)[:, None]
        angle = 2 * torch.pi * k * n / N_FFT
        cos = (torch.cos(angle) * full.double()).float()
        sin = (-torch.sin(angle) * full.double()).float()
        self.register_buffer("basis", torch.cat([cos, sin], 0)[:, None, :])  # [514, 1, 512]
        self.register_buffer("fb", feat.fb.detach().float()[0])  # [80, 257]
        self.preemph = float(feat.preemph)
        self.guard = float(feat.log_zero_guard_value)

    def forward(self, audio):
        x = torch.cat([audio[:, :1], audio[:, 1:] - self.preemph * audio[:, :-1]], 1)
        x = torch.nn.functional.pad(x[:, None, :], (N_FFT // 2, N_FFT // 2))
        spec = torch.nn.functional.conv1d(x, self.basis, stride=HOP)  # [1, 514, F]
        half = N_FFT // 2 + 1
        power = spec[:, :half] ** 2 + spec[:, half:] ** 2
        mel = torch.log(torch.matmul(self.fb, power) + self.guard)  # [1, 80, ⌊N/160⌋ + 1]
        # NeMo ne garde que ⌊N/160⌋ trames : la dernière du STFT centré tombe.
        mel = mel[:, :, :-1]
        count = torch._shape_as_tensor(mel)[2]
        mean = mel.mean(dim=2, keepdim=True)
        var = ((mel - mean) ** 2).sum(dim=2, keepdim=True) / (count - 1).clamp(min=1).to(mel.dtype)
        feats = (mel - mean) / (torch.sqrt(var) + 1e-5)
        return feats, count.reshape(1)


front = Front().eval()
front_path = os.path.join(dest, "front.onnx")
dummy = torch.from_numpy(read_wav(wavs[0]))[None]
torch.onnx.export(
    front, (dummy,), front_path, input_names=["audio"], output_names=["audio_signal", "length"],
    dynamic_axes={"audio": {1: "samples"}, "audio_signal": {2: "frames"}}, opset_version=17, dynamo=False,
)
report = {}
errs = []
sess = ort.InferenceSession(front_path)
for path in wavs:
    audio = read_wav(path)
    with torch.no_grad():
        ref, rlen = model.preprocessor(input_signal=torch.from_numpy(audio)[None], length=torch.tensor([audio.shape[0]]))
    got, glen = sess.run(None, {"audio": audio[None]})
    T = int(rlen[0])
    assert int(glen[0]) == T, (path, int(glen[0]), T)
    errs.append(float(np.abs(ref[:, :, :T].numpy() - got[:, :, :T]).max()))
report["spectre_ecart_max"] = max(errs)
print("spectre : écart max à NeMo", report["spectre_ecart_max"], flush=True)

# ---------------------------------------------------------------------------
# 2. ENCODEUR + CTC, QUANTIFIÉS, PUIS COLLÉS AU SPECTRE
# ---------------------------------------------------------------------------

from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402

hors = os.path.join(out, "hors")
ctc32 = os.path.join(hors, "ctc.onnx")
graph = onnx.load(ctc32, load_external_data=False)
pointwise = []
for node in graph.graph.node:
    if node.op_type != "Conv":
        continue
    kernel = next((list(a.ints) for a in node.attribute if a.name == "kernel_shape"), None)
    group = next((a.i for a in node.attribute if a.name == "group"), 1)
    if kernel == [1] and group == 1:
        pointwise.append(node.name)
print("convolutions ponctuelles :", len(pointwise), flush=True)

quantized = {"matmul": os.path.join(hors, "ctc.int8.onnx")}
q_pw = os.path.join(dest, "ctc.int8-pw.onnx")
if not os.path.exists(q_pw):
    names = [n.name for n in graph.graph.node if n.op_type == "MatMul"] + pointwise
    quantize_dynamic(ctc32, q_pw, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Conv"], nodes_to_quantize=names)
quantized["matmul-pw"] = q_pw

front_model = onnx.load(front_path)


def glue(ctc_path, target):
    ctc = onnx.load(ctc_path)
    ctc.ir_version = front_model.ir_version = min(ctc.ir_version, front_model.ir_version)
    merged = onnx.compose.merge_models(
        front_model, ctc, io_map=[("audio_signal", "audio_signal"), ("length", "length")], prefix1="front_", prefix2=""
    )
    # Le jeton le plus probable de chaque trame.
    logp = merged.graph.output[0].name
    merged.graph.node.append(onnx.helper.make_node("ArgMax", [logp], ["ids"], axis=2, keepdims=0, name="argmax_ids"))
    del merged.graph.output[:]
    merged.graph.output.append(onnx.helper.make_tensor_value_info("ids", onnx.TensorProto.INT64, [1, "trames"]))
    for node in merged.graph.node:
        node.input[:] = ["audio" if name == merged.graph.input[0].name else name for name in node.input]
    merged.graph.input[0].name = "audio"
    merged.producer_name = "ahzab"
    meta = {
        "source": "mohammed/fastconformer-quran-ar (CC-BY-4.0), phase3_full_wer0.0014.nemo, sur NVIDIA FastConformer (CC-BY-4.0)",
        "modifications": "export ONNX hors flux, tête CTC, spectre NeMo en convolution, ArgMax ; int8 dynamique",
        "blank": "1024",
    }
    for key, value in meta.items():
        merged.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))
    onnx.checker.check_model(merged)
    onnx.save(merged, target)
    return target


# La référence : spectre NeMo + CTC f32, le glouton fait ici.
ref_sess = ort.InferenceSession(ctc32)
refs = []
for path in wavs:
    audio = read_wav(path)
    with torch.no_grad():
        feats, flen = model.preprocessor(input_signal=torch.from_numpy(audio)[None], length=torch.tensor([audio.shape[0]]))
    lp = ref_sess.run(None, {"audio_signal": feats.numpy(), "length": flen.numpy()})[0]
    refs.append(lp.argmax(-1)[0])

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


def collapse(ids):
    outl, prev = [], -1
    for i in ids:
        if i != prev and i != 1024:
            outl.append(int(i))
        prev = i
    return outl


for recipe, path in quantized.items():
    target = glue(path, os.path.join(dest, f"juge-{recipe}.onnx"))
    sess = ort.InferenceSession(target, opts)
    same_frames = total_frames = same_text = 0
    secs = ms = 0.0
    for path_wav, ref in zip(wavs, refs):
        audio = read_wav(path_wav)
        started = time.perf_counter()
        ids = sess.run(None, {"audio": audio[None]})[0][0]
        ms += (time.perf_counter() - started) * 1000
        secs += audio.shape[0] / 16000
        n = min(len(ids), len(ref))
        same_frames += int((ids[:n] == ref[:n]).sum())
        total_frames += len(ref)
        same_text += int(collapse(ids) == collapse(ref))
    report[recipe] = {
        "taille": os.path.getsize(target),
        "trames_identiques": round(same_frames / total_frames, 4),
        "textes_identiques": f"{same_text}/{len(wavs)}",
        "ms_par_s": round(ms / secs, 1),
    }
    print(recipe, report[recipe], flush=True)

with open(os.path.join(dest, "tokens.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(list(model.tokenizer.vocab), f, ensure_ascii=False)
with open(os.path.join(dest, "controle3.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print("fini", flush=True)
