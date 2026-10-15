"""
Dernière main au modèle du juge (2026-10-15) : la sortie `ids` passe en int32
(Hermes lit mal les BigInt64Array, et 1 025 jetons tiennent large), puis les
empreintes pour la release et `lib/fastConformer.ts`.

Usage : python -I -X utf8 finir.py <juge-recette.onnx> <sortie.onnx> <wav>…
"""
import hashlib
import os
import sys
import wave

import numpy as np
import onnx
import onnxruntime as ort

src, dst, *wavs = sys.argv[1:]
model = onnx.load(src)
graph = model.graph
assert graph.output[0].name == "ids"
for node in graph.node:
    if node.output and node.output[0] == "ids":
        node.output[0] = "ids64"
graph.node.append(onnx.helper.make_node("Cast", ["ids64"], ["ids"], to=onnx.TensorProto.INT32, name="cast_ids"))
del graph.output[:]
graph.output.append(onnx.helper.make_tensor_value_info("ids", onnx.TensorProto.INT32, [1, "trames"]))
onnx.checker.check_model(model)
onnx.save(model, dst)


def read_wav(path):
    with wave.open(path) as w:
        return (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype("float32") / 32768).copy()


a = ort.InferenceSession(src)
b = ort.InferenceSession(dst)
for path in wavs:
    audio = read_wav(path)[None]
    x = a.run(None, {"audio": audio})[0]
    y = b.run(None, {"audio": audio})[0]
    assert y.dtype == np.int32 and np.array_equal(x.astype(np.int32), y), path
print("sortie int32 identique sur", len(wavs), "extraits")

data = open(dst, "rb").read()
print("fichier ", os.path.basename(dst))
print("taille  ", len(data))
print("md5     ", hashlib.md5(data).hexdigest())
print("sha256  ", hashlib.sha256(data).hexdigest())
