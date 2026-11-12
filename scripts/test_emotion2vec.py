import os
import torch
import numpy as np
from funasr import AutoModel

os.environ["HF_HUB_DISABLE_XET"] = "1"

model = AutoModel(
    model="emotion2vec/emotion2vec_base",
    hub="hf",
    device="cuda:0",
    disable_update=True
)

wav_path = os.path.join(
    model.model_path,
    "example",
    "test.wav"
)

result = model.generate(
    input=wav_path,
    granularity="utterance",
    extract_embedding=True
)

embedding = np.asarray(result[0]["feats"])

print("Embedding shape:", embedding.shape)
print("Feature dtype:", embedding.dtype)
print("Contains NaN:", np.isnan(embedding).any())
print("Feature extraction successful!")
