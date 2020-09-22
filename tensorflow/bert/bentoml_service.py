
import bentoml
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List

from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import DataframeInput


CLASSES = ["negative","positive"]
max_seq_len = 128

try:
    tf.config.set_visible_devices([], 'GPU')  # disable GPU, required when served in docker
except:
    pass


@bentoml.env(pip_dependencies=['tensorflow', 'bert-for-tf2'])
@bentoml.artifacts([TensorflowSavedModelArtifact('model'), PickleArtifact('tokenizer')])
class Service(bentoml.BentoService):

    def tokenize(self, inputs: pd.DataFrame):
        tokenizer = self.artifacts.tokenizer
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_numpy()[:, 0].tolist()
        else:
            inputs = inputs.tolist()  # for predict_clipper
        pred_tokens = map(tokenizer.tokenize, inputs)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))
        pred_token_ids = map(lambda tids: tids + [0] * (max_seq_len - len(tids)), pred_token_ids)
        pred_token_ids = tf.constant(list(pred_token_ids), dtype=tf.int32)
        return pred_token_ids

    @bentoml.api(input=DataframeInput(), mb_max_latency=3000, mb_max_batch_size=20, batch=True)
    def predict(self, inputs: pd.DataFrame) -> List[str]:
        model = self.artifacts.model
        pred_token_ids = self.tokenize(inputs)
        res = model(pred_token_ids).numpy().argmax(axis=-1)
        return [CLASSES[i] for i in res]
