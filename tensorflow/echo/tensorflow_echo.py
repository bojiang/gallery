
import bentoml
import tensorflow as tf
import numpy as np

from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.adapters import TfTensorInput


@bentoml.env(pip_dependencies=['tensorflow', 'numpy', 'scikit-learn'])
@bentoml.artifacts([TensorflowSavedModelArtifact('model')])
class EchoServicer(bentoml.BentoService):
    @bentoml.api(input=TfTensorInput(), batch=True)
    def predict(self, tensor):
        outputs = self.artifacts.model(tensor)
        return outputs
