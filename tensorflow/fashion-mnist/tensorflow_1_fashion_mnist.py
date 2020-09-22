
import bentoml
import tensorflow as tf

from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.adapters import TfTensorInput

tf.compat.v1.enable_eager_execution() # required

FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@bentoml.env(pip_dependencies=['tensorflow', 'numpy', 'pillow'])
@bentoml.artifacts([TensorflowSavedModelArtifact('trackable')])
class FashionMnistTensorflow(bentoml.BentoService):

    @bentoml.api(input=TfTensorInput(), batch=True)
    def predict(self, inputs):
        loaded_func = self.artifacts.trackable.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        pred = loaded_func(x=inputs, train_mode=tf.constant(False))
        output = pred['prediction']
        return [FASHION_MNIST_CLASSES[c] for c in output]
