from classification_models_3D.tfkeras import Classifiers
from numpy import interp
import keras

from .settings import MODEL_PATHES



class EstimationModelFactory():
    def get_estimation_model(self, model_type: str):
        model_path = MODEL_PATHES.get(model_type, None)
        if model_path is None:
            ValueError('No such model {model_type}')
        return EstimationModel(model_path)

    def estimation_model_generator(self, model_types: list[str]):
        for model_type in model_types:
            yield self.get_estimation_model(model_type)


class EstimationModel():
    def __init__(self, path_to_pretrained):
        self.model = keras.models.load_model(path_to_pretrained)

    def get_input_shape(self):
        return self.model.layers[0].input_shape[0][1:]

    def predict(self, batch):
        batch_proba = self.model.predict(batch)
        return self._get_estimates(batch_proba)

    def _get_estimates(self, batch_proba):
        estimates = []
        n = len(batch_proba[0])
        max_estimate = (2*n-1)/(2*n)
        min_estimate = (1)/(2*n)
        for proba in batch_proba:
            sum_ = 0
            for i, prob in enumerate(proba):
                sum_ += ((2*(i+1)-1)/(2*n))*prob
            estimates.append(interp(sum_, [min_estimate, max_estimate], [0, 1]))
        return estimates
