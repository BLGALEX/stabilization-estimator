from .settings import MODEL_PATHES

import keras


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
        batch_proba = self.model.predict_proba(batch)
        return self.get_estimation(batch_proba)

    def get_estimation(self, batch_proba):
        # TODO: сделать расчет оценки
        return [0]*len(batch_proba)
