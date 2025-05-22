from scikeras.wrappers import KerasClassifier

class CustomKerasClassifier(KerasClassifier):
    def __init__(self, preprocessing_layers, **kwargs):
        self.preprocessing_layers = preprocessing_layers
        super().__init__(model=self._build_model, **kwargs)

    def _build_model(self):
        from src.models.neural_network import build_model
        return build_model(self.preprocessing_layers)
