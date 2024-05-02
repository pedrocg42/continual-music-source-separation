import numpy as np
from madre.base.container.register import register
from madre.base.data.dataset.dataset_item import DatasetItem
from madre.evaluation.evaluators.evaluator import Evaluator


@register()
class MusicSourceSeparationEvaluator(Evaluator):
    def __init__(self, apply_softmax: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.apply_softmax = apply_softmax

    def execute(
        self,
        dataset_item: DatasetItem,
        **kwargs,
    ):
        """
            Compute non-genuine probability for the input features dict items.
        :param features_dict: Dict[str, Any]
            Features dict following the format below:
                * { "frame_key": Any}
                    or,
                * {"frame_hey": {"features_key": Any}
        :return: float
            Score!
        """
        if len(dataset_item.features) > 1:
            raise RuntimeError(
                "BW Evaluator Error: Found features for several frames, while evaluator expects only one"
            )
        frame_features = list(dataset_item.features.values())[0]
        _, feature = next(iter(frame_features.items()))
        preds = self.softmax(feature) if self.apply_softmax else feature
        dataset_item.score = 1.0 - preds[0]

    def execute_batch(self, features: np.ndarray) -> np.ndarray:
        return 1.0 - features[:, 0]

    @staticmethod
    def softmax(x):
        max = np.max(x, axis=0, keepdims=True)  # returns max of each row and keeps same dims
        e_x = np.exp(x - max)  # subtracts each row with its max value
        sum = np.sum(e_x, axis=0, keepdims=True)  # returns sum of each row and keeps same dims
        f_x = e_x / sum
        return f_x
