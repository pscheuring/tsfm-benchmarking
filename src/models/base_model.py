from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def _init_model(self):
        """Initialize the model."""
        pass

    @abstractmethod
    def prepare_zero_shot(self):
        """Prepare the model for zero-shot prediction."""
        pass

    # TODO: Implement few-shot and full-shot methods for all models
    # @abstractmethod
    # def fit_few_shot(self, series_df: pd.DataFrame):
    #     """Fit model for few-shot prediciton."""
    #     pass

    # @abstractmethod
    # def fit_full_shot(self, series_df: pd.DataFrame):
    #     """Fit model for full-shot prediciton."""
    #     pass

    @abstractmethod
    def predict(self):
        """Return a forecast vector of the specified horizon length."""
        pass
