from typing import Optional

from pydantic import BaseModel
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class MinShifter:

    def __init__(self):
        self.min: int = 0

    def fit(self, X: np.ndarray):
        self.min = X.min()

    def transform(self, X: np.ndarray):
        return X - self.min

class DatasetModel(BaseModel):

    numerical_feature_names: list[str]
    categorical_feature_names: list[str]
    numerical_features: np.ndarray
    categorical_features: np.ndarray
    target: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True

class PreProcessingPipeline:

    def __init__(
            self,
            dataset: pd.DataFrame,
            target_name: str,
            categorical_features: list[str],
            numerical_features: list[str],
            numerical_scaler: StandardScaler = StandardScaler,
            one_hot_encoder: OneHotEncoder = OneHotEncoder,
            val_split: float = 0.2,
            stratify: bool = True,
    ):
        self.dataset: pd.DataFrame = dataset.copy()
        self.target_name: str = target_name
        self.categorical_features: list[str] = categorical_features
        self.numerical_features: list[str] = numerical_features
        self.numerical_scaler: StandardScaler = numerical_scaler()
        self.one_hot_encoder: OneHotEncoder = one_hot_encoder()
        self.val_split: float = val_split
        self.stratify: bool = stratify
        self.categorical_transformers: dict[str, MinShifter] = {
            feature: MinShifter() for feature in self.categorical_features
        }

    def split(self):
        X = self.dataset.drop(columns=[self.target_name])
        y = self.dataset[self.target_name]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, stratify=y if self.stratify else None
        )

        return X_train, X_val, y_train, y_val
    
    def transform_target(self, y: np.ndarray, fit:bool = False)->np.ndarray:
        if fit:
            self.one_hot_encoder.fit(y.reshape(-1, 1))
        return self.one_hot_encoder.transform(y.reshape(-1, 1))
    
    def scale_numericals(self, X: np.ndarray, fit:bool = False)->np.ndarray:
        if fit:
            self.numerical_scaler.fit(X)
        return self.numerical_scaler.transform(X)
    
    def prepare_categoricals(self, X: np.ndarray, fit:bool = False)->np.ndarray:
        for k, feature in enumerate(self.categorical_features):
            if fit:
                self.categorical_transformers[feature].fit(X[:, k])
            X[:, k] = self.categorical_transformers[feature].transform(X[:, k])
        return X
    
    def _run_pipeline(self, dataset: DatasetModel, fit: bool)->DatasetModel:
        if dataset.target is not None:
            dataset.target = self.transform_target(dataset.target, fit)
        dataset.numerical_features = self.scale_numericals(dataset.numerical_features, fit)
        dataset.categorical_features = self.prepare_categoricals(dataset.categorical_features, fit)
        return dataset
    
    def dataset_pipeline(self)->tuple[DatasetModel, DatasetModel]:
        X_train, X_val, y_train, y_val = self.split()
        
        self.train = DatasetModel(
            numerical_feature_names=self.numerical_features,
            categorical_feature_names=self.categorical_features,
            numerical_features=X_train[self.numerical_features].values,
            categorical_features=X_train[self.categorical_features].values,
            target=y_train.values
        )

        self._run_pipeline(self.train, True)

        self.val = DatasetModel(
            numerical_feature_names=self.numerical_features,
            categorical_feature_names=self.categorical_features,
            numerical_features=X_val[self.numerical_features].values,
            categorical_features=X_val[self.categorical_features].values,
            target=y_val.values
        )

        self._run_pipeline(self.val, False)

        return self.train, self.val
    
    def run_pipeline_new_data(self, data: pd.DataFrame)->DatasetModel:
        dataset = DatasetModel(
            numerical_feature_names=self.numerical_features,
            categorical_feature_names=self.categorical_features,
            numerical_features=data[self.numerical_features].values,
            categorical_features=data[self.categorical_features].values,
        )

        return self._run_pipeline(dataset, False)
    
    def get_class_labels(self, predictions: np.ndarray)->np.ndarray:
        output_classes = self.one_hot_encoder.categories_[0].shape[0]
        predictions_ints = np.argmax(predictions, axis=1)
        predictions_ohe = np.zeros((len(predictions_ints), output_classes))
        predictions_ohe[np.arange(len(predictions_ints)), predictions_ints] = 1
        labels = self.one_hot_encoder.inverse_transform(predictions_ohe)
        return labels