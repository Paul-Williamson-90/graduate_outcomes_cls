from typing import Optional

import numpy as np

import torch
from torch.utils.data import Dataset

class GraduationDataset(Dataset):
    
        def __init__(
                self,
                numerical_features: np.ndarray,
                categorical_features: np.ndarray,
                target: Optional[np.ndarray] = None
        ):
            self.numerical_features = numerical_features
            self.categorical_features = categorical_features
            self.target = target
    
        def __len__(self):
            return self.numerical_features.shape[0]
    
        def __getitem__(self, idx):
            if self.target is not None:
                return {
                    "numerical": torch.tensor(self.numerical_features[idx], dtype=torch.float32),
                    "categorical": torch.tensor(self.categorical_features[idx], dtype=torch.long),
                    "target": torch.tensor(self.target.toarray()[idx], dtype=torch.float32)
                }
            else:
                return {
                    "numerical": torch.tensor(self.numerical_features[idx], dtype=torch.float32),
                    "categorical": torch.tensor(self.categorical_features[idx], dtype=torch.long)
                }