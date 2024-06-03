import pandas as pd
import numpy as np

import torch
from torch import nn

from src.preprocess import PreProcessingPipeline
from src.dataset import GraduationDataset
from src.model import ANN
from src.trainer import Trainer
from src.config import Config

np.random.seed(42)
torch.manual_seed(42)


if __name__ == "__main__":

    target = Config.target.value
    categoricals = Config.categoricals.value
    numericals = Config.numericals.value
    features = categoricals + numericals

    train = pd.read_csv(Config.train_path.value)

    preprocessing = PreProcessingPipeline(
        dataset=train,
        target_name=target,
        categorical_features=categoricals,
        numerical_features=numericals,
        val_split=Config.val_split.value
    )

    train_data, val_data = preprocessing.dataset_pipeline()

    train_dataset = GraduationDataset(
        numerical_features=train_data.numerical_features,
        categorical_features=train_data.categorical_features,
        target=train_data.target
    )

    val_dataset = GraduationDataset(
        numerical_features=val_data.numerical_features,
        categorical_features=val_data.categorical_features,
        target=val_data.target
    )

    numerical_features = train_data.numerical_features.shape[1]
    categorical_features = train_data.categorical_features.shape[1]
    output_classes = train_data.target.shape[1]
    hidden_units = Config.hidden_units.value

    model = ANN(
        numerical_features=numerical_features,
        categorical_max_values=[
            train_data.categorical_features[:, i].max() 
            for i in range(categorical_features)
        ],
        output_classes=output_classes,
        hidden_units=hidden_units
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.learning_rate.value,
        weight_decay=Config.weight_decay.value
    )

    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_save_path="./model",
        gradient_accumulation_steps=Config.gradient_accumulation_steps.value,
        eval_steps=Config.eval_steps.value,
        early_stopping_patience=Config.early_stopping_patience.value,
        batch_size=Config.batch_size.value,
    )

    trainer.train()

