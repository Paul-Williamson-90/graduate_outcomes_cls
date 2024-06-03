import torch
from torch import nn
import json
import os


class CategoryEmbedding(nn.Module):

    def __init__(
            self,
            num_embeddings: int,
            dropout: float = 0.1,
            init_mode: str = "fan_out",
            init_nonlinearity: str = "relu"
    ):
        super(CategoryEmbedding, self).__init__()
        self.embedding_dim = num_embeddings // 2
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim
        )
        nn.init.kaiming_normal_(self.embedding.weight, mode=init_mode, nonlinearity=init_nonlinearity)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        x = torch.where(x < self.num_embeddings, x, torch.tensor(self.num_embeddings - 1, dtype=torch.long))
        x = torch.where(x < 0, x, torch.tensor(0, dtype=torch.long))
        return self.dropout(self.embedding(x))

    def trainable(self, trainable: bool):
        self.embedding.weight.requires_grad = trainable

class FFUnit(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float = 0.1,
            init_mode: str = "fan_in",
            init_nonlinearity: str = "relu"
    ):
        super(FFUnit, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(self.linear.weight, mode=init_mode, nonlinearity=init_nonlinearity)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.activation(self.batch_norm(self.linear(x))))

class ANN(nn.Module):

    def __init__(
            self,
            numerical_features: int,
            categorical_max_values: list[int],
            output_classes: int,
            hidden_units: list[int],
            dropout: float = 0.05
    ):
        super(ANN, self).__init__()
        self.categorical_embeddings = nn.ModuleList([
            CategoryEmbedding(max_val + 1, dropout) for max_val in categorical_max_values
        ])
        self.numerical_features = numerical_features
        self.categorical_features = sum(embedding.embedding_dim for embedding in self.categorical_embeddings)
        self.hidden_units = hidden_units
        self.ff_units = nn.ModuleList([
            FFUnit(
                in_features=self.numerical_features + self.categorical_features,
                out_features=hidden_units[0],
                dropout=dropout,
                init_mode="fan_in",
            )
        ])
        for i in range(1, len(hidden_units)):
            self.ff_units.append(
                FFUnit(
                    in_features=hidden_units[i - 1],
                    out_features=hidden_units[i],
                    dropout=dropout,
                    init_mode="fan_in",
                )
            )
        self.output = nn.Linear(hidden_units[-1], output_classes)
        nn.init.kaiming_normal_(self.output.weight, mode="fan_in")
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, numerical: torch.Tensor, categorical: torch.Tensor):
        assert categorical.shape[1] == len(self.categorical_embeddings), "Mismatch between input data and model embeddings: {} != {}".format(categorical.shape[1], len(self.categorical_embeddings))
        embeddings = [self.categorical_embeddings[i](categorical[:, i]) for i in range(len(self.categorical_embeddings))]
        embeddings = torch.cat(embeddings, dim=1)
        x = torch.cat([numerical, embeddings], dim=1)
        for unit in self.ff_units:
            x = unit(x)
        logits = self.output(x)
        return logits

    def predict(self, numerical: torch.Tensor, categorical: torch.Tensor):
        return self.output_activation(self(numerical, categorical))
    
    def trainable_embeddings(self, trainable: bool):
        for embedding in self.categorical_embeddings:
            embedding.trainable(trainable)

    def _create_config(self) -> dict:
        categorical_max_values = [int(emb.num_embeddings - 1) for emb in self.categorical_embeddings]
        return {
            "numerical_features": self.numerical_features,
            "categorical_max_values": categorical_max_values,
            "hidden_units": self.hidden_units,
            "output_classes": self.output.out_features
        }

    def save_pretrained(self, model_save_path: str):
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(self.state_dict(), f"{model_save_path}/model.pth")
        config_path = f"{model_save_path}/config.json"
        with open(config_path, 'w') as f:
            json.dump(self._create_config(), f)

    @classmethod
    def load_pretrained(cls, model_save_path: str):
        config_path = f"{model_save_path}/config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        model = cls(
            numerical_features=config["numerical_features"],
            categorical_max_values=config["categorical_max_values"],
            hidden_units=config["hidden_units"],
            output_classes=config["output_classes"]
        )

        model_path = f"{model_save_path}/model.pth"
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=map_location))

        return model