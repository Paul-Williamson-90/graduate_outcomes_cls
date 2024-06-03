import os

import numpy as np
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model import ANN


class Trainer:

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            train_dataset: Dataset,
            val_dataset: Dataset,
            model_save_path: str,
            batch_size: int = 32,
            n_epochs: int = 100,
            gradient_accumulation_steps: int = 1,
            eval_steps: int|float = 10,
            early_stopping_patience: int = 3
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_steps = eval_steps
        self.early_stopping_patience = early_stopping_patience
        self.best_loss: float = float("inf")
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def _process_batch(
            self,
            batch: dict, 
            step: int
    ):
        numerical = batch["numerical"]
        categorical = batch["categorical"]
        target = batch["target"]
        
        numerical = numerical.to(self.device)
        categorical = categorical.to(self.device)
        target = target.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(numerical, categorical)
        loss = self.criterion(output, target)
        loss.backward()
        if step % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
        return loss.item()
    
    def _evaluate(
            self
    ):
        self.model.eval()
        targets = []
        predictions = []
        total_loss = 0
        progress_bar = tqdm(
            enumerate(self.val_loader), 
            total=len(self.val_loader), 
            desc=f"Validating...", 
            leave=False, 
            position=1
        )
        with torch.no_grad():
            for batch in self.val_loader:
                progress_bar.update(1)
                numerical = batch["numerical"]
                categorical = batch["categorical"]
                target = batch["target"]
                numerical = numerical.to(self.device)
                categorical = categorical.to(self.device)
                target = target.to(self.device)
                output = self.model(numerical, categorical)
                output = output.cpu()
                target = target.cpu()
                loss = self.criterion(output, target)
                targets.extend(target.numpy())
                predictions.extend(output.numpy())
                total_loss += loss.item()
        return targets, predictions, total_loss / len(self.val_loader)
    
    def _run_epoch(
            self,
            epoch: int,
            step: int,
            patience: int
    ):
        self.model.train()
        self.model.to(self.device)
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}", leave=False, position=0)
        for batch in self.train_loader:
            progress_bar.update(1)
            loss = self._process_batch(batch, step)
            total_loss += loss
            if step % self.eval_steps == 0 and step != 0:
                targets, predictions, val_loss = self._evaluate()
                predictions = np.argmax(predictions, axis=1)
                targets = np.argmax(targets, axis=1)
                accuracy = accuracy_score(targets, predictions)
                f1 = f1_score(targets, predictions, average="macro")
                print("="*50)
                print("="*50)
                print(f"Epoch: {epoch+1}")
                print(f"Training Loss: {total_loss / self.eval_steps}")
                print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}, F1: {f1}")
                print("="*50)
                print("="*50)
                if val_loss < self.best_loss:
                    print("** NEW BEST MODEL FOUND! **")
                    self.best_loss = val_loss
                    patience = 0
                    self.model.save_pretrained(self.model_save_path)
                else:
                    patience += 1
                if patience == self.early_stopping_patience:
                    break
                total_loss = 0
            step += 1
        return step, patience
    
    def train(
            self
    ):
        step = 0
        self.best_loss = float("inf")
        patience = 0
        early_stopped = False
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        if isinstance(self.eval_steps, float):
            self.eval_steps = int((len(self.train_dataset) / self.batch_size) * self.eval_steps)
        for epoch in range(self.n_epochs):
            step, patience = self._run_epoch(epoch, step, patience)
            if epoch == 1:
                self.model.trainable_embeddings(False)
            if patience == self.early_stopping_patience:
                early_stopped = True
                break
        if early_stopped:
            print("** EARLY STOPPING... **")
            self.model = ANN.load_pretrained(self.model_save_path)
        else:
            self.model.save_pretrained(self.model_save_path)
        return self.model