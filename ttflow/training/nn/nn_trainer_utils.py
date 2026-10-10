import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from loguru import logger




class NeuralNetworkTrainingUtils:
    @staticmethod
    def train_nn(
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_test: torch.Tensor,
            y_test: torch.Tensor,
            batch_size: int,
            epochs: int,
            min_rel_loss_delta: float,
            patience: int,
    ) -> dict:

        train_ds = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Batch the test set to prevent VRAM spikes
        test_ds = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        loss_fn = nn.MSELoss()

        learning_curve = {"train": [], "test": []}
        loss_delta_rel = np.inf
        loss_stagnation_count = 0
        alpha = 0.9

        # State tracking across epochs for smoothing
        s_train = None
        s_test = None

        # Helper function to evaluate test loss safely without storing gradients
        def evaluate_test_loss():
            model.eval()
            total_test_loss = 0.0
            with torch.inference_mode():  # Drops computation graph, saving massive VRAM
                for b_x, b_y in test_loader:
                    preds = model(b_x)
                    total_test_loss += loss_fn(preds, b_y).item() * b_x.size(0)
            return total_test_loss / len(test_loader.dataset)

        # Initial check (without autograd tracking)
        init_test_loss = evaluate_test_loss()
        logger.info(
            f"@epoch = 0 (before training), loss_fn = {loss_fn}, loss_value = {init_test_loss:.6f}"
        )

        # training loop
        with torch.no_grad():
            y_pred_test = model(x_test)
            logger.info(f"MSELoss(test) before training:{MSELoss()(y_test, y_pred_test)}")

        for epoch in tqdm(range(epochs), desc="Epoch"):
            # --- Training ---
            model.train()
            train_epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)  # Frees memory faster than standard zero_grad
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss = train_epoch_loss / len(train_loader.dataset)

            # Update smoothed train loss
            s_train = (
                epoch_loss
                if s_train is None
                else alpha * epoch_loss + (1 - alpha) * s_train
            )

            # --- Evaluation ---
            current_test_loss = evaluate_test_loss()

            # Update smoothed test loss
            s_test = (
                current_test_loss
                if s_test is None
                else alpha * current_test_loss + (1 - alpha) * s_test
            )

            learning_curve["train"].append(s_train)
            learning_curve["test"].append(s_test)

            # --- Early Stopping Checks ---
            if len(learning_curve["train"]) > 2:
                prev_loss = learning_curve["train"][-2]
                curr_loss = learning_curve["train"][-1]
                loss_delta_rel = (curr_loss - prev_loss) / abs(prev_loss)

                if abs(loss_delta_rel) <= min_rel_loss_delta:
                    loss_stagnation_count += 1
                    if loss_stagnation_count >= patience:
                        logger.info(
                            f"Loss stagnated @epoch = {epoch}, "
                            f"loss_delta_rel = {abs(loss_delta_rel):.6e} <= {min_rel_loss_delta}, "
                            f"for {patience} epochs. Exiting training loop."
                        )
                        break
                else:
                    loss_stagnation_count = 0

            logger.info(
                f"@epoch = {epoch + 1}, smoothed-train-loss = {s_train:.6f} | "
                f"smoothed-test-loss = {s_test:.6f} | "
                f"rel_loss_delta = {loss_delta_rel} | "
                f"loss_stagnation_count = {loss_stagnation_count}"
            )

        # after train
        y_pred_test = model(x_test)
        logger.info(f"MSELoss(test) after training:{MSELoss()(y_test, y_pred_test)}")

        assert len(learning_curve["train"]) == len(learning_curve["test"])
        return learning_curve
