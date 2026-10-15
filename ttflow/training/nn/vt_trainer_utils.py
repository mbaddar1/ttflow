from typing import List
from loguru import logger
import torch
from torch.nn import MSELoss

from ttflow.data_generation.data_utils import DataUtils
from ttflow.training.nn.nn_models import VelocityFieldRegressionNet
from ttflow.training.nn.nn_trainer_utils import NeuralNetworkTrainingUtils


class VTTrainerUtils:
    @staticmethod
    def reflow_nn_train(
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            z0_train: torch.Tensor,
            z1_train: torch.Tensor,
            z0_test: torch.Tensor,
            z1_test: torch.Tensor,
            batch_size: int,
            epochs: int,
            patience: int = 10,
            min_rel_loss_delta: float = 5e-5,
            zt_mode: str = "linear",
    ) ->dict:
        zt_train, t_train, y_train = (
            DataUtils.get_rectified_flow_get_l2_data(z0=z0_train, z1=z1_train,
                                                     zt_mode=zt_mode))

        zt_test, t_test, y_test = (
            DataUtils.get_rectified_flow_get_l2_data(z0=z0_test, z1=z1_test,
                                                     zt_mode=zt_mode))

        zt_train_aug = torch.cat([zt_train, t_train], dim=1)
        zt_test_aug = torch.cat([zt_test, t_test], dim=1)


        learning_curve = NeuralNetworkTrainingUtils.train_nn(model=model, optimizer=optimizer,
                                                             x_train=zt_train_aug, y_train=y_train,
                                                             x_test=zt_test_aug, y_test=y_test,
                                                             min_rel_loss_delta=min_rel_loss_delta,
                                                             batch_size=batch_size,
                                                             epochs=epochs,
                                                             patience=patience)
        return learning_curve
