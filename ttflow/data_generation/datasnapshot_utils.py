import os.path
from typing import Iterable
from datetime import datetime
from loguru import logger
from ttflow.data_generation.data_utils import DataUtils
from ttflow.viz.viz import Visualizer

TIMESTAMP = datetime.now().isoformat()
DATA_GENERATION_FIGURES_DIR = "figures/data_generation"


class DataSnapShotUtils:
    @staticmethod
    def generate_transport_data_snapshot(dataset_name: str, x_dim: int, y_dim: int, n_train: int, n_test: int,
                                         n_eval: int, **kwargs) -> dict:
        """
        :param n_test:
        :type n_test:
        :param n_train:
        :type n_train:
        :param dataset_name:
        :type dataset_name:
        :param x_dim:
        :type x_dim:
        :param y_dim:
        :type y_dim:
        :param n_eval:
        :type n_eval:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        logger.info(f"Generating samples for {dataset_name}")
        data_splits_sizes = {"train": n_train, "test": n_test, "eval": n_eval}
        data_snapshot_dict = {"dataset_name": dataset_name, "params": kwargs}
        for split in data_splits_sizes.keys():
            logger.info(f"Generating split = {split}")
            n_split = data_splits_sizes[split]
            _, x1 = DataUtils.get_transport_data(x_dim=x_dim, y_dim=y_dim, dataset_name=dataset_name,
                                                 n=n_split, **kwargs)
            data_snapshot_dict[split] = {"x1": x1}
            logger.debug(f"Finished generating split = {split} with x1.shape = {x1.shape}")

        logger.info(f"Finished generating samples for {dataset_name}")
        # viz

        return data_snapshot_dict

    @staticmethod
    def generate_manywell_snapshot(total_data_dim: int, manywell_dim: int, n_train_test: int, n_eval: int,
                                   time_grid: Iterable[float]) -> dict:
        raise NotImplementedError("To remove ")
        # TODO move to the generic function
        dataset_name = "manywell"

        # ( 1 ) Generate test dataset
        logger.info("Generating eval data...")
        x0, x1 = DataUtils.get_transport_data(x_dim=-1, y_dim=-1, dataset_name=dataset_name,
                                              dim=total_data_dim,
                                              many_well_dim=manywell_dim
                                              , n=n_eval)

        # x, t, y = DataUtils.rectified_flow_get_l2_xy(x0=x0, x1=x1, time_grid=None)
        data_snapshot_dict = {"dataset_name": dataset_name, "x0": x0, "x1": x1,
                              "data_dim": total_data_dim,
                              "manywell_dim": manywell_dim}
        # logger.info(
        #     f"Generated inference eval data-snapshot with x.shape = {x.shape}, "
        #     f"t.shape = {t.shape}, y.shape = {y.shape}")

        # ( 2 ) Generate train dataset for global-time-modeling arch for v(x(t),t) = G1G2 .. GdGt
        logger.info(f"Generating data for global arch (param-efficient)...")
        x0, x1 = DataUtils.get_transport_data(x_dim=-1, y_dim=-1, dataset_name=dataset_name,
                                              dim=total_data_dim,
                                              many_well_dim=manywell_dim,
                                              n=n_train_test)
        x, t, y = DataUtils.get_rectified_flow_get_l2_data(z0=x0, z1=x1, time_grid=None)
        data_snapshot_dict["train"] = {"l2": {"global-time": {"x": x, "t": t, "y": y}},
                                       "inference": {"x0": x0, "x1": x1}}
        logger.info(f"Finished generating train-data for space-time rch (param-efficient) "
                    f"with x.shape = {x.shape}, t.shape = {t.shape}")
        # ( 3 ) Generate train data-snapshot for v(x(t),t) \approx v_n(x(tn)) where each v_n(x(tn)) is trained individually
        x_grid, y = DataUtils.get_rectified_flow_get_l2_data(z0=x0, z1=x1, time_grid=time_grid)
        data_snapshot_dict["train"]["l2"]["time-grid"] = {"x_grid": x_grid, "time_grid": time_grid, "y": y}
        return data_snapshot_dict

    @staticmethod
    def generate_gaussian_mixture_2d(n_train_test: int, num_components: int, n_eval: int,
                                     time_grid: Iterable[float]) -> dict:
        raise NotImplementedError("To remove ")
        # TODO add to the generic function
        x_dim = 2
        y_dim = 2
        dataset_name = "gaussian_mixture_2d"
        # TODO add more parameters later
        # (1) generate eval dataset
        logger.info(f"Generating eval data...")
        x0, x1 = DataUtils.get_transport_data(x_dim=x_dim, y_dim=y_dim, dataset_name=dataset_name,
                                              n=n_eval, time_grid=None, num_components=num_components)
        x, t, y = DataUtils.get_rectified_flow_get_l2_data(x0, x1, time_grid=None)
        data_snapshot_dict = {"dataset_name": dataset_name, "params": {"num_components": num_components},
                              "eval": {"l2": {"x": x, "t": t, "y": y}, "inference": {"x0": x0, "x1": x1}},
                              "data_dim": 2,
                              "comp": num_components}
        # (2) generate space-time dataset (ordinary data)
        logger.info(f"Generating train data for space-time arch (param-efficient)...")
        x0, x1 = DataUtils.get_transport_data(x_dim=x_dim, y_dim=y_dim, dataset_name=dataset_name,
                                              n=n_train_test, time_grid=None,
                                              num_components=num_components)
        data_snapshot_dict["train"] = {"inference": {"x0": x0, "x1": x1}}

        Visualizer.plot_two_2d_distributions(x0, x1,
                                             output_file_name=
                                             os.path.join(DATA_GENERATION_FIGURES_DIR,
                                                          f"gaussian_mixture_2d_ncomp_{num_components}_{TIMESTAMP}.png"))
        x, t, y = DataUtils.get_rectified_flow_get_l2_data(x0, x1, time_grid=None)
        data_snapshot_dict["train"]["l2"] = {"space-time": {"x": x, "t": t, "y": y}}
        logger.info(f"Finished generating train-data for space-time arch (param-efficient) "
                    f"with x.shape = {x.shape}, t.shape = {t.shape}")
        # ( 3 ) Generate train data-snapshot for v(x(t),t) \approx v_n(x(tn)) where each v_n(x(tn)) is trained individually
        x_grid, y = DataUtils.get_rectified_flow_get_l2_data(z0=x0, z1=x1, time_grid=time_grid)
        data_snapshot_dict["train"]["l2"]["time-grid"] = {"x_grid": x_grid, "time_grid": time_grid, "y": y}
        return data_snapshot_dict
