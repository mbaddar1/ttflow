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
    def generate_data_snapshot(dataset_name: str, x_dim: int, y_dim: int, n_train_test: int, n_eval: int,
                               time_grid: Iterable[float], **kwargs) -> dict:
        """

        :param dataset_name:
        :type dataset_name:
        :param x_dim:
        :type x_dim:
        :param y_dim:
        :type y_dim:
        :param n_train_test:
        :type n_train_test:
        :param n_eval:
        :type n_eval:
        :param time_grid:
        :type time_grid:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        # ( 1 ) Generate test dataset
        logger.info("Generating eval data...")
        x0, x1 = DataUtils.rectified_flows_get_inference_x0x1(x_dim=x_dim, y_dim=y_dim, dataset_name=dataset_name,
                                                              n=n_eval, **kwargs)
        x, t, y = DataUtils.rectified_flow_get_l2_xy(x0=x0, x1=x1, time_grid=None)
        data_snapshot_dict = {"dataset_name": dataset_name,
                              "eval": {"l2": {"x": x, "t": t, "y": y}, "inference": {"x0": x0, "x1": x1}},
                              "params": kwargs}
        logger.info(
            f"Generated inference eval data-snapshot with x.shape = {x.shape}, t.shape = {t.shape}, y.shape = {y.shape}")

        # ( 2 ) Generate train dataset for space-time arch for v(x(t),t) = G1G2 .. GdGt
        logger.info(f"Generating data for space-time arch (param-efficient)...")
        x0, x1 = DataUtils.rectified_flows_get_inference_x0x1(x_dim=x_dim, y_dim=y_dim, dataset_name=dataset_name,
                                                              n=n_train_test, **kwargs)
        # visualize init and target distribution
        plot_file_name = os.path.join(DATA_GENERATION_FIGURES_DIR, f"{dataset_name}_params_{kwargs}.png")
        plot_title = f"{dataset_name}_{kwargs}"
        Visualizer.plot_two_2d_distributions(x0=x0, x1=x1, output_file_name=plot_file_name, title=plot_title)
        x, t, y = DataUtils.rectified_flow_get_l2_xy(x0=x0, x1=x1, time_grid=None)
        data_snapshot_dict["train"] = {"l2":{"space-time": {"x": x, "t": t, "y": y}}}
        logger.info(f"Finished generating train-data for space-time rch (param-efficient) "
                    f"with x.shape = {x.shape}, t.shape = {t.shape}")
        # ( 3 ) Generate train data-snapshot for v(x(t),t) \approx v_n(x(tn)) where each v_n(x(tn)) is trained individually
        x_grid, y = DataUtils.rectified_flow_get_l2_xy(x0=x0, x1=x1, time_grid=time_grid)
        data_snapshot_dict["train"]["space-time"] = {"x_grid": x_grid, "time_grid": time_grid, "y": y}
        return data_snapshot_dict

    @staticmethod
    def generate_manywell_snapshot(total_data_dim: int, manywell_dim: int, n_train_test: int, n_eval: int,
                                   time_grid: Iterable[float]) -> dict:
        # TODO move to the generic function
        dataset_name = "manywell"

        # ( 1 ) Generate test dataset
        logger.info("Generating eval data...")
        x0, x1 = DataUtils.rectified_flows_get_inference_x0x1(x_dim=-1, y_dim=-1, dataset_name=dataset_name,
                                                              dim=total_data_dim,
                                                              many_well_dim=manywell_dim
                                                              , n=n_eval)

        x, t, y = DataUtils.rectified_flow_get_l2_xy(x0=x0, x1=x1, time_grid=None)
        data_snapshot_dict = {"dataset_name": dataset_name,
                              "eval": {"l2": {"x": x, "t": t, "y": y}, "inference": {"x0": x0, "x1": x1}},
                              "data_dim": total_data_dim,
                              "manywell_dim": manywell_dim}
        logger.info(
            f"Generated inference eval data-snapshot with x.shape = {x.shape}, "
            f"t.shape = {t.shape}, y.shape = {y.shape}")

        # ( 2 ) Generate train dataset for space-time arch for v(x(t),t) = G1G2 .. GdGt
        logger.info(f"Generating data for space-time arch (param-efficient)...")
        x0, x1 = DataUtils.rectified_flows_get_inference_x0x1(x_dim=-1, y_dim=-1, dataset_name=dataset_name,
                                                              dim=total_data_dim,
                                                              many_well_dim=manywell_dim,
                                                              n=n_train_test)
        x, t, y = DataUtils.rectified_flow_get_l2_xy(x0=x0, x1=x1, time_grid=None)
        data_snapshot_dict["train"] = {"l2": {"space-time": {"x": x, "t": t, "y": y}},
                                       "inference": {"x0": x0, "x1": x1}}
        logger.info(f"Finished generating train-data for space-time rch (param-efficient) "
                    f"with x.shape = {x.shape}, t.shape = {t.shape}")
        # ( 3 ) Generate train data-snapshot for v(x(t),t) \approx v_n(x(tn)) where each v_n(x(tn)) is trained individually
        x_grid, y = DataUtils.rectified_flow_get_l2_xy(x0=x0, x1=x1, time_grid=time_grid)
        data_snapshot_dict["train"]["l2"]["time-grid"] = {"x_grid": x_grid, "time_grid": time_grid, "y": y}
        return data_snapshot_dict

    @staticmethod
    def generate_gaussian_mixture_2d(n_train_test: int, num_components: int, n_eval: int,
                                     time_grid: Iterable[float]) -> dict:
        # TODO add to the generic function
        x_dim = 2
        y_dim = 2
        dataset_name = "gaussian_mixture_2d"
        # TODO add more parameters later
        # (1) generate eval dataset
        logger.info(f"Generating eval data...")
        x0, x1 = DataUtils.rectified_flows_get_inference_x0x1(x_dim=x_dim, y_dim=y_dim, dataset_name=dataset_name,
                                                              n=n_eval, time_grid=None, num_components=num_components)
        x, t, y = DataUtils.rectified_flow_get_l2_xy(x0, x1, time_grid=None)
        data_snapshot_dict = {"dataset_name": dataset_name, "params": {"num_components": num_components},
                              "eval": {"l2": {"x": x, "t": t, "y": y}, "inference": {"x0": x0, "x1": x1}},
                              "data_dim": 2,
                              "comp": num_components}
        # (2) generate space-time dataset (ordinary data)
        logger.info(f"Generating train data for space-time arch (param-efficient)...")
        x0, x1 = DataUtils.rectified_flows_get_inference_x0x1(x_dim=x_dim, y_dim=y_dim, dataset_name=dataset_name,
                                                              n=n_train_test, time_grid=None,
                                                              num_components=num_components)
        data_snapshot_dict["train"] = {"inference": {"x0": x0, "x1": x1}}

        Visualizer.plot_two_2d_distributions(x0, x1,
                                             output_file_name=
                                             os.path.join(DATA_GENERATION_FIGURES_DIR,
                                                          f"gaussian_mixture_2d_ncomp_{num_components}_{TIMESTAMP}.png"))
        x, t, y = DataUtils.rectified_flow_get_l2_xy(x0, x1, time_grid=None)
        data_snapshot_dict["train"]["l2"] = {"space-time": {"x": x, "t": t, "y": y}}
        logger.info(f"Finished generating train-data for space-time arch (param-efficient) "
                    f"with x.shape = {x.shape}, t.shape = {t.shape}")
        # ( 3 ) Generate train data-snapshot for v(x(t),t) \approx v_n(x(tn)) where each v_n(x(tn)) is trained individually
        x_grid, y = DataUtils.rectified_flow_get_l2_xy(x0=x0, x1=x1, time_grid=time_grid)
        data_snapshot_dict["train"]["l2"]["time-grid"] = {"x_grid": x_grid, "time_grid": time_grid, "y": y}
        return data_snapshot_dict
