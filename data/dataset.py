import numpy as np
import torch
from matplotlib import pyplot as plt

from data.util.dataset_types import DatasetTypes
from evaluation.logger.log_tags import LogTag
from evaluation.logger.logger import Logger


class Dataset:
    """Contains the functionality and data needed for a dataset in this evaluation system"""
    def __init__(self, dataset_type: DatasetTypes, is_classification: bool, x: np.ndarray, y: np.ndarray):
        self.is_classification = is_classification
        self.dataset_type = dataset_type
        tensor_x = torch.tensor(x).type(torch.FloatTensor)
        tensor_y = torch.tensor(y).type(torch.LongTensor) if is_classification else torch.unsqueeze(torch.tensor(y).type(torch.FloatTensor), 1)
        p = np.random.permutation(len(tensor_y))
        self.x, self.y = tensor_x[p], tensor_y[p]

        splits_percent = (0.7, 0.3)
        n_data_points = len(y)

        n_train_points = int(splits_percent[0] * n_data_points)
        splits_sizes = (n_train_points, n_data_points - n_train_points)
        self.train_x, self.test_x = torch.split(self.x, splits_sizes)
        self.train_y, self.test_y = torch.split(self.y, splits_sizes)

    def get_dataset_name(self):
        return self.dataset_type.name

    def get_sample_amount(self):
        return len(self.y)

    def get_amount_inputs(self):
        return self.x.shape[1]

    def get_amount_outputs(self):
        return len(np.unique(self.y)) if self.is_classification else 1

    def visualize(self, x, y, logger: Logger, model, save=False):
        if self.dataset_type == DatasetTypes.Corner \
                or self.dataset_type == DatasetTypes.Vertical \
                or self.dataset_type == DatasetTypes.Spirals2 \
                or self.dataset_type == DatasetTypes.Spirals3 \
                or self.dataset_type == DatasetTypes.Spirals4 \
                or self.dataset_type == DatasetTypes.Curves \
                or self.dataset_type == DatasetTypes.Spheres \
                or self.dataset_type == DatasetTypes.Compound:

            additional_dimensions = np.zeros((x.shape[1] - 2))  # for multidimensional problems, add fixed 0z as dimensions
            decision_boundary_visualization_points = []

            is_numpy = type(x) is np.ndarray
            if is_numpy:
                x_and_bias = np.column_stack((x, np.ones(x.shape[0])))
                x = torch.Tensor(x_and_bias)
                y = torch.Tensor(y)

            x_spread = (np.min(x.data.numpy()[:, 0] - 0.1), np.max(x.data.numpy()[:, 0] + 0.1))
            y_spread = (np.min(x.data.numpy()[:, 1] - 0.1), np.max(x.data.numpy()[:, 1] + 0.1))
            n_points_x, n_points_y = 100, 100
            for point_x in np.arange(x_spread[0], x_spread[1], (x_spread[1] - x_spread[0]) / n_points_x):
                for point_y in np.arange(y_spread[0], y_spread[1], (y_spread[1] - y_spread[0]) / n_points_y):
                    decision_boundary_visualization_points.append(np.concatenate(([point_x, point_y], additional_dimensions), axis=0))

            decision_boundary_visualization_points_ndarray = np.array(decision_boundary_visualization_points)
            decision_boundary_visualization_points_array = torch.tensor(decision_boundary_visualization_points_ndarray).type(torch.FloatTensor)
            if is_numpy:
                decision_boundary_visualization_points_ndarray = np.column_stack((decision_boundary_visualization_points_ndarray, np.ones(decision_boundary_visualization_points_ndarray.shape[0])))
                test_point_classes = torch.max(torch.Tensor(model.forward(decision_boundary_visualization_points_ndarray)), 1)[1]
            else:
                test_point_classes = torch.max(model.forward(decision_boundary_visualization_points_array), 1)[1]

            fig = plt.figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(decision_boundary_visualization_points_array[:, 0], decision_boundary_visualization_points_array[:, 1], c=test_point_classes, cmap='brg')
            ax.scatter(x[:, 0], x[:, 1], c=y, edgecolors='black', cmap='brg')
            if save:
                plt.savefig(self.dataset_type.name + ".pdf")
            logger.log_figure(LogTag.VisualizationPlot, ax)
            plt.close(fig)
