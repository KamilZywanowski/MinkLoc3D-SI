# Author: Jacek Komorowski
# Warsaw University of Technology
# Modified by: Kamil Zywanowski, Adam Banaszczyk, Michal Nowicki (Poznan University of Technology 2021)

import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from numba import njit, jit
import numpy as np

from datasets.oxford import OxfordDataset, IntensityDataset, TrainTransform, TrainSetTransform
from datasets.samplers import BatchSampler
from misc.utils import MinkLocParams


def make_datasets(params: MinkLocParams, debug=False):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(params.aug_mode)
    train_set_transform = TrainSetTransform(params.aug_mode)
    if debug:
        max_elems = 1000
    else:
        max_elems = None

    use_intensity = params.model_params.version in ['MinkLoc3D-I', 'MinkLoc3D-SI']

    if params.dataset_name in ['USyd', 'IntensityOxford']:
        datasets['train'] = IntensityDataset(params.dataset_folder, params.train_file, params.num_points,
                                             params.max_distance, use_intensity, params.dataset_name, train_transform,
                                             set_transform=train_set_transform, max_elems=max_elems)
    else:
        datasets['train'] = OxfordDataset(params.dataset_folder, params.train_file, params.num_points,
                                          params.max_distance, train_transform,
                                          set_transform=train_set_transform, max_elems=max_elems)

    val_transform = None
    if params.val_file is not None:
        if params.dataset_name in ['USyd', 'IntensityOxford']:
            datasets['train'] = IntensityDataset(params.dataset_folder, params.val_file, params.num_points,
                                                 params.max_distance, use_intensity, params.dataset_name, val_transform)
        else:
            datasets['val'] = OxfordDataset(params.dataset_folder, params.val_file,
                                            params.num_points, params.max_distance, val_transform)

    return datasets


def make_eval_dataset(params: MinkLocParams):
    # Create evaluation datasets
    use_intensity = params.model_params.version in ['MinkLoc3D-I', 'MinkLoc3D-SI']

    if params.dataset_name in ['USyd', 'IntensityOxford']:
        dataset = IntensityDataset(params.dataset_folder, params.test_file, params.num_points,
                                   params.max_distance, use_intensity, params.dataset_name, transform=None)
    else:
        dataset = OxfordDataset(params.dataset_folder, params.test_file, params.num_points, params.max_distance,
                                transform=None)

    return dataset


def make_collate_fn(dataset: OxfordDataset, version, dataset_name, mink_quantization_size=None):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]
        batch = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, point_dim) tensor
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            batch = dataset.set_transform(batch)

        if mink_quantization_size is None:
            # Not a MinkowskiEngine based model
            batch = {'cloud': batch}
        else:
            if version == 'MinkLoc3D':
                coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                          for e in batch]
                coords = ME.utils.batched_coordinates(coords)
                # Assign a dummy feature equal to 1 to each point
                # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)

            elif version == 'MinkLoc3D-I':
                coords = []
                feats = []
                for e in batch:
                    c, f = ME.utils.sparse_quantize(coordinates=e[:, :3], features=e[:, 3].reshape([-1, 1]),
                                                    quantization_size=mink_quantization_size)
                    coords.append(c)
                    feats.append(f)
                coords = ME.utils.batched_coordinates(coords)
                feats = torch.cat(feats, dim=0)

            elif version == 'MinkLoc3D-S':
                coords = []
                for e in batch:
                    # Convert coordinates to spherical
                    spherical_e = torch.tensor(to_spherical(e.numpy(), dataset_name), dtype=torch.float)
                    c = ME.utils.sparse_quantize(coordinates=spherical_e[:, :3], quantization_size=mink_quantization_size)
                    coords.append(c)

                coords = ME.utils.batched_coordinates(coords)
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)

            elif version == 'MinkLoc3D-SI':
                coords = []
                feats = []
                for e in batch:
                    # Convert coordinates to spherical
                    spherical_e = torch.tensor(to_spherical(e.numpy(), dataset_name), dtype=torch.float)
                    c, f = ME.utils.sparse_quantize(coordinates=spherical_e[:, :3], features=spherical_e[:, 3].reshape([-1, 1]),
                                                    quantization_size=mink_quantization_size)
                    coords.append(c)
                    feats.append(f)
                coords = ME.utils.batched_coordinates(coords)
                feats = torch.cat(feats, dim=0)

            batch = {'coords': coords, 'features': feats}

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[dataset.queries[label]['positives'][e] for e in labels] for label in labels]
        negatives_mask = [[dataset.queries[label]['negatives'][e] for e in labels] for label in labels]

        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        return batch, positives_mask, negatives_mask

    return collate_fn


def make_dataloaders(params: MinkLocParams, debug=False):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, debug=debug)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'],  params.model_params.version, params.dataset_name,
                                       params.model_params.mink_quantization_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=params.num_workers, pin_memory=True)

    if 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=params.batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_collate_fn(datasets['val'],  params.model_params.version, params.dataset_name,
                                         params.model_params.mink_quantization_size)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    return dataloders


@njit
def to_spherical(points, dataset_name):
    spherical_points = []
    for point in points:
        if (np.abs(point[:3]) < 1e-4).all():
            continue

        r = np.linalg.norm(point[:3])

        # Theta is calculated as an angle measured from the y-axis towards the x-axis
        # Shifted to range (0, 360)
        theta = np.arctan2(point[1], point[0]) * 180 / np.pi
        if theta < 0:
            theta += 360

        if dataset_name == "USyd":
            # VLP-16 has 2 deg VRes and (+15, -15 VFoV).
            # Phi calculated from the vertical axis, so (75, 105)
            # Shifted to (0, 30)
            phi = (np.arccos(point[2] / r) * 180 / np.pi) - 75

        elif dataset_name in ['IntensityOxford', 'Oxford']:
            # Oxford scans are built from a 2D scanner.
            # Phi calculated from the vertical axis, so (0, 180)
            phi = np.arccos(point[2] / r) * 180 / np.pi

        elif dataset_name == 'KITTI':
            # HDL-64 has 0.4 deg VRes and (+2, -24.8 VFoV).
            # Phi calculated from the vertical axis, so (88, 114.8)
            # Shifted to (0, 26.8)
            phi = (np.arccos(point[2] / r) * 180 / np.pi) - 88

        if point.shape[-1] == 4:
            spherical_points.append([r, theta, phi, point[3]])
        else:
            spherical_points.append([r, theta, phi])

    return spherical_points
