import os
from pathlib import Path
# From https://github.com/mary-phuong/multiexit-distillation/blob/master/utils.py
PROJ_DIR = Path(os.path.realpath(__file__)).parent
RUNS_DB_DIR = PROJ_DIR / 'runs_db'

def dict_drop(dic, *keys):
    new_dic = dic.copy()
    for key in keys:
        if key in new_dic:
            del new_dic[key]
    return new_dic

# Masksemble implementation, get from https://github.com/nikitadurasov/masksembles/blob/main/masksembles/torch.py

import numpy as np

def generate_masks_(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.
    Results of this function are stochastic, that is, calls with the same sets
    of arguments might generate outputs of different shapes. Check generate_masks
    and generation_wrapper function for more deterministic behaviour.
    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """

    total_positions = int(m * s)
    masks = []

    for _ in range(n):
        new_vector = np.zeros([total_positions])
        idx = np.random.choice(range(total_positions), m, replace=False)
        new_vector[idx] = 1
        masks.append(new_vector)

    masks = np.array(masks)
    # drop useless positions
    masks = masks[:, ~np.all(masks == 0, axis=0)]
    return masks


def generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.
    Resulting masks are required to have fixed features size as it's described in [1].
    Since process of masks generation is stochastic therefore function evaluates
    generate_masks_ multiple times till expected size is acquired.
    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    References
    [1] `Masksembles for Uncertainty Estimation: Supplementary Material`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
    """

    masks = generate_masks_(m, n, s)
    # hardcoded formula for expected size, check reference
    expected_size = int(m * s * (1 - (1 - 1 / s) ** n))
    while masks.shape[1] != expected_size:
        masks = generate_masks_(m, n, s)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by c, n, scale params.
     Allows to generate masks sets with predefined features number c. Particularly
     convenient to use in torch-like layers where one need to define shapes inputs
     tensors beforehand.
    :param c: int, number of channels in generated masks
    :param n: int, number of masks in the set
    :param scale: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """

    if c < 10:
        raise ValueError("Masksembles approach couldn't be used in such setups where "
                         f"number of channels is less then 10. Current value is (channels={c}). "
                         "Please increase number of features in your layer or remove this "
                         "particular instance of Masksembles from your architecture.")

    if scale > 6.:
        raise ValueError("Masksembles approach couldn't be used in such setups where "
                         f"scale parameter is larger then 6. Current value is (scale={scale}).")

    # inverse formula for number of active features in masks
    active_features = int(int(c) / (scale * (1 - (1 - 1 / scale) ** n)))

    # FIXME this piece searches for scale parameter value that generates
    #  proper number of features in masks, sometimes search is not accurate
    #  enough and masks.shape != c. Could fix it with binary search.
    masks = generate_masks(active_features, n, scale)
    for s in np.linspace(max(0.8 * scale, 1.0), 1.5 * scale, 300):
        if masks.shape[-1] >= c:
            break
        masks = generate_masks(active_features, n, s)
    new_upper_scale = s

    if masks.shape[-1] != c:
        for s in np.linspace(max(0.8 * scale, 1.0), new_upper_scale, 1000):
            if masks.shape[-1] >= c:
                break
            masks = generate_masks(active_features, n, s)

    if masks.shape[-1] != c:
        raise ValueError("generation_wrapper function failed to generate masks with "
                         "requested number of features. Please try to change scale parameter")

    return masks

import torch
from torch import nn

class Masksembles2D(nn.Module):
    """
    :class:`Masksembles2D` is high-level class that implements Masksembles approach
    for 2-dimensional inputs (similar to :class:`torch.nn.Dropout2d`).

    :param channels: int, number of channels used in masks.
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C, H, W)
        * Output: (N, C, H, W) (same shape as input)

    Examples:

    >>> m = Masksembles2D(16, 4, 2.0)
    >>> input = torch.ones([4, 16, 28, 28])
    >>> output = m(input)

    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, channels: int, n: int, scale: float):
        super().__init__()

        self.channels = channels
        self.n = n
        self.scale = scale
        self.cnt = 0

        masks = generation_wrapper(channels, n, scale)
        masks = torch.from_numpy(masks).float().to("cuda" if torch.cuda.is_available() else "cpu")
        self.masks = torch.nn.Parameter(masks, requires_grad=False)
        if torch.cuda.is_available():
            self.masks = self.masks.cuda()

    def forward(self, inputs):
        batch = inputs.shape[0]
        if self.training:
            if batch % self.n != 0:
                raise ValueError('Batch size must be divisible by n, got batch {} and n {}'.format(batch, self.n))
            x = torch.split(inputs.unsqueeze(1), batch // self.n, dim=0)
            x = torch.cat(x, dim=1).permute([1, 0, 2, 3, 4])
            x = x * self.masks.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        else: 
            x = inputs * self.masks[self.cnt][None].unsqueeze(1).unsqueeze(-1).unsqueeze(-1) 
            # print("2D Sampling... Using mask: ", self.cnt)
            self.cnt = (self.cnt + 1) % self.n     
        return x.squeeze(0).float()
    
    def extra_repr(self):
        return 'scale={}, n={}'.format(
            self.scale, self.n
        )


class Masksembles1D(nn.Module):
    """
    :class:`Masksembles1D` is high-level class that implements Masksembles approach
    for 1-dimensional inputs (similar to :class:`torch.nn.Dropout`).

    :param channels: int, number of channels used in masks.
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C)
        * Output: (N, C) (same shape as input)

    Examples:

    >>> m = Masksembles1D(16, 4, 2.0)
    >>> input = torch.ones([4, 16])
    >>> output = m(input)


    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, channels: int, n: int, scale: float):

        super().__init__()

        self.channels = channels
        self.n = n
        self.scale = scale
        self.cnt = 0

        masks = generation_wrapper(channels, n, scale)
        masks = torch.from_numpy(masks).float().to("cuda" if torch.cuda.is_available() else "cpu")
        self.masks = torch.nn.Parameter(masks, requires_grad=False)

    def forward(self, inputs):
        batch = inputs.shape[0]
        if self.training:
            if batch % self.n != 0:
                raise ValueError('Batch size must be divisible by n, got batch {} and n {}'.format(batch, self.n))
            x = torch.split(inputs.unsqueeze(1), batch // self.n, dim=0)
            x = torch.cat(x, dim=1).permute([1, 0, 2])
            x = x * self.masks.unsqueeze(1)
            x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        else:
            x = inputs * self.masks[self.cnt][None].unsqueeze(1)
            # print("1D Sampling... Using mask: ", self.cnt)
            self.cnt = (self.cnt + 1) % self.n
        return x.squeeze(0)
    
    def extra_repr(self):
        return 'scale={}, n={}'.format(
            self.scale, self.n
        )