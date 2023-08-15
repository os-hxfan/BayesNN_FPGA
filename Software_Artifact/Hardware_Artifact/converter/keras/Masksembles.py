import numpy as np
import tensorflow as tf
import keras
from nn2bnn import strategy_fn, _convert_model, HlsLayer
from sympy import Symbol, Eq, solveset, N, S, solve
from collections.abc import Iterable

"""References:
    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
    Code: https://github.com/nikitadurasov/masksembles
"""


def _generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.

    Results of this function are stochastic, that is, calls with the same sets
    of arguments might generate outputs of different shapes. Check generate_masks
    and generation_wrapper function for more deterministic behaviour.

    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """

    total_positions = round(m * s)
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

    masks = _generate_masks(m, n, s)
    # hardcoded formula for expected size, check reference
    expected_size = round(m * s * (1 - (1 - 1 / s) ** n))
    while masks.shape[1] != expected_size:
        masks = _generate_masks(m, n, s)
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
        raise ValueError(
            "Masksembles approach couldn't be used in such setups where "
            f"number of channels is less then 10. Current value is (channels={c}). "
            "Please increase number of features in your layer or remove this "
            "particular instance of Masksembles from your architecture."
        )

    if scale > 6.0 or scale < 1.0:
        raise ValueError(
            "Masksembles approach couldn't be used in such setups where "
            f"scale parameter is larger then 6 or smaller than 1. Current value is (scale={scale})."
        )

    # inverse formula for number of active features in masks, m * s * (1 - (1 - 1 / s) ** n) <= c
    active_features = round(c / (scale * (1 - (1 - 1 / scale) ** n)))
    if active_features * n < c:
        raise ValueError("The currenct scale might be too large!")

    s = Symbol("s", positive=True, real=True)
    sol = solveset(active_features * s * (1 - (1 - 1 / s) ** n) - c, s, domain=S.Reals)
    if not sol:
        raise ValueError("No solution for scale parameter!")
    if isinstance(sol, Iterable):
        scale = min(sol, key=lambda x: abs(x - scale))
    scale = float(N(scale))
    # expected_size <= m * s * (1 - (1 - 1 / s) ** n) <= c <= m * n
    expected_size = round(active_features * scale * (1 - (1 - 1 / scale) ** n))
    if expected_size != c:
        raise ValueError(
            f"generation_wrapper function failed to generate masks with {scale}"
            "requested number of features. Please try to change scale parameter"
        )
    print("Found scale", scale, type(scale))
    return scale, generate_masks(active_features, n, scale)


class Masksembles(keras.layers.Layer):
    """
    :class:`Masksembles` is high-level class that implements Masksembles approach
    for both 1-dimensional and 2-dimensional inputs (similar to :class:`tensorflow.keras.layers.Dropout`).

    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C) or (N, H, W, C)
        * Output: (N, C) or (N, H, W, C) (same shape as input)

    Examples:

    >>> m = Masksembles(4, 2.0)
    >>> inputs = tf.ones([4, 16])
    >>> output = m(inputs)


    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, n: int, scale: float, masks: np.array = None):
        super(Masksembles, self).__init__()
        self.n = n
        self.scale = scale
        self.original_scale = scale
        self.init_masks = masks

    def build(self, input_shape):
        channels = input_shape[-1]
        scale, masks = generation_wrapper(channels, self.n, self.scale)
        print ("Generate:", masks)
        masks = self.init_masks if self.init_masks is not None else masks
        print ("Assigned:", masks)
        self.scale = scale
        # self.scale = 2.0
        # masks = tf.constant([[0, 1, 0], [1, 0, 1]], dtype="float32")
        # print(masks)
        if len(input_shape) == 2:
            masks = masks[:, tf.newaxis]
        elif len(input_shape) == 4:
            masks = masks[:, tf.newaxis, tf.newaxis, tf.newaxis]
        else:
            raise Exception(
                f"Masksembles supports only 1D or 2D inputs, the input shape is {input_shape}"
            )
        self.masks = self.add_weight(
            "kernel", shape=masks.shape, trainable=False, dtype="float32"
        )
        self.masks.assign(masks)
        print(self.masks.shape)

    def call(self, inputs, training=False):
        x = tf.stack(tf.split(inputs, self.n))
        x *= self.masks
        x = tf.concat(tf.split(x, self.n), axis=1)
        return tf.squeeze(x, axis=0)

    def get_config(self):
        config = {"num_masks": self.n, "scale": self.original_scale}
        base_config = super(Masksembles, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(n=config["num_masks"], scale=config["scale"])


class MasksemblesModel(HlsLayer):
    r"""
    This class uses Masksembles to convert a traditional neural network to a Bayesian neural network.
    Note that the model to be
    converted should be contructed using either Sequential or Functional APIs.
    """

    def __init__(
        self, model, num_masks=4, scale=1.0, num=0, strategy="default", input=None, **kwargs
    ):
        super().__init__()
        supported_layers = strategy_fn[strategy](model, Masksembles, **kwargs)
        self.model = _convert_model(
            model,
            "Masksembles",
            supported_layers,
            n=num_masks,
            scale=scale,
            input=input,
        ) if num > 0 else model 
        self.num_masks = num_masks
        self.scale = scale

    def call(self, input, training=True):
        if training:
            return self.model(input, training=True)
        else:
            if len(input.shape) == 2:
                input = tf.tile(input, [self.num_masks, 1])
            elif len(input.shape) == 4:
                input = tf.tile(input, [self.num_masks, 1, 1, 1])
            else:
                raise Exception(
                    f"Masksembles supports only 1D or 2D inputs, the input shape is {input.shape}"
                )
            prediction = self.model(input, training=False)
            if isinstance(prediction, list):
                prediction = [tf.reduce_mean(
                              tf.reshape(pred, [self.num_masks, -1, pred.shape[-1]]),
                              axis=0) for pred in prediction]
                prediction = sum(prediction) / len(prediction)
            else: 
                prediction = tf.reduce_mean(
                             tf.reshape(prediction, [self.num_masks, -1, prediction.shape[-1]]),
                             axis=0,
                             )
            return prediction

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = super(MasksemblesModel, self).get_config()
        config["model"] = self.model
        config["num_masks"] = self.num_masks
        config["scale"] = self.scale
        return config
