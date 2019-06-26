from typing import Dict
from typing import Tuple

import pandas as pd
import numpy as np
import tensorflow as tf

__all__ = [
    'generate_tune',
]


def truncate_distribution(distribution: np.ndarray, top_p: float = 0.95) -> Tuple[pd.Index, np.ndarray]:
    """Truncate the given distribution, leaving only the top probable entries,
    such that their probabilities sum up to the given `top_p`.

    :param distribution: 1-dimensional numpy array
    :param top_p: float
    :return: tuple of resulting entry ids and corresponding probabilities
    """

    distribution = pd.Series(distribution)
    distribution_cumsum = distribution.sort_values(ascending=False).cumsum().shift().fillna(0)
    distribution = distribution[distribution_cumsum <= top_p]
    distribution = distribution / distribution.sum()

    return distribution.index, distribution.values


def generate_next_char_id(sequence: np.ndarray, n_chars: int, model: tf.keras.models.Model, top_p: float = 0.95) -> int:
    """Generate next char id, based on the given sequence.

    :param sequence: previous char id sequence (list of ints)
    :param n_chars: number of chars (int)
    :param model: tensorflow model
    :param top_p: see `truncate_distribution` (float)
    :return: char id (int)
    """

    x = tf.keras.utils.to_categorical(np.array([sequence]), num_classes=n_chars, dtype='int8')
    distribution = model.predict_proba(x)[0]
    ids, probas = truncate_distribution(distribution, top_p=top_p)

    char_id = np.random.choice(ids, p=probas)

    return char_id


def generate_tune(
        model: tf.keras.models.Model,
        sequence_length: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        id2char: Dict[int, str],
        top_p: float = 0.95,
) -> str:
    """Generate a tune from start to finish, using the given `model`.

    :param model: tensorflow model
    :param sequence_length: sequence length (int)
    :param pad_id: encoded id of 'PAD' (int)
    :param bos_id: encoded id of 'BOS' (int)
    :param eos_id: encoded id of 'EOS' (int)
    :param id2char: char decoding dict
    :param top_p: see `truncate_distribution` (float)
    :return: tune str
    """

    n_chars = len(id2char)

    sequence = ([pad_id] * (sequence_length - 1)) + [bos_id]
    next_char_id = None
    tune = []

    while next_char_id != eos_id:
        next_char_id = generate_next_char_id(sequence=sequence, n_chars=n_chars, model=model, top_p=top_p)
        tune.append(id2char[next_char_id])
        sequence = sequence[1:] + [next_char_id]

    tune = ''.join(tune[:-1])
    tune = '\n'.join(line for line in tune.splitlines() if len(line) > 0)

    return tune
