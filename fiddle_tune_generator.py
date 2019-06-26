"""Fiddle tune generator.

Usage: python fiddle_tune_generator.py --help
"""

import random
import argparse
from typing import Optional
from typing import Tuple

from silence_tensorflow import silence_tensorflow
from scripts.models_training import DATA_DIR, MODELS_DIR, SEQUENCE_LENGTH, PAD, BOS, EOS
from src.utils import generate_tune

from myutils.json_tools import load_json, save_json

# TODO:
# Integrated tune midi and pdf support

silence_tensorflow()

CHAR2ID = load_json(MODELS_DIR / 'char2id.json')
ID2CHAR = {value: key for key, value in CHAR2ID.items()}
PAD_ID = CHAR2ID[PAD]
BOS_ID = CHAR2ID[BOS]
EOS_ID = CHAR2ID[EOS]

AVAILABLE_TYPE_MODE_PAIRS = []

for fpath in MODELS_DIR.iterdir():
    if fpath.name.endswith('.hdf5'):
        name_ = fpath.name.split('.')[0]
        tune_type_, tune_mode_ = name_.split('_')
        AVAILABLE_TYPE_MODE_PAIRS.append((tune_type_, tune_mode_))

AVAILABLE_TYPE_MODE_PAIRS_STR = '; '.join(repr(item) for item in AVAILABLE_TYPE_MODE_PAIRS)

AVAILABLE_TYPES = sorted({item[0] for item in AVAILABLE_TYPE_MODE_PAIRS})
AVAILABLE_TYPES_STR = ', '.join(repr(item) for item in AVAILABLE_TYPES)

AVAILABLE_MODES = sorted({item[1] for item in AVAILABLE_TYPE_MODE_PAIRS})
AVAILABLE_MODES_STR = ', '.join(repr(item) for item in AVAILABLE_MODES)

GENERATED_TUNES_FPATH = DATA_DIR / 'generated_tunes.json'


def process_parameters(tune_type: Optional[str], tune_mode: Optional[str]) -> Tuple[str, str]:
    """Process the provided tune type and mode. Here's what can happen:
        * if both parameters are None - pick a random available tune type/mode pair and return it,
        * if only `tune_type` is not None:
            * if it is invalid - raise KeyError,
            * otherwise, pick a random available `tune_mode` for this type and return the type/mode pair,
        * if only `tune_mode` is not None: do the same as in the previous step in reverse,
        * if both parameters are not None:
            * if such type/mode pair is unavailable - raise KeyError,
            * otherwise, return this pair.

    :param tune_type: str or None
    :param tune_mode: str or None
    :return: tuple of str and str
    """

    if (tune_type is None) and (tune_mode is None):
        return random.choice(AVAILABLE_TYPE_MODE_PAIRS)

    elif tune_mode is None:  # tune_type is not None
        if tune_type not in AVAILABLE_TYPES:
            raise KeyError(f"Tune type '{tune_type}' not found. Available types: {AVAILABLE_TYPES_STR}.")
        else:
            tune_mode = random.choice(list(item[1] for item in AVAILABLE_TYPE_MODE_PAIRS if item[0] == tune_type))
            return tune_type, tune_mode

    elif tune_type is None:  # tune_mode is not None
        if tune_mode not in AVAILABLE_MODES:
            raise KeyError(f"Tune mode '{tune_mode}' not found. Available modes: {AVAILABLE_MODES_STR}.")
        else:
            tune_type = random.choice(list(item[0] for item in AVAILABLE_TYPE_MODE_PAIRS if item[1] == tune_mode))
            return tune_type, tune_mode

    else:  # both tune_type and tune_mode are not None
        if (tune_type, tune_mode) not in AVAILABLE_TYPE_MODE_PAIRS:
            raise KeyError(f"No model for tune type '{tune_type}' and mode '{tune_mode}'. "
                           f"Available tune type/mode pairs: {AVAILABLE_TYPE_MODE_PAIRS_STR}.")
        else:
            return tune_type, tune_mode


def save_tune(tune: str) -> None:
    """Save a generated tune into a json file.

    :param tune: str
    """

    try:
        generated_tunes = load_json(GENERATED_TUNES_FPATH)
    except FileNotFoundError:
        generated_tunes = []

    generated_tunes.append(tune)
    save_json(generated_tunes, GENERATED_TUNES_FPATH)


def main(tune_type: str, tune_mode: str, n_tunes: int = 1) -> None:
    """Use a model corresponding to a certain tuple of tune type and tune mode
    to generate a given number of  tunes. The function prints this tune and saves
    it to a json file.

    :param tune_type: str
    :param tune_mode: str
    :param n_tunes: int
    """

    import tensorflow as tf

    model = tf.keras.models.load_model(MODELS_DIR / f'{tune_type}_{tune_mode}.hdf5')

    for _ in range(n_tunes):
        tune = generate_tune(
            model=model,
            sequence_length=SEQUENCE_LENGTH,
            pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID,
            id2char=ID2CHAR,
        )

        print(tune, '\n')
        save_tune(tune)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"Fiddle tune generator. "
                                                 f"Available tune type/mode pairs: {AVAILABLE_TYPE_MODE_PAIRS_STR}.")

    parser.add_argument('-t', '--type', help=f"specify tune type: {AVAILABLE_TYPES_STR}")
    parser.add_argument('-m', '--mode', help=f"specify tune mode: {AVAILABLE_MODES_STR}")
    parser.add_argument('-n', '--number', type=int, help="specify the number of tunes to generate")

    args = parser.parse_args()

    n_tunes_ = args.number or 1

    try:
        tune_type_, tune_mode_ = process_parameters(tune_type=args.type, tune_mode=args.mode)
    except KeyError as e:
        print(e.args[0], "For help: python fiddle_tune_generator.py --help")
    else:
        print(f'Generating a {tune_type_} in {tune_mode_} mode. This will take a few seconds.\n')
        main(tune_type=tune_type_, tune_mode=tune_mode_, n_tunes=n_tunes_)
        print('To convert it to midi or pdf, you can use this web service:\n'
              'http://www.mandolintab.net/abcconverter.php')
