"""Fiddle tune generator.

Usage: python fiddle_tune_generator.py --help
"""

import argparse
import random
from typing import List
from pathlib import Path

from silence_tensorflow import silence_tensorflow
from scripts.train_general_model import DATA_DIR, MODELS_DIR, SEQUENCE_LENGTH, PAD, BOS, EOS
from src.utils import generate_tune

from myutils.json import load_json, save_json

# TODO:
# Integrated tune midi and pdf support

silence_tensorflow()

GENERATED_TUNES_FPATH = DATA_DIR / 'generated_tunes.json'


def get_available_models_list(models_dir: Path) -> List[str]:
    """Get a list of available models. These are models for which there is:
        - a char2id encoding json file (of form `<tune_type>_char2id.json`),
        - a pre-trained model file (of form `<tune_type>.hdf5`).

    :param models_dir: models directory (Path)
    :return: list of strs
    """

    fnames = [fpath.name for fpath in models_dir.iterdir()]
    available_models = []

    for fname in fnames:
        if fname.endswith('.hdf5'):
            tune_type = fname.split('.')[0]

            if f'{tune_type}_char2id.json' in fnames:
                available_models.append(tune_type)

    return available_models


AVAILABLE_MODELS = get_available_models_list(MODELS_DIR)
AVAILABLE_MODELS_STR = ', '.join(AVAILABLE_MODELS)


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


def main(tune_type: str, n_tunes: int = 1) -> None:
    """Use a pre-trained model to generate a given number of tunes of a given
    type. The function prints this tune and saves it to a json file.

    :param tune_type: tune type (str)
    :param n_tunes: int
    """

    import tensorflow as tf

    char2id = load_json(MODELS_DIR / f'{tune_type}_char2id.json')
    model = tf.keras.models.load_model(MODELS_DIR / f'{tune_type}.hdf5')

    id2char = {value: key for key, value in char2id.items()}
    pad_id = char2id[PAD]
    bos_id = char2id[BOS]
    eos_id = char2id[EOS]

    for _ in range(n_tunes):
        tune = generate_tune(
            model=model,
            sequence_length=SEQUENCE_LENGTH,
            pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
            id2char=id2char,
        )

        print(tune, '\n')
        save_tune(tune)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"Fiddle tune generator. Available types: {AVAILABLE_MODELS_STR}.")
    parser.add_argument('-t', '--type', help="specify the tune type to generate")
    parser.add_argument('-n', '--number', type=int, help="specify the number of tunes to generate")

    args = parser.parse_args()

    n_tunes_ = args.number or 1
    tune_type_ = args.type or random.choice(AVAILABLE_MODELS)

    if tune_type_ not in AVAILABLE_MODELS:
        print(f"Tune type '{tune_type_}' not available. Available types: {AVAILABLE_MODELS_STR}.")
    else:
        print(f'Generating a tune. This will take a few seconds.\n')
        main(tune_type=tune_type_, n_tunes=n_tunes_)
        print('To convert it to midi or pdf, you can use this web service:\n'
              'http://www.mandolintab.net/abcconverter.php')
