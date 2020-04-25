"""General model training script.

This script performs training of a general model for tunes of certain type.

If a model for a given type is already present, aborts.

Requires:
    - a json file `data/tunes_merged.json` (available after running the transposing script)
Provides:
    - a json file `models/<tune_type>_char2id.json` for the given tune type
    - a model `models/<tune_type>.hdf5` for the given tune type
    - a json file `data/training_history_<tune_type>.json` for the given tune type

Warning!
Proceed with caution when changing any of the configuration parameters, since it
may render the models unusable for inference of further training.
"""

import argparse
import multiprocessing as mpl
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

import tensorflow as tf
import pandas as pd
from scipy import sparse

from myutils import log
from myutils.json_tools import save_json, load_json

thread_count = mpl.cpu_count() // 3

tf.config.threading.set_intra_op_parallelism_threads(thread_count)
tf.config.threading.set_inter_op_parallelism_threads(thread_count)

ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = ROOT_DIR / 'data'

TRAIN_DATA_DIR = DATA_DIR / 'train_data'
TRAIN_DATA_DIR.mkdir(exist_ok=True)

MODELS_DIR = ROOT_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

PAD = 'PAD'
BOS = 'BOS'
EOS = 'EOS'

SEQUENCE_LENGTH = 70

LSTM_LAYER_SIZE = {
    'reel': 512,
    'jig': 400,
    'polka': 256,
    'waltz': 256,
    'hornpipe': 256,
    'slip_jig': 192,
}

EPOCHS = {
    'reel': 15,
    'jig': 20,
    'polka': 25,
    'waltz': 25,
    'hornpipe': 30,
    'slip_jig': 40,
}

BATCH_SIZE = 64


def load_tunes(tunes_fpath: Path) -> pd.DataFrame:
    """Load the tunes json file, convert it into a pandas DataFrame
    and filter the tunes that are too long.

    :param tunes_fpath: tunes file path (Path)
    :return: tunes (DataFrame)
    """

    tunes = load_json(tunes_fpath)
    tunes_df = pd.DataFrame(tunes)
    tunes_df.info()

    log('Truncating data by length.')

    length_limit = tunes_df['length'].median() + 1.5 * tunes_df['length'].quantile(0.75)

    log(f"{length_limit} char limit. {(tunes_df['length'] > length_limit).sum()} tunes out of bounds.")

    tunes_df = tunes_df[tunes_df['length'] <= length_limit]

    return tunes_df


def get_model_data_fpaths(tune_type: str) -> Tuple[Path, Path, Path, Path]:
    """Get all the file paths needed to load/save a model. These are:
        - `char2id` (encoding dict)
        - `x`, `y` (encoded tunes)
        - `model` (the model itself)

    :param tune_type: str
    :return: char2id_fpath, x_fpath, y_fpath, model_fpath
    """

    char2id_fpath = MODELS_DIR / f'{tune_type}_char2id.json'
    x_fpath = TRAIN_DATA_DIR / f'{tune_type}_x.npz'
    y_fpath = TRAIN_DATA_DIR / f'{tune_type}_y.npz'
    model_fpath = MODELS_DIR / f'{tune_type}.hdf5'

    return char2id_fpath, x_fpath, y_fpath, model_fpath


def prepare_data_and_initialize_model(
        tune_type: str,
) -> Tuple[int, List[List[int]], List[int], tf.keras.models.Model]:
    """Prepare the data and initialize a model. This is done during the first
    training phase, when there's no serialized model present.

    :param tune_type: str
    :return: n_chars, x, y, model
    """

    tunes_df = load_tunes(DATA_DIR / 'tunes_merged.json')
    tunes_df = tunes_df[tunes_df['type'] == tune_type]

    char2id_fpath, x_fpath, y_fpath, _ = get_model_data_fpaths(tune_type)

    log(f'{tunes_df.shape[0]} tunes. Preparing the vocabulary.')

    char_counts = pd.Series(list(''.join(tunes_df['abc_transposed']))).value_counts()
    log(f'{char_counts.shape[0]} unique chars.')

    chars = char_counts.index.tolist()
    chars.extend([PAD, BOS, EOS])

    n_chars = len(chars)
    char2id = dict(zip(chars, range(n_chars)))

    save_json(char2id, char2id_fpath)

    chars_set = set(chars)

    log(f'Vocabulary size: {n_chars}.')

    pad_id = char2id[PAD]
    bos_id = char2id[BOS]
    eos_id = char2id[EOS]

    log('Preparing data.')

    x = []
    y = []

    for _, row in tunes_df.iterrows():
        tune = row['abc_transposed']

        if len(set(tune) - chars_set) > 0:
            continue

        encoded_tune = [char2id[char] for char in tune]
        encoded_tune = [pad_id] * (SEQUENCE_LENGTH - 1) + [bos_id] + encoded_tune + [eos_id]

        for i in range(len(encoded_tune) - SEQUENCE_LENGTH):
            j = i + SEQUENCE_LENGTH

            x.append(encoded_tune[i:j])
            y.append(encoded_tune[j])

    x_sparse = sparse.csr_matrix(x)
    y_sparse = sparse.csr_matrix(y)

    sparse.save_npz(x_fpath.as_posix(), x_sparse)
    sparse.save_npz(y_fpath.as_posix(), y_sparse)

    lstm_layer_size = LSTM_LAYER_SIZE[tune_type]

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(SEQUENCE_LENGTH, n_chars)),
        tf.keras.layers.LSTM(lstm_layer_size, dropout=0.1, return_sequences=True),
        tf.keras.layers.LSTM(lstm_layer_size, dropout=0.3, return_sequences=True),
        tf.keras.layers.LSTM(lstm_layer_size, dropout=0.5),
        tf.keras.layers.Dense(n_chars, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return n_chars, x, y, model


def load_data_and_model(tune_type: str) -> Tuple[int, List[List[int]], List[int], tf.keras.models.Model]:
    """Load the prepared data and model. This is done when a model has been trained and
    serialized previously, now the user wants to load it and train some more.

    :param tune_type: str
    :return: n_chars, x, y, model
    """

    log('Found existing model. Loading it for further training.')

    char2id_fpath, x_fpath, y_fpath, model_fpath = get_model_data_fpaths(tune_type)

    char2id = load_json(char2id_fpath)
    n_chars = len(char2id)

    x_sparse = sparse.load_npz(x_fpath.as_posix())
    y_sparse = sparse.load_npz(y_fpath.as_posix())

    x = x_sparse.A.tolist()
    y = y_sparse.A.flatten().tolist()

    model = tf.keras.models.load_model(model_fpath)

    return n_chars, x, y, model


def save_history(history: Dict[str, List[float]], tune_type: str):
    """Save training history log.

    :param history: dict
    :param tune_type: str
    """

    history = {str(key): [float(item) for item in values] for key, values in history.items()}

    history_fpath = DATA_DIR / f'general_training_history_{tune_type}.json'

    if history_fpath.exists():
        previous_history = load_json(history_fpath)
    else:
        previous_history = dict()

    previous_history = defaultdict(list, previous_history)

    for key, values in history.items():
        previous_history[key].extend(values)

    save_json(previous_history, history_fpath)


def main(tune_type: str, epochs: Optional[int] = None):
    _, _, _, model_fpath = get_model_data_fpaths(tune_type)

    if model_fpath.exists():
        n_chars, x, y, model = load_data_and_model(tune_type=tune_type)
    else:
        n_chars, x, y, model = prepare_data_and_initialize_model(tune_type=tune_type)

    model.summary()

    x = tf.keras.utils.to_categorical(x, num_classes=n_chars, dtype='int8')
    y = tf.keras.utils.to_categorical(y, num_classes=n_chars, dtype='int8')

    log(f'x.shape = {x.shape}, y.shape = {y.shape}')

    epochs = epochs or EPOCHS[tune_type]

    fit = model.fit(
        x, y,
        epochs=epochs,
        batch_size=BATCH_SIZE,
    )

    log('Finished. Saving.')

    model.save(model_fpath)
    save_history(history=fit.history, tune_type=tune_type)

    log('Done.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', help=(f"specify the tune type to use for training. "
                                              f"Possible types: {list(LSTM_LAYER_SIZE.keys())}"))
    parser.add_argument('-e', '--epochs', type=int, help="specify the number of epochs to train for")

    args = parser.parse_args()
    tune_type_ = args.type
    epochs_ = args.epochs

    if tune_type_ is None:
        print('You must specify a tune type for training.')
    elif tune_type_ not in LSTM_LAYER_SIZE:
        print(f"Invalid tune type: '{tune_type_}'. Possible types: {list(LSTM_LAYER_SIZE.keys())}.")
    else:
        main(tune_type=tune_type_, epochs=epochs_)
