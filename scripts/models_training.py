"""Models training script.

This script performs training of a model for each pair of tune type
and tune mode. See the function `get_trained_model` for model signature.

Requires: a json file `data/tunes_merged.json` (available after running
    the transposing script.
Provides:
    - a json file `models/char2id.json`
    - train data files (in npz format) in `data/train_data/<tune_type>_<tune_mode>/`
    for each pair of tune type and tune mode
    - a model `models/<tune_type>_<tune_mode>.hdf5` for each pair of
    tune type and tune mode
"""

from pathlib import Path
from itertools import product

import tensorflow as tf
import pandas as pd
from tqdm import trange
from scipy import sparse
from sklearn.model_selection import train_test_split

from myutils import log
from myutils.json_tools import load_json, save_json

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

TRAINABLE_TYPES = ['reel', 'jig', 'polka', 'waltz', 'hornpipe', 'slip jig']
TRAINABLE_MODES = ['maj', 'min', 'dor', 'mix']

EPOCHS = 5
BATCH_SIZE = 64


def get_trained_model(tune_type: str, tune_mode: str, tunes_df: pd.DataFrame, char2id: dict) -> tf.keras.models.Model:
    """Prepare a certain slice of data, create a model, and train this model, before returning it.

    :param tune_type: str
    :param tune_mode: str
    :param tunes_df: pandas DataFrame
    :param char2id: character encoding dict
    :return: trained tensorflow model
    """

    current_tunes_df = tunes_df[(tunes_df['type'] == tune_type) & (tunes_df['mode'] == tune_mode)].copy(deep=True)
    train_data_dir = TRAIN_DATA_DIR / f'{tune_type}_{tune_mode}'
    train_data_dir.mkdir(exist_ok=True)

    log(f'{current_tunes_df.shape[0]} tunes.')

    chars = set(char2id.keys())
    n_chars = len(chars)

    pad_id = char2id[PAD]
    bos_id = char2id[BOS]
    eos_id = char2id[EOS]

    log('Preparing data.')

    x = []
    y = []

    for _, row in current_tunes_df.iterrows():
        tune = row['abc_transposed']

        if len(set(tune) - chars) > 0:
            continue

        encoded_tune = [char2id[char] for char in tune]
        encoded_tune = [pad_id] * (SEQUENCE_LENGTH - 1) + [bos_id] + encoded_tune + [eos_id]

        for i in range(len(encoded_tune) - SEQUENCE_LENGTH):
            j = i + SEQUENCE_LENGTH

            x.append(encoded_tune[i:j])
            y.append(encoded_tune[j])

    x = tf.keras.utils.to_categorical(x, num_classes=n_chars, dtype='int8')
    y = tf.keras.utils.to_categorical(y, num_classes=n_chars, dtype='int8')

    log(f'x.shape = {x.shape}, y.shape = {y.shape}')

    log('Saving train data.')

    for i in trange(x.shape[1]):
        x_slice = x[:, i, :]
        x_slice_sparse = sparse.csr_matrix(x_slice)
        sparse.save_npz((train_data_dir / f'x_{i}.npz').as_posix(), x_slice_sparse)

    y_sparse = sparse.csr_matrix(y)
    sparse.save_npz((train_data_dir / 'y.npz').as_posix(), y_sparse)

    log('Training the model.')

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.1)

    lstm_layer_size = x_train.shape[0] // (20 * n_chars)

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=x_train[0].shape),
        tf.keras.layers.LSTM(lstm_layer_size, dropout=0.1, return_sequences=True),
        tf.keras.layers.LSTM(lstm_layer_size, dropout=0.3, return_sequences=True),
        tf.keras.layers.LSTM(lstm_layer_size, dropout=0.5),
        tf.keras.layers.Dense(n_chars, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))

    log('Finished.')

    return model


def main():
    log('Loading and preprocessing data.')

    tunes = load_json(DATA_DIR / 'tunes_merged.json')
    tunes_df = pd.DataFrame(tunes)
    tunes_df.info()
    
    log('Truncating data by length.')
    
    length_limit = tunes_df['length'].median() + 1.5 * tunes_df['length'].quantile(0.75)

    log(f"{length_limit} char limit. {(tunes_df['length'] > length_limit).sum()} tunes out of bounds.")
    
    tunes_df = tunes_df[tunes_df['length'] <= length_limit]
    
    log('Preparing the vocabulary.')
    
    char_counts = pd.Series(list(''.join(tunes_df['abc_transposed']))).value_counts()
    log(f'{char_counts.shape[0]} unique chars.')
    
    chars = set(char_counts.index.tolist())
    chars.update([PAD, BOS, EOS])

    n_chars = len(chars)
    char2id = dict(zip(chars, range(n_chars)))
    log(f'Vocabulary size: {n_chars}. Training models.')

    save_json(char2id, MODELS_DIR / 'char2id.json')

    failed_attempts = []

    for tune_type, tune_mode in product(TRAINABLE_TYPES, TRAINABLE_MODES):
        if (MODELS_DIR / f'{tune_type}_{tune_mode}.hdf5').exists():
            log(f"Model for tune type '{tune_type}' and mode '{tune_mode}' already exists. Skipping.")

        else:
            log(f"Tune type: '{tune_type}', tune mode: '{tune_mode}'.")
    
            try:
                model = get_trained_model(tune_type=tune_type, tune_mode=tune_mode, tunes_df=tunes_df, char2id=char2id)
                model.save(MODELS_DIR / f'{tune_type}_{tune_mode}.hdf5')
            except Exception as e:
                log(f'Failed to obtain a model. Reason: {repr(e)}')
                failed_attempts.append({
                    'tune_type': tune_type,
                    'tune_mode': tune_mode,
                    'reason': repr(e),
                })

    log(f'Finished. {len(failed_attempts)} failed attempts.')

    if len(failed_attempts) > 0:
        save_json(failed_attempts, DATA_DIR / 'models_training_failed_attempts.json')

    log('Done.')
    

if __name__ == '__main__':
    main()
