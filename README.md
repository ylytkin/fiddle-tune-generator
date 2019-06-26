# fiddle-tune-generator

Data: https://github.com/adactio/TheSession-data/blob/master/json/tunes.json

Usage:
1. `pip install -r requirements.txt`
2. `python fiddle_tune_generator.py --help`

Current pipeline:
1. (See `scripts/transposing.py`) Transpose all tunes to unified keys :
    * G for major tunes,
    * A for dorian tunes,
    * E for minor tunes,
    * D for mixolydian tunes.
2. (See `scripts/models_training.py`) For each pair of tune type (reel, jig, waltz, etc.) and mode (maj, min, dor, mix):
    1. generate a character-level train and test data,
    2. train a neural network with two LSTM layers (256 neurons each with 0.2 and 0.5 dropout rate, respectively) and a dense softmax output.
3. (See `src/utils.py`) Inference:
    1. Based on the current sequence of character, get predictions for the next character,
    2. Truncate the distribution of these predictions, leaving only the top probable predictions, such that their probabilities sum up to 0.95,
    3. Randomly sample a character from this truncated distribution,
    4. If the next character is the `EOS` tag, stop iteration.
