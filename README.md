# fiddle-tune-generator

Data: https://github.com/adactio/TheSession-data/blob/master/json/tunes.json

Usage:
1. `pip install -r requirements.txt`
2. `python fiddle_tune_generator.py --help`

Current pipeline:
1. (See `scripts/transpose.py`) Transpose all tunes to unified keys:
    * G for major tunes,
    * A for dorian tunes,
    * E for minor tunes,
    * D for mixolydian tunes.
2. (See `scripts/train_general_model.py`) For each type of tunes (reel, jig, etc.):
    1. generate a character-level train and test data,
    2. train a neural network with three LSTM layers and a dense softmax output.
3. (See `src/utils.py`) Inference:
    1. based on the current sequence of characters, get predictions for the next character,
    2. truncate the distribution of these predictions, leaving only the top probable predictions, such that their probabilities sum up to 0.95,
    3. randomly sample a character from this truncated distribution,
    4. if the next character is the `EOS` tag, stop iteration.

Ideas for further improvement.
* Currently, we train a general model for all tunes of a given type, regardless of the mode (maj, min, dor, mix, etc.). It is not obvious, whether this is a problem, since all mode information is set in the preamble to the tune, and not in the abc code of the tune. Thus, most likely, everything will be fine with the current approach.   
Nevertheless, in case it turns out to be a problem, it would be a cool possibility to try transfer learning on such general models (i.e. taking a general reels model as a base and transfer it to a major reels model or a mixolydian reels model, for example).   
Therefore, it's a good idea to try such approach and see if it improves accuracy.
