"""Transposing script.

This script uses the ABC converter from mandolintab.net for transposing
all the tunes into unified keys:
    - G for major tunes,
    - E for minor tunes,
    - A for dorian tunes,
    - D for mixolydian tunes.

This also substitutes all spaces in the tune types by underscores for further
referencing. The changes are present in the file `data/tunes_merged.json`.

Requires:
    - a json file `data/tunes.json`
Provides:
    - a json file `data/transposed_tunes.json`
    - (possibly) a json file `data/transposing_failed_attempts.json`
    - a json file `data/tunes_merged.json`
"""

import re
import requests
from pathlib import Path

import bs4
import pandas as pd
from tqdm import tqdm

from myutils import log
from myutils.json import load_json, save_json

ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = ROOT_DIR / 'data'


def format_tune(tune: dict) -> str:
    """Return a properly formatted abc string.

    :param tune: tune dict
    :return: tune str
    """

    x = tune['setting']
    t = tune['name']
    r = tune['type']
    m = tune['meter']
    k = tune['mode'][:4]
    abc = tune['abc']

    return f'X: {x}\nT: {t}\nR: {r}\nM: {m}\nK: {k}\n{abc}'


URL = 'http://www.mandolintab.net/abcconverter.php'

POST_DATA_BLANK = {
    'dispoff[0]': 'Q',
    'replace': 'on',
    'ratio': 0,
    'stress': 0,
    'tab': 0,
    'tabkey': 12,
    'autolinebreak': 'off',
    'abcver': '2.1',
    'paper': 'letter',
    'graphic': 'png',
    'output': 'abc',
    'submit': 'submit',
}

TRANSPOSE_AMOUNT_MAJ = {
    'C': 7,
    'D': 5,
    'E': 3,
    'F': 2,
    'G': 0,
    'A': -2,
    'B': -4,
}

TRANSPOSE_AMOUNT_MIN = {
    'C': 4,
    'D': 2,
    'E': 0,
    'F': -1,
    'G': -3,
    'A': -5,
    'B': -7,
}

TRANSPOSE_AMOUNT_DOR = {
    'C': 9,
    'D': 7,
    'E': 5,
    'F': 4,
    'G': 2,
    'A': 0,
    'B': -2,
}

TRANSPOSE_AMOUNT_MIX = {
    'C': 2,
    'D': 0,
    'E': -2,
    'F': -3,
    'G': -5,
    'A': -7,
    'B': -9,
}

MAX_RETRIES = 10


def main():
    log('Loading data.')
    tunes = load_json(DATA_DIR / 'tunes.json')

    log(f'{len(tunes)} tunes. Transposing.')

    transposed_tunes = []
    failed_attempts = []
    failed_ids = set()

    tqdm_bar = tqdm(total=len(tunes))

    for i, tune in enumerate(tqdm(tunes)):
        for j in range(MAX_RETRIES):
            try:
                key = tune['mode'][0]
                mode = tune['mode'][1:4]
                
                if mode == 'maj':
                    transpose_amount = TRANSPOSE_AMOUNT_MAJ[key]
                elif mode == 'min':
                    transpose_amount = TRANSPOSE_AMOUNT_MIN[key]
                elif mode == 'dor':
                    transpose_amount = TRANSPOSE_AMOUNT_DOR[key]
                elif mode == 'mix':
                    transpose_amount = TRANSPOSE_AMOUNT_MIX[key]
                else:
                    raise Exception(f"Unknown mode '{mode}' for tune setting {tune['setting']}.")
                
                post_data = POST_DATA_BLANK.copy()
                post_data['abc'] = re.sub(r'\r', '\n', format_tune(tune))
                post_data['transpose'] = transpose_amount

                response = requests.post(
                    URL,
                    data=post_data,
                    headers={'User-Agent': (f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) "
                                            f"AppleWebKit/{i}.{j}.15 (KHTML, like Gecko) "
                                            f"Version/13.1 Safari/{i}.{j}.15")},
                )

                bs = bs4.BeautifulSoup(response.text, features='lxml')
                transposed_tune = bs.find('textarea', {'name': 'abc'}).text

                transposed_tune = '\n'.join(line for line in transposed_tune.splitlines() if len(line) > 0)

                transposed_tunes.append({
                    'id': i,
                    'setting': tune['setting'],
                    'abc': transposed_tune,
                })
                break

            except Exception as e:
                failed_attempts.append({
                    'id': i,
                    'setting': tune.get('setting', ''),
                    'reason': repr(e),
                })
                failed_ids.add(i)

    tqdm_bar.close()

    log(f'Finished. {len(failed_attempts)} failed attempts on {len(failed_ids)} tunes. Saving.')

    save_json(transposed_tunes, DATA_DIR / 'transposed_tunes.json')

    log('Merging tunes.')

    tunes_df = pd.DataFrame(tunes)
    transposed_tunes_df = pd.DataFrame(transposed_tunes)
    transposed_tunes_df = transposed_tunes_df.rename(columns={'abc': 'abc_transposed'})

    tunes_df = tunes_df.merge(transposed_tunes_df, on='setting')
    tunes_df['length'] = tunes_df['abc_transposed'].apply(len)

    tunes_df['key'] = tunes_df['mode'].str[0]
    tunes_df['mode'] = tunes_df['mode'].str[1:4]
    tunes_df['type'] = tunes_df['type'].apply(lambda x: '_'.join(x.split()))

    tunes_df.info()

    if tunes_df['abc_transposed'].isna().any():
        log('Note that some of the tunes have not been transposed.')

        untransposed_tune_settings = tunes_df.loc[tunes_df['abc_transposed'].isna(), 'setting'].tolist()
        failed_attempts.append(untransposed_tune_settings)

    tunes_merged = [row.to_dict() for _, row in tunes_df.iterrows()]
    save_json(tunes_merged, DATA_DIR / 'tunes_merged.json')

    if len(failed_attempts) > 0:
        save_json(failed_attempts, DATA_DIR / 'transposing_failed_attempts.json')

    log('Done.')


if __name__ == '__main__':
    main()
