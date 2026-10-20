"""Frozen public candles for offline replay.
Never executes archive contents.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TFS = ('15m', '1h', '4h', '1d')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(folder, data, gc, start, end, failures):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    manifest = {
        'version': 1,
        'start': start.isoformat(),
        'end': end.isoformat(),
        'assets': {},
        'load_errors': failures,
        'gc': None,
        'note': (
            'Frozen inputs only. Replay uses the current code '
            'and configured assumptions.'
        ),
        'source_sha256': {},
    }

    for name in (
        'agent.py',
        'scanner.py',
        'scanner_data.py',
        'gold_volume.py',
        'backtest.py',
        'upgrade_agent.py',
        'candle_archive.py',
    ):
        p = Path(name)
        if p.is_file():
            manifest['source_sha256'][name] = digest(p)

    def write(d, name):
        path = folder / name
        d.to_csv(
            path,
            index_label='time',
            compression='gzip',
            float_format='%.17g',
        )
        return {
            'file': name,
            'sha256': digest(path),
        }

    for key, fr in data.items():
        if key not in ('BTC', 'ETH', 'SOL', 'GOLD', 'NDX'):
            raise ValueError('unknown asset')

        manifest['assets'][key] = {
            'metadata': {
                k: v for k, v in fr.items()
                if k.startswith('_')
                and isinstance(v, (str, int, float, bool))
            },
            'frames': {
                tf: write(fr[tf], f'{key}_{tf}.csv.gz')
                for tf in TFS
                if tf in fr
            },
        }

    if gc is not None:
        manifest['gc'] = write(gc, 'GC_15m.csv.gz')

    (folder / 'manifest.json').write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding='utf-8',
    )


def load(folder):
    folder = Path(folder).resolve()
    m = json.loads(
        (folder / 'manifest.json').read_text(encoding='utf-8')
    )

    if m.get('version') != 1:
        raise ValueError('unsupported candle archive version')

    start = pd.Timestamp(m['start'])
    end = pd.Timestamp(m['end'])

    if (
        start.tzinfo is None
        or end.tzinfo is None
        or start >= end
    ):
        raise ValueError('invalid archive period')

    def read(info):
        path = (folder / info['file']).resolve()

        if (
            path.parent != folder
            or digest(path) != info['sha256']
        ):
            raise ValueError('archive path or checksum mismatch')

        d = pd.read_csv(
            path,
            index_col='time',
            float_precision='round_trip',
        )
        d.index = pd.to_datetime(d.index, utc=True)
        d['end'] = pd.to_datetime(d['end'], utc=True)

        columns = ['o', 'h', 'l', 'c', 'v']
        invalid = (
            d.empty
            or d.index.has_duplicates
            or not d.index.is_monotonic_increasing
            or d.index.isna().any()
            or d['end'].isna().any()
            or not np.isfinite(d[columns].to_numpy()).all()
            or (d['end'] <= d.index).any()
            or (d[['o', 'h', 'l', 'c']] <= 0).any().any()
            or (d['v'] < 0).any()
            or (
                d['h'] < d[['o', 'l', 'c']].max(axis=1)
            ).any()
            or (
                d['l'] > d[['o', 'h', 'c']].min(axis=1)
            ).any()
        )
        if invalid:
            raise ValueError('invalid archived candles')

        if 'tb' in d and (
            not np.isfinite(d['tb']).all()
            or (d['tb'] < 0).any()
            or (d['tb'] > d['v']).any()
        ):
            raise ValueError('invalid archived taker volume')

        return d

    data = {}

    for key, entry in m['assets'].items():
        if key not in ('BTC', 'ETH', 'SOL', 'GOLD', 'NDX'):
            raise ValueError('unknown archived asset')

        frames = entry['frames']
        if (
            not all(tf in frames for tf in ('15m', '1h', '4h'))
            or any(tf not in TFS for tf in frames)
        ):
            raise ValueError('invalid archive timeframes')

        data[key] = {
            tf: read(info)
            for tf, info in frames.items()
        }
        data[key].update({
            k: v
            for k, v in entry['metadata'].items()
            if k in ('_source', '_provider', '_symbol')
        })

        if not all(
            k in data[key]
            for k in ('_source', '_provider', '_symbol')
        ):
            raise ValueError('missing archived source metadata')

    gc = read(m['gc']) if m.get('gc') else None

    return (
        data,
        gc,
        start,
        end,
        m.get('load_errors', {}),
    )