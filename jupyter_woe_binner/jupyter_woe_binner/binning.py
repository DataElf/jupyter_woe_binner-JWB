#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np


def calculate_woe_iv(series, target, bins, event_flag=1, non_event_flag=0, min_val=0.5):
    df = pd.DataFrame({'var': series, 'target': target})
    df['bin'] = pd.cut(series, bins=bins, right=True, include_lowest=True, precision=0)

    total_event = (target == event_flag).sum()
    total_nonevent = (target == non_event_flag).sum()

    grouped = df.groupby('bin', observed=False)

    event = grouped['target'].apply(lambda x: (x == event_flag).sum()).rename('event')
    nonevent = grouped['target'].apply(lambda x: (x == non_event_flag).sum()).rename('nonevent')

    stats = pd.DataFrame({'event': event, 'nonevent': nonevent}).fillna(0)
    stats['total'] = stats['event'] + stats['nonevent']

    total_sum = stats['total'].sum()
    stats['pct_total'] = stats['total'] / total_sum if total_sum > 0 else 0

    stats['pct_event'] = stats['event'] / total_event
    stats['pct_nonevent'] = stats['nonevent'] / total_nonevent

    stats['low_rate'] = stats['event'] / stats['total']
    stats['odds'] = stats['nonevent'] / (stats['event'] + min_val)

    stats['woe'] = np.log(
        (stats['pct_nonevent'] + min_val / total_nonevent) /
        (stats['pct_event'] + min_val / total_event)
    )

    stats['iv_part'] = (stats['pct_nonevent'] - stats['pct_event']) * stats['woe']
    total_iv = stats['iv_part'].sum()

    stats['label'] = [str(b) for b in stats.index]
    stats_reset = stats.reset_index(drop=True)
    stats_reset['index'] = range(len(stats_reset))

    stats_out = stats_reset[['index', 'label', 'nonevent', 'event', 'total', 'pct_total',
                             'pct_nonevent', 'pct_event', 'odds', 'low_rate', 'woe', 'iv_part']]
    stats_out.columns = ['Index', 'Label', 'High', 'Low', 'Total', '%Total',
                         '% High', '% Low', 'Odds', 'Low Rate', 'WoE', 'IV']
    return stats_out, total_iv


def calculate_spc_woe_iv(series, target, spc_values, event_flag=1, non_event_flag=0, min_val=0.5):
    total_event = (target == event_flag).sum()
    total_nonevent = (target == non_event_flag).sum()

    rows = []
    for sv in spc_values:
        mask = series == sv
        ev = (target[mask] == event_flag).sum()
        nev = (target[mask] == non_event_flag).sum()
        tot = ev + nev
        pct_total = tot / len(series) if len(series) > 0 else 0
        pct_event = ev / total_event if total_event > 0 else 0
        pct_nonevent = nev / total_nonevent if total_nonevent > 0 else 0
        low_rate = ev / tot if tot > 0 else 0
        odds = nev / (ev + min_val)
        woe = np.log(
            (pct_nonevent + min_val / total_nonevent) /
            (pct_event + min_val / total_event)
        ) if total_event > 0 and total_nonevent > 0 else 0
        iv_part = (pct_nonevent - pct_event) * woe
        rows.append({
            'Index': 0, 'Label': f'[{sv}, {sv}]', 'High': nev, 'Low': ev,
            'Total': tot, '%Total': pct_total,
            '% High': pct_nonevent, '% Low': pct_event,
            'Odds': odds, 'Low Rate': low_rate, 'WoE': woe, 'IV': iv_part
        })

    spc_df = pd.DataFrame(rows)
    return spc_df
