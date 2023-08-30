import itertools as itt
from math import factorial

import numpy as np
import pandas as pd

def _calculate_pupil_first_order_coefficient(funcDF):
    """
    Given a dataframe in long format describing the firing rate for individual
    neuron probe context combination, calculates the effect of pupil size on
    the overall firing rate of sound responses, i.e., the effect on the context
    independent probe responses. This is calculated for each probe averaging
    across all different contexts. This is denominated first order modulation
    coefficient.

    Furthermore, to ensure numerical stability and avoid division by zero or
    very small numbers, we use a modified modulation coefficient in which the
    values summed in the denominator are absolute values such that they cannot
    cancel each other.

    This is a very specific function and requires a dataframe with information
    on cell id, site, probe, pupil size, and the time interval (chunk) over
    which the firing rate is calculated.

    Args:
        funcDF: pandas DF with columns: 'part' , 'chunk', 'metric',
        'id', 'site', 'probe', 'pupil' and 'value'

    Returns: pandas DF in wide format with columns for pupil dependent firing
    rate and their associated modulation coefficient.

    """

    # Calculate probe firing rate averaged across all contexts, and split
    # by small and large pupil
    # Filters out unused data, removes singleton columns,
    # average across contexts
    funcDF = funcDF.query(
        "part == 'probe' "
        "and chunk == 'full'"
        "and metric == 'firing_rate'"
        # " and pupil != 'full'"
    ).drop(
        columns=['chunk', 'part', 'metric']
    ).groupby(
        ['id', 'site', 'probe', 'pupil'],
        observed=True, dropna=True
    ).agg(
        value=('value', np.nanmean)
    ).reset_index()

    # Pivots so large and small pupil are side by side as columns for later
    # calculation
    funcDF = funcDF.pivot_table(
        index=['id', 'site', 'probe'], columns='pupil', values='value',
        dropna=True, observed=True
    ).reset_index()

    # Modulation coefficient, with an absolute denominator for numerical
    # stability: avoids division by zero (nan) or very small numbers (inf)
    funcDF['mod_coeff_corrected'] = (
            (funcDF['big'] - funcDF['small']) /
            (np.abs(funcDF['big']) + np.abs(funcDF['small']))
    )

    return funcDF


def _calculate_delta_firing_rates(funcDF):
    """
    Given a dataframe in long format describing the firing rate for individual
    neuron probe context combination, calculates the delta firing rates for
    all pairs of context for every neuron probe combination.
    This specific function is geared towards calculating delta firing rates
    dependent on the pupil state associated with the recording, and requires
    a DF with columns containing this data.

    Args:
        funcDF: pandas DF with columns: 'id', 'site', 'probe', 'pupil

    Returns:

    """
    # Calculate delta firing rate for pairs of contexts

    funcDF = funcDF.query(
        "part == 'probe'"
    ).pivot_table(
        index=['id', 'site', 'chunk', 'probe', 'pupil'], columns=['context'],
        values='value', aggfunc='first', dropna=True, observed=True
    )

    n_comb = int(
        factorial(funcDF.shape[1]) /
        (2 * factorial(funcDF.shape[1] - 2))
    )

    ctx_pair_R = np.full((funcDF.shape[0], n_comb), np.nan)
    new_cols = list()

    for pr_idx, (c0, c1) in enumerate(
            itt.combinations(funcDF.columns, r=2)
    ):
        new_cols.append(f'{c0:02}_{c1:02}')
        ctx_pair_R[:, pr_idx] = funcDF.values[:, c0] - funcDF.values[:, c1]

    # Melts back into long format and ensures column typing for memory efficiency
    funcDF = pd.DataFrame(
        index=funcDF.index, columns=new_cols, data=ctx_pair_R
    ).melt(
        var_name='context_pair', value_name='delta_fr', ignore_index=False
    ).dropna().reset_index()

    funcDF['context_pair'] = funcDF['context_pair'].astype('category')

    return funcDF


def _calculate_pupil_second_order_coefficient(funcDF):
    """
    Given a dataframe in long format describing the delta firing rate for
    context instances (neuron probe context-pairs), calculates the influence of
    pupil size on the context effects calculated as this delta firing rate.
    This is denominated second order modulation coefficient.

    Considering the sign of delta firing rate is arbitrary (as the order of
    the substraction), normalizes large and small pupils signs such that the
    sign of the corresponding pupil independent delta firing rate is always
    positive.

    Furthermore, to ensure numerical stability and avoid division by zero or
    very small numbers, we use a modified modulation coefficient in which the
    values summed in the denominator are absolute values such that they cannot
    cancel each other.

    This is a very specific function and requires a dataframe with information
    on cell id, site, probe, pupil size, and the time interval (chunk) over
    which the firing rate is calculated.

    Args:
        funcDF: pandas DF with columns: 'part' , 'chunk', 'metric',
        'id', 'site', 'probe', 'pupil' and 'value'

    Returns: pandas DF in wide format with columns for pupil dependent firing
    rate and their asociated modulation coefficient.

    """

    # pivot SO to plot

    funcDF = funcDF.pivot_table(
        index=['id', 'site', 'context_pair', 'probe', 'chunk'],
        columns='pupil', values='delta_fr', aggfunc='first',
        dropna=True, observed=True
    ).reset_index()

    # Ensures all delta firing rates are "positive" such that if the pupil
    # independent "full" delta FR is negative, the small and large pupils
    # delta firing rates are flipped, this might yield negative values for the
    # pupil classified delta FR, but this is inteded.
    flipper = (funcDF.full.values > 0) * 2 - 1
    funcDF['small_flipped'] = funcDF['small'] * flipper
    funcDF['big_flipped'] = funcDF['big'] * flipper
    funcDF['full_flipped'] = funcDF['full'] * flipper

    # Modulation coefficient, with an absolute denominator for numerical
    # stability: avoids division by zero (nan) or very small numbers (inf)
    funcDF['mod_coeff_corrected'] = (
            (funcDF['big_flipped'] - funcDF['small_flipped']) /
            (np.abs(funcDF['big_flipped']) + np.abs(funcDF['small_flipped']))
    )

    # some instances of zero FR and no difference between pupils leads to nan
    # and inf mod_coeff, removes those rows
    funcDF.replace([-np.inf, np.inf], np.nan, inplace=True)
    funcDF.dropna(axis='index', inplace=True)

    return funcDF


def _filter_by_instance_significance(funcDF, filterDF):
    """
    Uses the cluster mass significance calculation for context effects to
    filter the simple firing rates and delta firing rates asociated to
    differente pupil sizes. Since the pupil related FR and dFR are calculated
    over the whole lenght of the probe duration, and over 250ms intervals, this
    funciton uses metrics with significance calculated over the same time
    intervals.

    Args:
        funcDF: pandas DF with columns 'id', 'site', 'context_pair', 'probe',
        filterDF: pandas DF of instances
        'chunk'

    Returns: pandas DF of equal as the input DF without non non-significant
    instances

    """
    # Integral calculated over 250ms time intervals, which matches the
    # pupil dependent firing rates and delta firing rates over the same
    # intervals.
    metrics = [
        'integral', 'integral_A', 'integral_B', 'integral_C', 'integral_D'
    ]
    filterDF = filterDF.query(
        f"source == 'real' "  # measured, not simulated values
        f"and metric in {metrics} "
        f"and cluster_threshold == 0.05 "  # alpha level
        f"and mult_comp_corr == 'bf_cp' "  # Bonferroni for cell and probes
        f"and analysis == 'SC' "  # Single cell values
        f"and diff_metric == 'delta_FR' "
        f"and value > 0"  # values == 0 mean non significant effects.
    ).copy()

    # matches the filter columns integral_A ... to the chunk column A...+'full'
    filterDF['chunk'] = filterDF['metric'].str.split('_').str[-1]
    filterDF.loc[:, 'chunk'].replace({'integral': 'full'}, inplace=True)

    filterDF.drop(
        columns=['source', 'cluster_threshold', 'mult_comp_corr',
                 'diff_metric',
                 'analysis', 'metric', 'stim_count'], inplace=True
    )
    filterDF.reset_index(drop=True, inplace=True)

    for col in ['id', 'context_pair', 'probe', 'site', 'region', 'chunk']:
        filterDF[col] = filterDF[col].astype('category')

    filterDF['value'] = pd.to_numeric(filterDF.value, downcast='float')
    filterDF.head()

    # Filter by inner join
    funcDF = pd.merge(
        funcDF, filterDF, on=['id', 'site', 'context_pair', 'probe', 'chunk'],
        validate='m:1'
    )

    return funcDF
