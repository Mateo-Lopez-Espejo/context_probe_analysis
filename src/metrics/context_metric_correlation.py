import itertools as itt

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from publication.globals import RASTER_META, pivoted_f2 as pivoted

from src.metrics.time_series_summary import metrics_to_DF
from src.metrics.consolidated_tstat import tstat_cluster_mass
from src.metrics.significance import _significance
from src.metrics.delta_fr import pairwise_delta_FR
from src.utils.subsets import good_sites


def calculate_site_shuffled_time_metrics(
        siteid: str, n_shuffles: int = 100
) -> pd.DataFrame:
    """
    Loads a site context difference metric (delta FR) and the associated
    significance of said difference (cluster mass analysis). This contains
    information for all neurons in the site and all context instances for
    each neuron.
    Then shuffles together the difference and significance in time n_shuffles
    times, calculates the "last_bin" duration metric and returns a formated
    dataframe containing all the shuffled metrics for significant instances
    in the site.
    Args:
        siteid: str, site name
        n_shuffles: int, number of shuffles to perform

    Returns: pandas dataframe

    """
    # hardcodes some of the variables to ignore unused variations
    # todo some of these might become global project wide variables
    load_fn = 'SC'
    cluster_threshold = 0.05
    alpha = 0.05
    montecarlo = 11000
    # for this time shuffle, we only need to recalculate last bin as the
    # integral should remain constant
    metrics = ['last_bin']
    # Bonferroni corrections on dimensions 1 and 2 i.e. contexts and probes
    multiple_correction = [1, 2]

    # Ensurese the cluster_mass quantification exists in cache
    if tstat_cluster_mass.check_call_in_cache(
            siteid,
            cluster_threshold=cluster_threshold,
            montecarlo=montecarlo,
            raster_meta=RASTER_META,
            load_fn=load_fn
    ):

        _, clust_quant_pval, goodcells, _ = tstat_cluster_mass(
            siteid,
            cluster_threshold=cluster_threshold,
            montecarlo=montecarlo,
            raster_meta=RASTER_META,
            load_fn=load_fn
        )
    else:
        print(f"{siteid} cluster mass not found in cache, skipping")
        return None

    pvalues = clust_quant_pval['pvalue']

    significance = _significance(pvalues, multiple_correction, alpha=alpha)
    diff_metric = pairwise_delta_FR(siteid, raster_meta=RASTER_META,
                                    load_fn=load_fn)

    # defines a dictionary to label a dataframe
    # todo translate into x-array

    contexts = list(range(0, pvalues.shape[2] + 1))
    probes = list(range(1, pvalues.shape[2] + 1))

    dim_labl_dict = {
        'id': goodcells,
        'context_pair': [
            f'{c1:02d}_{c2:02d}' for c1, c2 in
            itt.combinations(contexts, 2)
        ],
        'probe': probes,
        'time': np.linspace(
            0,
            pvalues.shape[-1] / RASTER_META['raster_fs'],
            pvalues.shape[-1],
            endpoint=False
        ) * 1000
    }

    # randomly shuffles in time the difference and the significance,
    # since they must be coordinated, uses the same random indexer

    rind = np.arange(diff_metric.shape[-1])

    # initializes array for shuffled data with shape
    # (neurons*contexts_pairs*probes) x n_shuffles
    shuffled_last_bin_array = np.empty(
        [np.prod(diff_metric.shape[:3]), n_shuffles]
    )

    for ns in range(n_shuffles):
        np.random.shuffle(rind)
        masked_dprime = np.ma.array(diff_metric[..., rind],
                                    mask=significance[..., rind] == 0)
        df = metrics_to_DF(masked_dprime, dim_labl_dict, metrics=metrics)

        shuffled_last_bin_array[:, ns] = df['value'].values

    shuffled_last_bin_df = pd.DataFrame(
        index=df.set_index(['id', 'context_pair', 'probe']).index,
        columns=[f'shuffle_{n}' for n in range(n_shuffles)],
        data=shuffled_last_bin_array
    ).query("shuffle_0 > 0")

    return shuffled_last_bin_df


def calculate_metric_correlation_null_distribution(
        sites:set[str]=good_sites, n_shuffles=100
)->tuple[float, np.array, float]:
    """
    Runs the time shuffling and context duration metric calculation for all
    sites specified, organizes in a dataframe and matches with the
    context amplitude metric, which is not recalcuated since its invariant to
    time shuffling. Then, calculates the context effect metrics correlation
    for all the shuffles, and the real metric correlation, and finally,
    calculates the p-value from the null distribution and the correlation.

    Args:
        sites: set of site names
        n_shuffles: int

    Returns:
        calc_r_value: float, real Pearson's R
        null_r_distr: 1d np.array, null distribution
        pvalue: float

    """
    calc_r_value = pearsonr(pivoted['integral'], pivoted['last_bin'])[0]
    print(
        f"Pearson's R between context effect"
        f" amplitude and duration: {calc_r_value}"
    )

    # generates a dataframe of shuffled time bins for each site, and merge all
    # single site data frames into a single one
    print(
        'shuffling significant bins in time and '
        'calculating duration metric for all sites ...'
    )

    time_shuffle_DF = list()
    for site in tqdm(sites):
        time_shuffle_DF.append(
            calculate_site_shuffled_time_metrics(site, n_shuffles))

    # ensures original and shuffled values are row matched by merging
    time_shuffle_DF = pd.concat(time_shuffle_DF)
    merged = pd.merge(
        left=pivoted.loc[:,
             ('id', 'context_pair', 'probe', 'integral', 'last_bin')],
        right=time_shuffle_DF.reset_index(),
        on=['id', 'context_pair', 'probe'],
        validate="1:1"
    ).set_index(['id', 'context_pair', 'probe'])

    # Gets the null distribution of the metrics correlation by calculating
    # the R value for every calculated shuffle
    print("calculating Pearson's R for the shuffled time metrics ...")
    null_r_distr = np.empty(n_shuffles)
    for ns in tqdm(range(n_shuffles)):
        null_r_distr[ns] = pearsonr(
            merged['integral'],
            merged[f'shuffle_{ns}'],
        )[0]

    pvalue = np.sum(null_r_distr > calc_r_value) / n_shuffles
    print(f"p-value of metric correlation not being due to chance: {pvalue}")

    return calc_r_value, null_r_distr, pvalue
