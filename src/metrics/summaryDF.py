import itertools as itt
from warnings import warn

import joblib as jl
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.region_map import region_map
from src.metrics.delta_fr import pairwise_delta_FR
from src.metrics.consolidated_tstat import tstat_cluster_mass
from src.metrics.significance import _significance
from src.metrics.time_series_summary import metrics_to_DF
from src.utils.dataframes import ndim_array_to_long_DF


def create_summary_DF(sites, loading_functions, cluster_thresholds, alpha, montecarlo, raster_meta,
                      metrics, sources, multiple_corrections, DF_file, recacheDF=True, diff_metrics=('T-Score')):

    print(f'all sites: \n{sites}\n')
    if DF_file.exists() and not recacheDF:
        DF = jl.load(DF_file)
        ready_sites = set(DF.site.unique())
        sites = sites.difference(ready_sites)
        print('appening new sites to existing DF', sites)
        to_concat = [DF,]
    else:
        DF_file.parent.mkdir(parents=True, exist_ok=True)
        to_concat = list()

    for site, fname, clust_thresh in tqdm(itt.product(
            sites, loading_functions, cluster_thresholds),
            total=len(sites)*len(loading_functions)*len(cluster_thresholds)):

        if tstat_cluster_mass.check_call_in_cache(
                site,cluster_threshold=float(clust_thresh), montecarlo=montecarlo, raster_meta=raster_meta, load_fn=fname):

            tstat, clust_quant_pval, goodcells, shuffled_eg = tstat_cluster_mass(
                site,cluster_threshold=float(clust_thresh), montecarlo=montecarlo, raster_meta=raster_meta, load_fn=fname)
        else:
            print(f'{site}, {fname}, {clust_thresh} not yet in cache, skipping')
            continue

        # for analysis with dimensionality reduction, changes the cellname to nan for proper dimension labeling.
        if fname == 'SC':
            chan_name = goodcells
        elif fname == 'PCA':
            chan_name = list(goodcells.keys())
        else:
            raise ValueError(f'unknown loading funciton name {fname}')

        # literal contexts and probe for dimlabdict
        contexts = list(range(0, tstat.shape[2] + 1))
        probes = list(range(1, tstat.shape[2] + 1))

        # creates label dictionary
        dim_labl_dict = {'id': chan_name,
                         'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
                         'probe': probes,
                         'time': np.linspace(0, tstat.shape[-1] / raster_meta['raster_fs'], tstat.shape[-1],
                                             endpoint=False) * 1000}

        # iterates over real data and shuffled example
        for source in sources:
            # saves the raw p values  independent of the different  metrics, but for both real and shuffled calculations
            if source == 'real':
                pvals = clust_quant_pval['pvalue']
            elif source == 'shuffled_eg':
                pvals = shuffled_eg['pvalue']
            else:
                raise ValueError(f'unrecoginzed source {source}')

            # keeps raw p values
            pval_lbl_dict = dim_labl_dict.copy()
            pval_lbl_dict.pop('time')
            min_pval = np.min(pvals, axis=-1)
            df = ndim_array_to_long_DF(min_pval, pval_lbl_dict)
            df['metric'] = 'pvalue'
            df['analysis'] = fname
            df['site'] = site
            df['region'] = region_map[site]
            df['source'] = source
            df['cluster_threshold'] = clust_thresh
            df['stim_count'] = len(probes)

            to_concat.append(df)

            # consider different multiple comparisons corrections for the significance dependent metrics
            for corr_name, corr in multiple_corrections.items():
                significance = _significance(pvals, corr, alpha=alpha)

                # Use delta FR as an alternative metric of difference
                for diff_met_name in diff_metrics:
                    if diff_met_name == 'T-score':
                        if source == 'real':
                            diff_metric = tstat
                        elif source == 'shuffled_eg':
                            diff_metric = shuffled_eg['dprime'] # this is a misnomer, it's not a dprime but a T-score
                    elif diff_met_name == 'delta_FR':
                        if source == 'real':
                            diff_metric = pairwise_delta_FR(site, raster_meta=raster_meta, load_fn=fname)
                        else:
                            warn(f'delta_FR has only real as source but {source} was given, skipping')
                            continue
                    else:
                        raise ValueError(f'unrecognized diff_metric {diff_met_name}')

                    masked_dprime = np.ma.array(diff_metric, mask=significance == 0)
                    df = metrics_to_DF(masked_dprime, dim_labl_dict, metrics=metrics)
                    df['mult_comp_corr'] = corr_name
                    df['analysis'] = fname
                    df['site'] = site
                    df['region'] = region_map[site]
                    df['source'] = source
                    df['cluster_threshold'] = clust_thresh
                    df['stim_count'] = len(probes)

                    to_concat.append(df)

    DF = pd.concat(to_concat, ignore_index=True, axis=0)

    dups = np.sum(DF.duplicated().values)
    if dups > 0:
        print(f'{dups} duplicated rows, what is wrong?, droping duplicates')
        DF.drop_duplicates(inplace=True)

    print(DF.head(10))
    print(DF.shape)
    jl.dump(DF, DF_file)

    return None