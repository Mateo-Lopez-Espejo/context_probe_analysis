from time import time
import pathlib as pl
from configparser import ConfigParser
import itertools as itt
from math import factorial

import joblib as jl
import pandas as pd
import numpy as np

from src.root_path import config_path

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

# Source DF
# t_statistic = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS' # old and questionable
mass_clust_pval_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220602_SC_pvals_subsample'
longDF = jl.load(mass_clust_pval_DF_file)

# Destination DF
# subsampling_path = pl.Path(config['paths']['analysis_cache']) / '220310_pval_subsamp_DF' # the output of the  old and questionable
subsampling_path = pl.Path(config['paths']['analysis_cache']) / '220310_pval_subsamp_DF_update'
recache = False
calc_site_coverage = False



if (not subsampling_path.exists()) or recache:
    print(f'creating subsamp_signif dataframe')

    # filteres dataframe and adds required columns
    to_subsample = longDF.query("analysis == 'SC' and metric == 'pvalue' "
                                "and cluster_threshold == 0.05 and stim_count == 10").copy()

    # transform the ctx_pair column (str) into tow ctxN columns (int)
    ctx_pairs = np.stack(
        [list(map(int, pair.split('_'))) for pair in to_subsample.context_pair],
        axis=0)
    to_subsample.loc[:,('ctx_0', 'ctx_1')] = ctx_pairs

    all_contexts = np.unique(ctx_pairs)
    all_probes = to_subsample.probe.sort_values().unique()

    # all_contexts = [0, 1, 2]
    # all_probes = [1, 2]

    subDF = list()

    # iter over all combination os samplesizes 1, 2, 3, ... prbs * 2, 3, 4, ... ctxs
    for nctx, nprb in itt.product(range(2, len(all_contexts) + 1),
                                  range(1, len(all_probes) + 1)):
        print(f'subsampling {nctx} contexts and {nprb} probes')

        # randomly permutes to get random n first elements
        ctx_sub_samps = np.random.permutation(
            list(itt.combinations(all_contexts, nctx)))
        prb_sub_samps = np.random.permutation(
            list(itt.combinations(all_probes, nprb)))

        n_ctx_pairs = factorial(nctx) / (2 * factorial(nctx - 2))
        n_comparison = n_ctx_pairs * nprb
        alpha = 0.05 / n_comparison

        n_samps = 1000  # ideally 100??
        # joblib parallel returns a list of second groupings


        # defines inner loop as function for parallelism
        def ctx_prb_smp_grouping(ctx_smp, prb_smp, alpha):
            # subsamples the dataframe
            spl_df = to_subsample.query(f"probe in {prb_smp.tolist()} and ctx_0 in {ctx_smp.tolist()} and ctx_1 in {ctx_smp.tolist()}")

            # first grouping: gets the modulation space coverage of individual neurons
            neuron_coverage = spl_df.groupby(['source','region', 'site', 'id']).agg(
                coverage_percent=("value", lambda x: sum(x < alpha) / len(x) * 100))

            # second grouping pass: counts the total modulated neurons recruited per site
            neuron_recuitment = neuron_coverage.groupby(
                ['source', 'region', 'site']).agg(
                recruited_neu_pct=('coverage_percent', lambda x: sum(x > 0) / len(x) * 100))

            if calc_site_coverage:
                # alternatively look at the tiling of modulation space by neurons
                # This second plot is difficult to interpret, It escentially says that as you increase
                # the stimulus space, the site union is less likely to cover it all

                # first collapse across neurons in a site by taking the min pvalue for each position in modulation space
                # keeps number of neurons per site
                site_tiling = spl_df.groupby(['source', 'region', 'site', 'context_pair', 'probe']).agg(
                    pvalue=('value', 'min'), n_cells =('value', 'count'))
                # second, defines the coverage percent by finding significant position in modulation space.
                # here further corrects alpha with bonferroni correction for number of neurons
                site_tiling['significant'] = site_tiling['pvalue'] < (alpha / site_tiling['n_cells'])
                site_coverage = site_tiling.groupby(['source', 'region', 'site']).agg(
                    coverage_percent=("significant", lambda x: sum(x) / len(x) * 100))

                out_DF = pd.concat([neuron_recuitment, site_coverage], axis=1)
            else:
                out_DF = neuron_recuitment

            out_DF['n_prb'] = nprb
            out_DF['n_ctx'] = nctx

            return out_DF.reset_index()

        # iterates over all possible combinations of context and probes
        t = time()
        n_out = jl.Parallel(n_jobs=7)(jl.delayed(ctx_prb_smp_grouping)
                                            (ctx_smp, prb_smp, alpha)
                                            for ctx_smp, prb_smp in
                                            itt.islice(itt.product(ctx_sub_samps, prb_sub_samps), 0, n_samps))
        print(f'added {len(n_out)} random samples, '
              f'it took {time() - t}s')

        subDF.extend(n_out)

    subDF = pd.concat(subDF)

    jl.dump(subDF, subsampling_path)
    print(f'site_n_rec_neu chached at {subsampling_path}')
else:
    subDF = jl.load(subsampling_path)