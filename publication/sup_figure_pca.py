import plotly.graph_objects as go
import numpy as np
import pandas as pd

from src.dim_redux.PCA import load_site_formated_PCs
from src.utils.subsets import good_sites
from src.visualization.palette import (
    add_opacity, Black, REGION_COLORMAP
)
from src.visualization.interactive import plot_site_coverages
from publication.globals import RASTER_META, PCA_DF, MINIMAL_DF

recache = False
eg_site = 'ARM021b'  # site from the example cells in figure 2


def plot_cumulative_variances_explained():
    # load all principal components and organize in a working dataframe
    # very fast step if its cached

    pca_DF = list()
    for site in good_sites:
        # only loads the cache if it exists

        if load_site_formated_PCs.check_call_in_cache(
                site, recache_rec=True, **RASTER_META
        ):
            _, PCs = load_site_formated_PCs(
                site, recache_rec=True, **RASTER_META
            )
        else:
            print(f"PCA for {site} not found in cache")

        pca_DF.append({'site': site, 'exp_var': list(PCs.values())})

    pca_DF = pd.DataFrame(pca_DF)
    pca_DF['n_PCs'] = pca_DF['exp_var'].apply(lambda x: len(x))

    # add site region information by merging
    pca_DF = pd.merge(
        left=pca_DF,
        right=MINIMAL_DF.loc[:,
              ['site', 'region']
              ].drop_duplicates().reset_index(drop=True),
        on='site'
    )

    fig = go.Figure()

    for region in pca_DF.region.unique():
        innerdf = pca_DF.query(f"region == '{region}'")
        color = REGION_COLORMAP[region]

        padded_vars = np.full((innerdf.shape[0], innerdf['n_PCs'].max()),
                              np.nan)
        for rr, (_, row) in enumerate(innerdf.iterrows()):
            cum_var_exp = np.cumsum(row.exp_var)
            padded_vars[rr, :len(cum_var_exp)] = cum_var_exp
            fig.add_trace(
                go.Scatter(x=np.arange(1, len(cum_var_exp) + 1),
                           y=cum_var_exp, mode='lines',
                           line=dict(color=add_opacity(color, 0.3),
                                     width=1),
                           showlegend=False)
            )

        fig.add_trace(
            go.Scatter(x=np.arange(1, padded_vars.shape[1] + 1),
                       y=np.nanmean(padded_vars, axis=0), mode='lines',
                       line=dict(color=color, width=2),
                       name=region, showlegend=True)
        )

        fig.add_hline(y=0.9, line=dict(color='black', width=1, dash='dash'))

    # add the example site
    example = np.cumsum(
        pca_DF.query(f"site == '{eg_site}'")['exp_var'].iloc[0])

    fig.add_trace(
        go.Scatter(x=np.arange(1, len(example) + 1),
                   y=example, mode='lines',
                   line=dict(color=Black, width=1),
                   name='example site', showlegend=True)
    )

    h, w = 1.6, 2.9
    fig.update_layout(template='simple_white', width=96 * w, height=96 * h,
                      margin=dict(l=10, r=10, t=10, b=10),
                      font_family='Arial',
                      showlegend=True,
                      legend=dict(x=1, y=0, xanchor='right', yanchor='bottom',
                                  font=dict(size=8, color="black")),
                      xaxis=dict(title=dict(text="principal component #",
                                            font_size=10,
                                            standoff=0),
                                 tickfont_size=9),
                      yaxis=dict(
                          title=dict(text="explained variance",
                                     font_size=10,
                                     standoff=0),
                          tickfont_size=9)
                      )

    return fig


def plot_example_pca_coverages():
    # ToDo Document
    # Pic top 4 and bottom 4 principal components
    # find how many PC are in the selected site
    toplot_pca = PCA_DF.query(f"site == '{eg_site}'").copy()

    available_PCs = toplot_pca.PC.unique()
    selected_PCs = available_PCs[[0, 1, 2, 3, -4, -3, -2, -1]].tolist()
    toplot_pca.query(f"PC in {selected_PCs}", inplace=True)

    fig = plot_site_coverages(toplot_pca, rows=2, cols=4)

    h, w = 3.1, 2.9
    fig.update_layout(width=96 * w, height=96 * h)

    return fig
