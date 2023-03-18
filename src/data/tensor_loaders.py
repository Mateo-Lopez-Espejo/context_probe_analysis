from src.data.rasters import load_site_formated_raster
from src.dim_redux.PCA import load_site_formated_PCs
from src.dim_redux.dPCA import load_site_formated_dPCs
from src.data.diagonalization import load_site_dense_raster

"""
One cannot pass a cached function as a parameter to another cached function.
dirtbag hack to have a simple name for all loader/preprocesors that output a tensor of shape  rep x neu x ctx x prb x tme
this so they can be cached with joblib and the reference to them is only a string.
This is necessary so I can pass the string reference as a parameter to other functions being caches e.g the cluster mass 
analysis
"""
tensor_loaders = {'SC': load_site_formated_raster,
                  'PCA': load_site_formated_PCs,
                  'dPCA': load_site_formated_dPCs,
                  'dense': load_site_dense_raster
                  }
