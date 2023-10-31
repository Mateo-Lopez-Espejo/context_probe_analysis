Context Probe Analysis
==============================
This is the code base asociated with the publication
[A sparse code for natural sound context in auditory cortex](https://doi.org/10.1101/2023.06.14.544866)

## Ongoing beautification effort
The functions of this code base are already cristalized since the analysis they
perform has been tested, and the output of this analysis is published. However,
this is the ongoing work of many years, and as such it has accumulated detritus
and code smells (the code works, its just ugly). There is an ongoing effort to 
beautify the code (without altering its function) to make it more accessible.

In order to run the code you need some data which is too big to have published
on github. Do not hesitate on reaching out and asking for this data.

In the near future, I hope to simplify and compress the data required, so it
can be published alongside the code.

Project Organization
------------

    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── consolidated_tstat      <- joblib cache for the t statistic significance analysis. One file per site
    │   ├── sound_files             <- .wav files used for stim generation
    │   ├── sound_quantification    <- sound spectrograms and metric table
    │   └── tensors                 <- joblib cache for site raster, folded into 5 dim arrays
    │                                        Rep * Neuron * Context * Probe * Time 
    │
    ├── notebooks          <- Exploratory Jupyter notebooks. Naming convention 
    │                         is <dateYYMMDD>_<name> Naming convention is a number (for ordering),
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── scripts            <- Collection of (old) scripts to enqueue jobs in the lab cluster
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts / functions to loand and preprocess data
        │   └── make_dataset.py
        │
        ├── dim_redux      <- Scripts / functions to reduce (neuron) dimensionality of recording sites
        │
        ├── metrics        <- Scripts / functions calculate contex effect and other metrics
        │
        ├── models         <- Scripts / functions to describe and run model trainings.
        │
        ├── pupil          <- Scripts / functions process pupil information 
        │
        ├── utils          <- mixed bag of usefull Scripts / functions. 
        │
        └── visualization  <- Plotting functions.
            └── visualize.py

--------