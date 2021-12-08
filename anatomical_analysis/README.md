### Anatomical analysis

This directory has code, data, and analysis notebooks for the morphology and connectivity analysis of [Structure and Function of Axo-axonic Inhibition](https://elifesciences.org/articles/73783#annotations) (Schneider-Mizell et al. eLife 2021).

Note that much of the basic data for this dataset (e.g. whole-neuron meshes or synapse tables) can be found on [Microns Explorer](https://www.microns-explorer.org) and are beyond the scope of this repo.

The directories are split according to:
* `src` : The preprocessing and data extraction steps.
Unfortunately, much of the input to this code was a database that is no longer running.
Similar data can be extracted from available files on microns explorer, but the precise code that was used originally will not work as-is.
Nonetheless, it is provided to demonstrate the precise transformations applied in the data analysis.

* `data` : Processed data. 

* `notebooks` : Analysis notebooks.


If you have questions about the anatomical analysis here, please contact Casey Schneider-Mizell (caseys@alleninstitute.org)