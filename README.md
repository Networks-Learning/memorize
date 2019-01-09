# Memorize

This is a repository containing code and data for the paper:

> B. Tabibian, U. Upadhyay, A. De, A. Zarezade, Bernhard Schölkopf, and M. Gomez-Rodriguez. Optimizing Human Learning via Spaced Repetition Optimization. To appear at the Proceedings of the National Academy of Sciences (PNAS), 2019.

## Pre-requisites

This code depends on the following packages:

 1. `numpy`: Installation instructions are at [http://www.numpy.org/](http://www.numpy.org/) or `pip install numpy`.
 2. `pandas`: Installation instructions are at [https://pandas.pydata.org/](https://pandas.pydata.org/) or `pip install pandas`.

## Code structure

 - `memorize.py` contains the memorize algorithm.
 - `preprocesed_weights.csv` contains estimated model parameters for the [HLR model](https://github.com/duolingo/halflife-regression), as described in section 8 of supplementary materials.
 - `observations_1k.csv` contains a set of 1K user-item pairs and associated number of total/correct attempts by every user for given items. This dataset has been curated from a larger dataset released by Duolingo, available [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME).

 ## Execution

 The code can by executed as follows:

 `python memorize.py`

 The code will use default parameter value (q) used in the code.
