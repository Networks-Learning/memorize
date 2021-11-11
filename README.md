# Memorize

This is a repository containing code and data for the paper:

> B. Tabibian, U. Upadhyay, A. De, A. Zarezade, Bernhard Schölkopf, and M. Gomez-Rodriguez. _Enhancing Human Learning via Spaced Repetition Optimization._ Proceedings of the National Academy of Sciences (PNAS), March, 2019. 

The paper is available [from PNAS website](https://www.pnas.org/content/116/10/3988) and the [supporting website](http://learning.mpi-sws.org/memorize/) also gives a description of our algorithm in a nutshell.

As a follow-up of this work, we tested a variant of the algorithm presented here (named [Select](https://github.com/Networks-Learning/spaced-selection)) in the wild by means of a Randomized Trial and found that it performed significantly better than competetive baselines. We present those findings in the following [paper](https://www.nature.com/articles/s41539-021-00105-8):

> U. Upadhyay, G. Lancashire, C. Moser and M. Gomez-Rodriguez. Large-scale randomized experiment reveals machine learning helps people learn and remember more effectively., npj Science of Learning, 6, Article number: 26 (2021).

## Pre-requisites

This code depends on the following packages:

 1. `numpy`
 2. `pandas`
 3. `matplotlib`
 4. `seaborn`
 5. `scipy`
 6. `dill`
 7. `click`
 
Apart from this, the instructions assume that the [Duolingo dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME) has been downloaded, extracted, and saved at `./data/raw/duolingo.csv`.

## Code structure

 - `memorize.py` contains the memorize algorithm.
 - `preprocesed_weights.csv` contains estimated model parameters for the [HLR model](https://github.com/duolingo/halflife-regression), as described in section 8 of supplementary materials.
 - `observations_1k.csv` contains a set of 1K user-item pairs and associated number of total/correct attempts by every user for given items. This dataset has been curated from a larger dataset released by Duolingo, available [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME).

## Execution

The code can by executed as follows:

`python memorize.py`

The code will use default parameter value (q) used in the code.

----

# Experiments with Duolingo data

## Pre-processing

Convert to Python `dict` by `user_id, lexeme_id` and pruning it for reading it:

    python dataset2dict.py ./data/raw/duolingo.csv ./data/duo_dict.dill --success_prob 0.99 --max_days 30 
    python process_raw_data.py ./data/raw/duolingo.csv ./data/duolingo_reduced.csv

## Plots

See the notebook `plots.ipynb`.
