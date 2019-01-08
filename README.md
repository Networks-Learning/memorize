# Memorize

This is a repository containing code for the paper:

> B. Tabibian, U. Upadhyay, A. De, A. Zarezade, Bernhard Schölkopf, and M. Gomez-Rodriguez. Optimizing Human Learning. [arXiv:1712.01856](https://arxiv.org/abs/1712.01856)

## Pre-requisites

This code depends on the following packages:

 1. `numpy`: Installation instructions are at [http://www.numpy.org/](http://www.numpy.org/) or `pip install numpy`.
 2. `pandas`: Installation instructions are at [https://pandas.pydata.org/](https://pandas.pydata.org/) or `pip install pandas`.

## Code structure

 - `memorize.py` contains the algorithm to obtain samples from optimal reviewing intensity.
 - `preprocesed_weights.csv` contains weights for [HLR model](https://github.com/duolingo/halflife-regression) as described in section 8 of supplementary materials.
 - `observations_1k.csv` contains a set of 1K user-item pairs and associated number of total/correct attempts by every user for given items.

 ## Execution

 The code can by executed as follows:

 `python memorize.py`

 The code will use default parameter value (q) used in the code.
