####
# This code contains the algorithm to obtain samples from optimal reviewing intensity.
# This script read two csv files, preprocesed_weights.csv, observations_1k.csv and outputs revieiw times.
# For user-item pairs that are not reviewed in the period, we show empty string.
####

import numpy as np
import pandas as pd

Q = 1.0  # parameter q defined in eq. 8 in the paper.
T = 10.0  # number of days in the future to generate reviewing timeself.


def intensity(t, n_t, q):
    return 1.0/np.sqrt(q)*(1-np.exp(-n_t*t))


def sampler(n_t, q, T):
    t = 0
    while(True):
        max_int = 1.0/np.sqrt(q)
        t_ = np.random.exponential(1 / max_int)
        if t_ + t > T:
            return None
        t = t+t_
        proposed_int = intensity(t, n_t, q)
        if np.random.uniform(0, 1, 1)[0] < proposed_int / max_int:
            return t

if __name__ == '__main__':
    output = []
    output_file = open("output.csv", "w")
    results_hlr = pd.read_csv("hlr.duolingo.weights",index_col=0)

    results_hlr = results_hlr.set_index("label")
    right = results_hlr['value'].loc["right"]
    wrong = results_hlr['value'].loc["wrong"]
    results_hlr["n0"] = 2**(-(results_hlr['value'][3:]+results_hlr['value'].loc['bias']))
    duo_alpha = (-2**(-results_hlr['value'].loc['right'])+1)
    duo_beta = (2**(-results_hlr['value'].loc['wrong'])-1)
    #results_hlr['lexeme_id'] = results_hlr['lexeme_id'].str.slice(start=3
    hlr = results_hlr[3:].reset_index().set_index('lexeme_id')
    i = 0
    print("Generating reviewing times for user-item pairs")
    print("Maximum duration: {}, q: {}".format(T, Q))
    with open("observation_1k.csv") as f:
        f.readline()
        output_file.write(", ".join(("user-id", "lexeme_id", "review time (in days)\n")))
        i += 1
        for line in f:
            values = line.split(",")
            lid = values[2]
            n_correct = float(values[3])
            n_wrong = float(values[4]) - float(values[3])

            n_t = hlr['value'].loc[lid] *\
                            2**(-(right*n_correct+\
                                  wrong*n_wrong))
            t_rev = sampler(n_t, Q, T)
            output_file.write(", ".join((values[1], values[2], str(t_rev) if t_rev is not None else ""))+"\n")
            if i%100 == 0:
                print("Finished processing {} lines.".format(i))
            i+=1
    print("Finished generating review times.")
