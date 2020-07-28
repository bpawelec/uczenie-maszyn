import numpy as np
from numpy.random import randn
from scipy.stats import wilcoxon
import pandas as pd

if __name__ == "__main__":

    x1: np.ndarray = pd.read_csv('reduced_dataset_enn_pokerhand.csv')
    x2: np.ndarray = pd.read_csv('reduced_dataset_ncr_pokerhand.csv')
    print(type(x1.values))
    stat, p = wilcoxon(x1.values.ravel(), x2.values.ravel())
    print('Statistics=%.3f, p=%.20f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')