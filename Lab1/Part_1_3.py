import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import svd
from sklearn.model_selection import KFold
import argparse

def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--folds', type=int, default = 10)
    # parser.add_argument('--neighbours', type=int, default=10)
    # parser.add_argument('--max_dim', type=int, default=2000)
    # parser.add_argument('--dim_step', type=int, default=1)
    # parser.add_argument('--train_share', type=float, default=0.8)
    args = parser.parse_args()
    main(args)