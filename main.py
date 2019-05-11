import numpy as np
import pandas as pd
import sys


# copulas module: github.com/DAI-Lab/Copulas

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    print(df.head())
