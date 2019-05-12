import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns


# copulas module: github.com/DAI-Lab/Copulas
# pycopula
# gaussianize
# hybridnaivebayes
# ambhas

# Split ranges into lower and upper bound makes us able to treat them as discrete.
# Alternative hierachical modelling
# Alternative model cumulative (But might give wrong answers?)
# mat_age Can be modelled as discrete maybe?

# Depression: (anx_band_15), (dep_band_07), (dep_band_10), (dep_band_13), (dep_band_15), dep_score, dep_thoughts, has_dep_diag

# Computer use: comp_...

# agg_score is a mess and too difficult to model as anything other than discrete, since they are sums, but could be ordinal.
# comp_house, comp_int_bed_16, comp_noint_bed_16, doesn't make sense as is either yes or NaN (IC has to be inferred)
# creat_14 actually Nominal

# Discrete / Discrete
# Ordinal / Categorical with dependance
# Nominal / two vars or just not ordered?
# Continuous / Continuous


# comp_bed_9, child has computer in bedroom, 9yo, yes, no
# comp_noint_bed_16, computer without internet permanent in bedroom, 166mo, yes
# comp_int_bed_16, computer with internet permanent in bedroom, 166mo, yes
# mat_dep, mother depressed, 8mo, 0-29
# mat_age, mother age, 0yo, mix of categorical and discrete: <16, >43, 16-43
# weight_16, child weight, 15.5yo, continous > 0
# height_16, child height, 15.5yo, continous > 0
# iq, iq, 8yo, continous > 0
## comp_wend, computer use weekend day, 16.5yo, ordinal: 'Not at all', 'Less than 1 hour', '1-2 hours', '3 or more hours'
## comp_week, computer use week day, 16.5yo, ordinal: 'Not at all', 'Less than 1 hour', '1-2 hours', '3 or more hours'
## dep_score, 0-4, sum of 4x 0-1 questions.
## Below could be combined to an ordinal variable
# pat_pres_10, paternal parent present, 122mo
# pat_pres_8, paternal parent present, 97mo
# pat_pres, paternal parent present, 47mo
# num_home, number of people in home, 47mo, mix of ordinal and discrete and 1 doesn't make sense, as 2 should be lowest?
# agg_score, aggression score (high = less aggression), sum of 3 questions scored 1-5, so range 3-18. Discrete or sum?
# pat_ses, partners social class, NOT ordinal, but categorical
# mat_ses, mothers social class, NOT ordinal, but categorical
# mat/pat_edu categorical since I don't know what the levels mean.
# parity, number of previous pregnancies, discrete 0-22!
# 
### dep_score, dep_thoughts, has_dep_diag


def print_summary(df):
    # Show everything
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        for name in df.columns.values:
            print(df[name].describe())
            print(df[name].unique())
            print()

def examine_depression(df):
    x = df['has_dep_diag'].values
    x[x == ' No ICD-10 diagnosis of depression'] = 0
    x[x == 'Yes ICD-10 diagnosis of depression'] = 1
    no_mask = (x==0)
    yes_mask = (x==1)
    y = df['dep_score'].values
    z = df['dep_thoughts'].values
    t = df['prim_diag'].values
    w = df['secd_diag'].values
    print(sum(yes_mask) / (sum(no_mask) + sum(yes_mask)))


    for arr in y,z,t,w:
        nan_mask = ~np.isnan(arr)
        n = len(np.unique(arr[nan_mask]))
        if n > 15:
            print(np.unique(arr[nan_mask]))
            quit("lol")
        for mask in no_mask, yes_mask:
            masked_arr = arr[mask * nan_mask]
            plt.hist(arr[mask], bins=n, alpha=0.5, density=True)
        plt.show()
    plt.hist(y[no_mask])
    plt.show()
    quit()
    print(np.unique(y[ no_mask]))
    print(np.unique(y[yes_mask]))
    print(np.unique(z[ no_mask]))
    print(np.unique(z[yes_mask]))
    print(np.unique(t[ no_mask]))
    print(np.unique(t[yes_mask]))
    print(np.unique(w[ no_mask]))
    print(np.unique(w[yes_mask]))

def plot_correlation(df):
    def plot_self_categorical():
        x = df[x_name]
        x_mask = ~pd.isnull(x)
        x_values = x[x_mask].values
        n_classes = np.unique(x_values).size
        sns.countplot(x_values)
        plt.title(x_name)
        plt.show()
    names = df.columns.values[3:]
    n = len(names)
    for i in range(n):
        x_name = names[i]
        if df[x_name].dtype == object:
            plot_self_categorical()
        else:
            print(x_name)
            quit(df[x_name].dtype)
        continue
        for j in range(i+1,n):
            y_name = names[j]
            y = df[y_name].values
            print(x_name, y_name, np.unique(x), np.unique(y))
            quit()

if __name__ == "__main__":
    df = pd.read_csv('maps-synthetic-data-v1.1.csv')
    #print_summary(df)
    #examine_depression(df)
    plot_correlation(df)
