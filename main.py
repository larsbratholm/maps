

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



## comp_wend, computer use weekend day, 16.5yo, ordinal: 'Not at all', 'Less than 1 hour', '1-2 hours', '3 or more hours'
## comp_week, computer use week day, 16.5yo, ordinal: 'Not at all', 'Less than 1 hour', '1-2 hours', '3 or more hours'

TEAM_NAME = "Conan the Barbayesian"
VERSION = "3.6.8"

def load_data(path):
    """
    Loads the csv file as a pandas dataframe
    """
    import pandas as pd
    return pd.read_csv(path)

def missing_001(data):
    """
    Adds a new column named 'n_ignore' that indicates where either comp_wend, comp_week or has_dep_diag are missing.
    """
    import numpy as np
    mask = (data.comp_wend.isnull() | data.comp_week.isnull() | data.has_dep_diag.isnull())
    n = len(data)
    values = np.zeros(n, dtype=bool)
    values[mask] = True
    data['n_ignore'] = values

    return data

def depression_001(data):
    """
    Adds a column named 'depression' to the dataframe.
    Depression is just defined as whether or not a person has been diagnosed with depression.
    """
    import numpy as np

    # Make masks to indicate depression
    mask = (data.has_dep_diag == "Yes ICD-10 diagnosis of depression")
    # Construct depression column
    n = len(data)
    values = np.zeros(n, dtype=bool)
    values[mask] = True
    data['depression'] = values

    return data

def computer_use_001(data):
    """
    Adds a column named 'comp_use' to the dataframe.
    Computer use is defined somewhat arbitrarily, but should be clear from the report.
    """
    import numpy as np

    # Make masks to indicate computer use
    mask_comp_wend_0 = (data.comp_wend == "Not at all")
    mask_comp_wend_1 = (data.comp_wend == "Less than 1 hour")
    mask_comp_wend_2 = (data.comp_wend == "1-2 hours")
    mask_comp_wend_3 = (data.comp_wend == "3 or more hours")
    mask_comp_week_0 = (data.comp_week == "Not at all")
    mask_comp_week_1 = (data.comp_week == "Less than 1 hour")
    mask_comp_week_2 = (data.comp_week == "1-2 hours")
    mask_comp_week_3 = (data.comp_week == "3 or more hours")

    # Low/High computer use according to some arbitrary definition
    mask_comp_low = (mask_comp_wend_0 | mask_comp_wend_1) & (mask_comp_week_0 | mask_comp_wend_1)
    mask_comp_high = (mask_comp_wend_3 & mask_comp_week_3) | (mask_comp_wend_2 & mask_comp_week_3) | (mask_comp_wend_3 & mask_comp_week_2)

    # Construct computer use column
    n = len(data)
    values = np.empty(n, dtype='|U8')
    # Set computer use that is neither high or low as other
    values[:] = "other"
    values[mask_comp_low] = "low"
    values[mask_comp_high] = "high"
    data['comp_use'] = values

    return data

def missing_002(data):
    """
    Update ignore column to include all entries where computer use is neither high or low
    """
    mask = (data.comp_use == "other")
    values = data.n_ignore.values
    values[mask] = True
    data['n_ignore'] = values
    return data

# TODO cluster computer use
# TODO get conditional distribution
# TODO figure out missing
# TODO add computer use to model

def specify_model(data):
    """
    Create simple MLE model
    """
    import numpy as np
    import bayespy

    # Create some masks of the data
    mask_depression = (data.depression == True)
    mask_high_computer_use = (data.comp_use == 'high')
    depression_observables = mask_depression.values[:,None]
    computer_use_observables = mask_high_computer_use.values[:,None]

    # Keep only data points that we chose not to ignore
    mask = (~data.n_ignore).values


    # Hidden nodes
    n_hidden_nodes = 1
    # Data size
    n_samples = data.shape[0]
    # Probability for specific hidden nodes (plates = (n_hidden_nodes,)
    # This correspond to the weight prefactor of the independent mixture model
    # 1/2 is Jeffreys prior
    p_hidden_node = bayespy.nodes.Dirichlet([1/2 for n in range(n_hidden_nodes)])
    # Observable for which mixture a sample belong to. Hidden. (plates = (n_samples, 1))
    obs_hidden_node = bayespy.nodes.Categorical(p_hidden_node, plates=(n_samples, 1))

    print(obs_hidden_node.get_values())
    quit()
    # Probability of depression in each node (plates = (1, n_hidden_nodes)
    p_depression = bayespy.nodes.Beta([1/2,1/2], plates=(1,n_hidden_nodes))
    obs_depression = bayespy.nodes.Mixture(obs_hidden_node, bayespy.nodes.Bernoulli, p_depression)
    # Probability of high computer use in each node (plates = (1, n_hidden_nodes)
    p_computer_use = bayespy.nodes.Beta([1/2,1/2], plates=(1,n_hidden_nodes))
    obs_computer_use = bayespy.nodes.Mixture(obs_hidden_node, bayespy.nodes.Bernoulli, p_computer_use)
    # Construct model
    model = bayespy.inference.VB(obs_hidden_node, p_hidden_node, obs_depression, p_depression, obs_computer_use, p_computer_use)
    # Break symmetry
    p_depression.initialize_from_random()
    p_computer_use.initialize_from_random()
    # Observe data
    obs_depression.observe(depression_observables[:,None], mask=mask[:,None])
    obs_computer_use.observe(computer_use_observables[:,None], mask=mask[:,None])
    model.update(repeat=1000, verbose=True, tol=1e-6)
    print(p_depression)
    print(p_computer_use)
    print(p_hidden_node)
    quit()


    # Get the probabilities of having depression given high computer use (p_YH),
    # depression given low computer use (p_YL), no depression given high computer use
    # (p_NH) and no depression given low computer use (p_NL)
    p_YH = count_comp_high_dep_yes / (count_comp_high_dep_no + count_comp_high_dep_yes)
    p_NH = count_comp_high_dep_no / (count_comp_high_dep_no + count_comp_high_dep_yes)
    p_YL = count_comp_low_dep_yes / (count_comp_low_dep_no + count_comp_low_dep_yes)
    p_NL = count_comp_low_dep_no / (count_comp_low_dep_no + count_comp_low_dep_yes)

    # Assertion that the probabilities add up
    assert abs(p_YH + p_NH - 1) < 1e-6
    assert abs(p_YL + p_NL - 1) < 1e-6

    # Log-odds-ratio
    L = np.log(p_YH) - np.log(p_NH) - np.log(p_YL) + np.log(p_NL)
    # Variance
    se = np.sqrt(1/count_comp_low_dep_no + 1/count_comp_low_dep_yes + 1/count_comp_high_dep_no + 1/count_comp_high_dep_yes)
    # Odds-ratio (odds of depression given high computer use over odds of depression given low computer use)
    OR = np.exp(L)
    # Confidence interval
    ci = (np.exp(L-1.96*se), np.exp(L+1.96*se))
    # 2-sided p-value
    p_value = ss.norm.sf(abs(L/se))*2

    # Just calculate the maximum likelihood for the remaining data, with a model
    # placing each point in 4 categories as above. I don't really have any hidden variables
    # so I don't think an information criterion makes sense on it's own
    log_likelihood = count_comp_low_dep_no * np.log(p_NL) \
                   + count_comp_low_dep_yes * np.log(p_YL) \
                   + count_comp_high_dep_no * np.log(p_NH) \
                   + count_comp_high_dep_yes * np.log(p_YH)

    # 4 categories, so 3 variables
    aic = 2 * 3 - 2 * log_likelihood

    # Make dictionary for results
    results = {}
    # There is no model object
    #results['mod'] = None
    # odds ratio
    results['or_1'] = OR
    # confidence interval as a tuple
    results['ci_1'] = ci
    # 2-sided p-value
    results['p_1'] = p_value
    # I assume that multiple odds ratio are optional?
    #results['or_2'] = None
    #results['ci_2'] = None
    #results['p_2'] = None
    # AIC
    results['AIC'] = aic

    return data, results


if __name__ == "__main__":
    path = 'maps-synthetic-data-v1.1.csv'
    data = load_data(path)
    data = missing_001(data)
    data = depression_001(data)
    data = computer_use_001(data)
    data = missing_002(data)
    data, results = specify_model(data)
    print(results)
