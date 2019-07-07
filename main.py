

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

def transformation_001(data):
    """
    Do transformations to data
    """
    import numpy as np

    def transform_has_dep_diag(data):
        # Bool indicator
        data.replace('Yes ICD-10 diagnosis of depression', True, inplace=True)
        data.replace(' No ICD-10 diagnosis of depression', False, inplace=True)

    def transform_comp_wend(data):
        # Ordinal variables
        data.replace('Not at all', 0, inplace=True)
        data.replace('Less than 1 hour', 1, inplace=True)
        data.replace('1-2 hours', 2, inplace=True)
        data.replace('3 or more hours', 3, inplace=True)

    def transform_comp_week(data):
        # Ordinal variables
        data.replace('Not at all', 0, inplace=True)
        data.replace('Less than 1 hour', 1, inplace=True)
        data.replace('1-2 hours', 2, inplace=True)
        data.replace('3 or more hours', 3, inplace=True)

    transform_has_dep_diag(data)
    transform_comp_wend(data)
    transform_comp_week(data)

    return data

# TODO seed clusters
# TODO cluster computer use
# TODO get conditional distribution
# TODO add computer use to model
# TODO check how priors affect results
# TODO try to marginalize parameters?
# TODO try different optimization strategies?
# TODO simulated annealing solver to avoid local minima
# TODO stochastic inference for faster learning and avoiding local minima?

def specify_model(data):
    """
    Create bayesian model
    """
    import numpy as np
    import bayespy
    import pomegranate

    x = np.asarray([[0,0,1]*10000, [0,1,0]*10000]).T
    x = np.random.randint(0,4,size=1000)

    dist0 = pomegranate.DiscreteDistribution({0:1, 1:0, 2:0, 3:0})
    dist1 = pomegranate.DiscreteDistribution({0:0, 1:1, 2:0, 3:0})
    dist2 = pomegranate.DiscreteDistribution({0:0, 1:0, 2:1, 3:0})
    dist3 = pomegranate.DiscreteDistribution({0:0, 1:0, 2:0, 3:1})
    dist23 = pomegranate.GeneralMixtureModel([dist2, dist3], weights = [0.4,0.6])
    dist123 = pomegranate.GeneralMixtureModel([dist23, dist1], weights = [0.4,0.6])
    model = pomegranate.GeneralMixtureModel([dist123, dist0], weights = [0.4,0.6])
    model.fit(x)
    quit()

    i1 = pomegranate.IndependentComponentsDistribution([pomegranate.BernoulliDistribution(np.random.random()), pomegranate.BernoulliDistribution(np.random.random())])
    i2 = pomegranate.IndependentComponentsDistribution([pomegranate.BernoulliDistribution(np.random.random()), pomegranate.BernoulliDistribution(np.random.random())])
    model = pomegranate.GeneralMixtureModel([1,i2], weights = [0.4,0.6])
    model.fit(x)
    print(model)



    quit()

    # Hidden nodes in mixture model
    n_hidden_nodes = 1
    # Number of clusters for computer use
    n_computer_use_clusters = 2




    # Assertions on user values
    assert n_hidden_nodes >= 1
    assert isinstance(n_hidden_nodes, int)
    assert n_computer_use_clusters >= 2
    assert isinstance(n_computer_use_clusters, int)

    # Data size
    n_samples = data.shape[0]


    # Dictionary to keep all the bayespy.nodes objects, since we're going to have many of them
    nodes = {}

    # Create categorical indicator variable to define computer use (just clustering with a mixture model)
    # 0.5 corresponds to Jeffrey's prior
    nodes["p_computer_use_classification"] = bayespy.nodes.Dirichlet([0.5 for n in range(n_computer_use_clusters)])
    # Observable for which cluster a sample belong to. Hidden. (plates = (n_samples, 1))
    nodes["obs_computer_use_classification"] = bayespy.nodes.Categorical(nodes["p_computer_use_classification"], plates=(n_samples, 1))
    # comp_week is an ordinal variable, so need to first model if X = 0 is observed, or X > 0.
    # Then if X>0, we need to check if X = 1 or X > 1. This requires the same number of parameters as modelling them as independent
    # categories, but captures some of the neighbouring effects between the categories.
    nodes["p_comp_week_0"] = bayespy.nodes.Beta([0.5,0.5], plates=(1,n_computer_use_clusters))
    nodes["p_comp_week_1"] = bayespy.nodes.Beta([0.5,0.5], plates=(1,n_computer_use_clusters))
    nodes["p_comp_week_2"] = bayespy.nodes.Beta([0.5,0.5], plates=(1,n_computer_use_clusters))
    # Observables of comp_week
    nodes["obs_comp_week_0"] = bayespy.nodes.Mixture(nodes["obs_computer_use_classification"], bayespy.nodes.Bernoulli, nodes["p_comp_week_0"])
    nodes["obs_comp_week_1"] = bayespy.nodes.Mixture(nodes["obs_computer_use_classification"], bayespy.nodes.Bernoulli, nodes["p_comp_week_1"])
    nodes["obs_comp_week_2"] = bayespy.nodes.Mixture(nodes["obs_computer_use_classification"], bayespy.nodes.Bernoulli, nodes["p_comp_week_2"])
    # Same procedure for comp_wend
    nodes["p_comp_wend_0"] = bayespy.nodes.Beta([0.5,0.5], plates=(1,n_computer_use_clusters))
    nodes["p_comp_wend_1"] = bayespy.nodes.Beta([0.5,0.5], plates=(1,n_computer_use_clusters))
    nodes["p_comp_wend_2"] = bayespy.nodes.Beta([0.5,0.5], plates=(1,n_computer_use_clusters))
    nodes["p_comp_wend_0"] * nodes["p_comp_wend_1"]
    quit()
    # Observables of comp_wend
    nodes["obs_comp_week_0"] = bayespy.nodes.Mixture(nodes["obs_computer_use_classification"], bayespy.nodes.Bernoulli, nodes["p_comp_week_0"])
    nodes["obs_comp_week_1"] = bayespy.nodes.Mixture(nodes["obs_computer_use_classification"], bayespy.nodes.Bernoulli, nodes["p_comp_week_1"])
    nodes["obs_comp_week_2"] = bayespy.nodes.Mixture(nodes["obs_computer_use_classification"], bayespy.nodes.Bernoulli, nodes["p_comp_week_2"])
    # Initialize the nodes such that node0 will be low computer use and node1 will be high computer use
    # and any other node will be intemediate values
    nodes["p_computer_use_classification"].initialize_from_random()
    quit()
    #nodes["p_comp_week_0"].initialize_from_value([0.99 - 0.01 * (n_computer_use_clusters - 2)]
    #                                               + [0.01 for n in range(n_computer_use_clusters - 1)])
    #nodes["p_comp_week_1"].initialize_from_random()
    #nodes["p_comp_week_2"].initialize_from_random()


    # Create some masks of the data
    mask_depression = (data.depression == True)
    mask_high_computer_use = (data.comp_use == 'high')
    depression_observables = mask_depression.values[:,None]
    computer_use_observables = mask_high_computer_use.values[:,None]

    # Keep only data points that we chose not to ignore
    mask = (~data.n_ignore).values[:,None]

    # For testing
    #depression_observables = np.asarray([0,0,1]*1000)[:,None]
    #computer_use_observables = np.asarray([0,1,0]*1000)[:,None]
    #mask = np.asarray([True]*3000)[:,None]

    # Probability for specific hidden nodes (plates = (n_hidden_nodes,)
    # This correspond to the weight prefactor of the independent mixture model
    p_hidden_node = bayespy.nodes.Dirichlet([0.5 for n in range(n_hidden_nodes)])
    # Observable for which mixture a sample belong to. Hidden. (plates = (n_samples, 1))
    obs_hidden_node = bayespy.nodes.Categorical(p_hidden_node, plates=(n_samples, 1))

    # Probability of depression in each node (plates = (1, n_hidden_nodes)
    p_depression = bayespy.nodes.Beta([0.5,0.5], plates=(1,n_hidden_nodes))
    obs_depression = bayespy.nodes.Mixture(obs_hidden_node, bayespy.nodes.Bernoulli, p_depression)
    # Probability of high computer use in each node (plates = (1, n_hidden_nodes)
    p_computer_use = bayespy.nodes.Beta([0.5,0.5], plates=(1,n_hidden_nodes))
    obs_computer_use = bayespy.nodes.Mixture(obs_hidden_node, bayespy.nodes.Bernoulli, p_computer_use)
    # Construct model
    model = bayespy.inference.VB(obs_hidden_node, p_hidden_node, obs_depression, p_depression, obs_computer_use, p_computer_use)
    # Break symmetry by initializing from random
    p_depression.initialize_from_random()
    p_computer_use.initialize_from_random()
    p_hidden_node.initialize_from_random()
    # Observe data
    obs_depression.observe(depression_observables, mask=mask)
    obs_computer_use.observe(computer_use_observables, mask=mask)
    # Fit model
    model.update(repeat=10000, verbose=True, tol=1e-6)
    #model.optimize(obs_depression, p_depression, obs_computer_use, p_computer_use, maxiter=10000, tol=1e-6, verbose=True, collapsed=[obs_hidden_node, p_hidden_node])
    print(p_depression)
    print(p_computer_use)
    print(p_hidden_node)
    #print(p_computer_use.get_parameters())
    #print(obs_computer_use.integrated_logpdf_from_parents([0,1],0)[0:3])
    #obs_computer_use.unobserve()
    #obs_depression.unobserve()
    #obs_depression.observe(np.zeros((n_samples,1), dtype=int))
    #model2 = bayespy.inference.VB(obs_hidden_node, obs_depression)
    #model2.update(repeat=1000, verbose=True, tol=1e-7)
    #print(obs_computer_use.integrated_logpdf_from_parents([0,1],0)[0:3])
    #obs_depression.unobserve()
    #computer_use.unobserve()
    #p_computer_use.observe([1,1,1])

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
    data = transformation_001(data)
    data, results = specify_model(data)
    print(results)
