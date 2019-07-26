

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

    #x = np.asarray([[0,0,1]*10000, [0,1,0]*10000]).T
    #x = np.random.randint(0,4,size=1000)

    # Hidden nodes in mixture model
    n_hidden_nodes = 4
    # Number of clusters for computer use
    n_computer_use_clusters = 4

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
    # Treat comp_week as an ordinal variable.
    nodes["p_comp_week"] = bayespy.nodes.Beta([1e-6,1e-6], plates=(n_computer_use_clusters,))
    nodes["z_comp_week"] = bayespy.nodes.Mixture(nodes["obs_computer_use_classification"], bayespy.nodes.Categorical, nodes["p_comp_week"], plates=(n_samples, 1))
    nodes["p0_comp_week"] = bayespy.nodes.Beta([1e-6,1e-6], plates=(n_computer_use_clusters,2))
    nodes["z0_comp_week"] = bayespy.nodes.MultiMixture([nodes["obs_computer_use_classification"], nodes["z_comp_week"]], bayespy.nodes.Categorical, nodes["p0_comp_week"], plates=(n_samples,1))
    nodes["p1_comp_week"] = bayespy.nodes.Beta([1e-6,1e-6], plates=(n_computer_use_clusters,2))
    nodes["z1_comp_week"] = bayespy.nodes.MultiMixture([nodes["obs_computer_use_classification"], nodes["z_comp_week"]], bayespy.nodes.Categorical, nodes["p1_comp_week"], plates=(n_samples,1))
    # Treat comp_wend as an ordinal variable.
    nodes["p_comp_wend"] = bayespy.nodes.Beta([1e-6,1e-6], plates=(n_computer_use_clusters,))
    nodes["z_comp_wend"] = bayespy.nodes.Mixture(nodes["obs_computer_use_classification"], bayespy.nodes.Categorical, nodes["p_comp_wend"], plates=(n_samples, 1))
    nodes["p0_comp_wend"] = bayespy.nodes.Beta([1e-6,1e-6], plates=(n_computer_use_clusters,2))
    nodes["z0_comp_wend"] = bayespy.nodes.MultiMixture([nodes["obs_computer_use_classification"], nodes["z_comp_wend"]], bayespy.nodes.Categorical, nodes["p0_comp_wend"], plates=(n_samples,1))
    nodes["p1_comp_wend"] = bayespy.nodes.Beta([1e-6,1e-6], plates=(n_computer_use_clusters,2))
    nodes["z1_comp_wend"] = bayespy.nodes.MultiMixture([nodes["obs_computer_use_classification"], nodes["z_comp_wend"]], bayespy.nodes.Categorical, nodes["p1_comp_wend"], plates=(n_samples,1))

    nodes["p_computer_use_classification"].initialize_from_random()


    model = bayespy.inference.VB(nodes["z0_comp_week"], nodes["p0_comp_week"], nodes["z1_comp_week"], nodes["p1_comp_week"], nodes["z_comp_week"], nodes["p_comp_week"],
                        nodes["z0_comp_wend"], nodes["p0_comp_wend"], nodes["z1_comp_wend"], nodes["p1_comp_wend"], nodes["z_comp_wend"], nodes["p_comp_wend"],
            nodes["obs_computer_use_classification"], nodes["p_computer_use_classification"])

    # Observe z_comp_week values (if comp_week >= 2)
    z_comp_week = np.zeros((n_samples, 1), dtype=int)
    z_comp_week[data["comp_week"].values.astype(int) == 2] = 1
    z_comp_week[data["comp_week"].values.astype(int) == 3] = 1
    z_comp_week_mask = ~data["comp_week"].isnull()[:,None]
    nodes["z_comp_week"].observe(z_comp_week, mask=z_comp_week_mask)
    # Observe z0_comp_week values (if comp_week is 0 or 1)
    z0_comp_week = np.zeros((n_samples, 1), dtype=int)
    z0_comp_week[data["comp_week"].values.astype(int) == 1] = 1
    nodes["z0_comp_week"].observe(z0_comp_week, mask=z_comp_week_mask)
    # Observe z1_comp_week values (if comp_week is 2 or 3)
    z1_comp_week = np.zeros((n_samples, 1), dtype=int)
    z1_comp_week[data["comp_week"].values.astype(int) == 3] = 1
    nodes["z1_comp_week"].observe(z1_comp_week, mask=z_comp_week_mask)

    # Observe z_comp_wend values (if comp_wend >= 2)
    z_comp_wend = np.zeros((n_samples, 1), dtype=int)
    z_comp_wend[data["comp_wend"].values.astype(int) == 2] = 1
    z_comp_wend[data["comp_wend"].values.astype(int) == 3] = 1
    z_comp_wend_mask = ~data["comp_wend"].isnull()[:,None]
    nodes["z_comp_wend"].observe(z_comp_wend, mask=z_comp_wend_mask)
    # Observe z0_comp_wend values (if comp_wend is 0 or 1)
    z0_comp_wend = np.zeros((n_samples, 1), dtype=int)
    z0_comp_wend[data["comp_wend"].values.astype(int) == 1] = 1
    nodes["z0_comp_wend"].observe(z0_comp_wend, mask=z_comp_wend_mask)
    # Observe z1_comp_wend values (if comp_wend is 2 or 3)
    z1_comp_wend = np.zeros((n_samples, 1), dtype=int)
    z1_comp_wend[data["comp_wend"].values.astype(int) == 3] = 1
    nodes["z1_comp_wend"].observe(z1_comp_wend, mask=z_comp_wend_mask)

    # Prime the clusters with the extreme values to make convergence faster
    obs_computer_use_classification = np.zeros((n_samples, 1), dtype=int)
    obs_computer_use_classification[(z1_comp_wend) & (z1_comp_week)] = 1
    obs_computer_use_mask = ((z_comp_wend_mask[:,0] & z_comp_week_mask[:,0]) & (((z1_comp_wend[:,0]) & (z1_comp_week[:,0])) | ((data["comp_week"].values.astype(int) == 0) & (data["comp_wend"].values.astype(int) == 0))))[:,None].astype(bool)
    nodes["obs_computer_use_classification"].observe(obs_computer_use_classification, mask=obs_computer_use_mask)

    # Fit model
    model.update(repeat=10000, verbose=True, tol=1e-7)


if __name__ == "__main__":
    path = 'maps-synthetic-data-v1.1.csv'
    data = load_data(path)
    data = transformation_001(data)
    data, results = specify_model(data)
    print(results)
