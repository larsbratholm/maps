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
