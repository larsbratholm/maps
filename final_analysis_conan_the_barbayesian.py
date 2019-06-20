
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

def specify_model(data):
    """
    Create simple MLE model
    """
    import numpy as np
    import scipy.stats as ss

    # Keep only data points that we chose not to ignore
    sub_data = data[~data.n_ignore]

    # Create some masks of the data
    mask_depression = (sub_data.depression == True)
    mask_high_computer_use = (sub_data.comp_use == 'high')

    # get counts needed for MLE estimates
    count_comp_low_dep_no = sum(~mask_depression & ~mask_high_computer_use)
    count_comp_low_dep_yes = sum(mask_depression & ~mask_high_computer_use)
    count_comp_high_dep_no = sum(~mask_depression & mask_high_computer_use)
    count_comp_high_dep_yes = sum(mask_depression & mask_high_computer_use)

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
