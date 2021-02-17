import pandas as pd
import numpy as np


##################################################################
###################  Utility/Helper Functions  ###################
##################################################################


def get_propensity_scores(model, data, verbose = False):
    '''
    Utilizes a logistic regression framework to calculate propensity scores
    based on a specified model.

    Parameters
    ----------
    model : string
        a model specification in the form Y ~ X1 + X2 + ... + Xn
    data : Pandas DataFrame
        the data used to calculate propensity scores
    verbose : boolean
        verbosity of the model output

    Returns
    -------
    An array of propensity scores.
    '''
    import statsmodels.api as sm
    glm_binom = sm.formula.glm(formula = model, data = data, family = sm.families.Binomial())
    result = glm_binom.fit()
    if verbose:
        print(result.summary)
    return result.fittedvalues

def find_case(data, unmatched):
    '''
    remove unmatched cases.

    Parameters
    ----------
    data : Pandas DataFrame
    unmatched : Set
    Returns
    -------
    A Pandas DataFrame of cases.
    '''
    cases = data[data.CASE == 1].copy().reset_index(drop=True)
    cases = cases[~cases.PATID.isin(unmatched)].reset_index(drop=True)
    return cases

def write_matched_data(path, cases, controls):
    '''
    Writes Cases and Controls data to csv file.

    Parameters
    ----------
    path : string
        a file path used to derive the saved file path
    cases : Pandas Dataframe
    controls : Pandas Dataframe
        the dataframe to be written to file.
    '''
    print("Writing matched data to file ...", end = " ")
    save_file_case = "matched_cases.csv"
    save_file_control = "matched_controls.csv"
    cases.to_csv(save_file_case, index = False)
    controls.to_csv(save_file_control, index = False)
    print("DONE!")