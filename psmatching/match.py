import timeit
import datetime
import sys

import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from functools import partial, reduce

from psmatching.utilities import *


####################################################
###################  Base Class  ###################
####################################################


class PSMatch(object):
    '''
    Parameters
    ----------
    file : string
        The file path of the data; assumed to be in .csv format.
    model : string
        The model specification for calculating propensity scores; in the format Y ~ X1 + X2 + ... + Xn
    k : string
        The number of controls to be matched to each treated case.
    gap : int
        The time gap between case's index date and control's index date.
    '''

    def __init__(self, path, model, k, gap):
        self.path = path
        self.model = model
        self.k = int(k)
        self.gap = int(gap)


    def prepare_data(self, **kwargs):
        '''
        Prepares the data for matching.

        Parameters
        ----------
        path : string
            The file path of the data that need to apply PSM. Assumed to be in .csv format.

        Returns
        -------
        A Pandas DataFrame containing raw data plus a column of propensity scores.
        '''
        # Read in the data specified by the file path
        df = pd.read_csv(self.path)
        # Obtain propensity scores and add them to the data
        print("\nCalculating propensity scores ...", end = " ")
        propensity_scores = get_propensity_scores(model = self.model, data = df, verbose = False)
        print("Preparing data ...", end = " ")
        df["PSCORE"] = propensity_scores
        # Assign the df attribute to the Match object
        self.df = df
        print("DONE!")

    def set_caliper(self, caliper_scale, caliper):
        '''
        The caliper for matching.
        A caliper of 0 means no caliper is used.

        Parameters
        ----------
        caliper_scale : 'standardized' or 'logit'
        caliper : A caliper is the distance which is acceptable for any match. 
        Returns
        -------
        propensity score calipers
        '''
        propensity = self.df.PSCORE
        if caliper_scale == None:
            caliper = 0
        if not(0<=caliper<1):
            if caliper_scale == "propensity" and caliper>1:
                raise ValueError('Caliper for "propensity" method must be between 0 and 1')
            elif caliper<0:
                raise ValueError('Caliper cannot be negative')

        # Transform the propensity scores and caliper when caliper_scale is "logit" or None
        if caliper_scale == "logit":
            propensity = np.log(propensity/(1-propensity))
            caliper = caliper*np.std(propensity)
        
        print('caliper has been set to:', caliper)
        return caliper

    def match_by_neighbor(self, caliper):
        '''
        Performs propensity score matching.

        Parameters
        ----------
        caliper : the attribute returned by the set_caliper() function
        Returns
        -------
        matched_controls : Pandas DataFrame
        unmatched: Set
        under_matched: Set
        '''
        ignore_list = set()
        under_matched = set()
        unmatched = set()
        matched_controls = pd.DataFrame()

        ratio = self.k
        data = self.df

        # convert data type
        data.INDEX_DATE = pd.to_datetime(data.INDEX_DATE)
        data.CASE = data.CASE.astype(int)

        controls = data[data.CASE == 0]
        cases = data[data.CASE == 1]
        
        neigh = NearestNeighbors(radius=caliper, algorithm='ball_tree', n_jobs=1)
        neigh.fit(controls[['PSCORE']])

        # calculate time
        i = 1
        total_cases = cases.shape[0]
        start = timeit.default_timer()
        
        #loop through each case
        for index, case in cases.iterrows():
            
            # case index date
            case_indexdate = cases[cases.PATID == case['PATID']].INDEX_DATE.values[0]
            
            # current case's pscore
            pscore = case.PSCORE
            
            # find all controls with pscore within the caliper distance
            distances, indices = neigh.radius_neighbors([[pscore]])
            
            sample = controls.iloc[indices[0]]
            
            # pick out those that have NOT been used
            sample = sample[~sample['PATID'].isin(ignore_list)].copy()
            
            ## verify index date for control
            sample['INDEX_DATE_GAP'] = abs(sample.INDEX_DATE - case_indexdate) / np.timedelta64(1, 'D')
            sample = sample[sample.INDEX_DATE_GAP <= self.gap].sort_values(by=['PATID', 'INDEX_DATE_GAP'])
            
            # rank the samples by their distances to the case's pscore
            sample['DIST'] = abs(sample['PSCORE']-pscore)
            sample.sort_values(by='DIST', ascending=True, inplace=True)
            
            # picked the closest "ratio"
            sample = sample.head(ratio).copy().reset_index(drop=True)
            
            if (sample.shape[0] < ratio and sample.shape[0] != 0):
                under_matched.add(case['PATID'])
            if (sample.shape[0] == 0):
                unmatched.add(case['PATID'])
                
            # exclude the selected sample from the matching pool (i.e., without replacement)
            ignore_list.update(sample['PATID'])
            
            sample['MATCHED_CASE'] = case['PATID']
            sample['MATCHED_CASE_INDEX_DATE'] = case_indexdate
            
            matched_controls = matched_controls.append(sample, ignore_index=True)
            
            # track progress
            stop = timeit.default_timer()
            
            print("Current progress:", np.round(i/total_cases * 100, 2), "%")
            print("Current run time:", np.round((stop - start) / 60, 2), "min")
            
            i = i+1

        matched_controls = matched_controls.reset_index(drop=True)
        cases = find_case(data, unmatched)

        self.matched_controls = matched_controls
        self.unmatched = unmatched

        write_matched_data(self.path, cases, matched_controls)

        return under_matched, unmatched, matched_controls


    def run(self, **kwargs):
        self.prepare_data()
        caliper = self.set_caliper('logit', caliper=0.01)
        self.match_by_neighbor(caliper)



























































