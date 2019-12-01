#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:37:39 2019

@author: ekrem
"""

import time
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import phm.phm #whereever phm.py script is

class Wrapper():
    def __init__(self, data, num_sets, *args):
        
        
        def toFormat(conclusion):
            return 
        
        data.syllog_conclusion = data.syllog_conclusion.apply(toFormat)
        
        self.data = data
        self.participants_ID = np.unique(data.participant_code.values)
        self.num_sets = num_sets
        self.models = # phm model
        
    
    def uniform_criterion(self):
        if self.capacity:
            scores = list(self.capacity.values())
        else:
            print('Capacity dictionary is missing. Stopping!')
            return
        min_score, max_score = min(scores), max(scores)
        uniform_range = (max_score-min_score)/self.num_sets
        
        criterion = []
        for i in range(self.num_sets):
            min_cap = min_score+i*uniform_range
            max_cap = min_score+(i+1)*uniform_range
            criterion.append((min_cap, max_cap))
        
        self.criterion = criterion
        
        self.data.wm_capacity = self.data.apply(self.split, axis = 1)
    
    def split(self, participant):
        score = self.capacity[participant.participant_code]
        
        for i in range(self.num_sets):
            min_cap, max_cap = self.criterion[i]
            if score >= min_cap and score <= max_cap:
                return i
            
    def generate_predictions(self):
        # Iterate self.data in phm model to generate predictions

    def get_score(self):
        
        # Return percentage of successful guesses
        
    def get_cluster_score(self):
        
        # Return percentage of successful guesses for each cluster
        
initial_data = pd.read_csv('', sep = ';', header = 0)
w = Wrapper(initial_data, num_sets = 1)

# There is nothing to fit, there is only one model so once we 
# generate predictions we can get both overall score and score 
# with respect to each wm capacity.
