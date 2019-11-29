#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:22:22 2019

@author: ekrem
"""
import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import mreasoner

class Wrapper():
    
    def __init__(self, data, num_sets, *args):
        
        def toFormat(conclusion):
            return 'NVC' if conclusion == 'nvc' else conclusion[0].upper()+conclusion[1:]
        
        def launch_mreasoner_model():
            cloz = mreasoner.ClozureCL()
            mreas_path = mreasoner.source_path()
            
            return mreasoner.MReasoner(cloz.exec_path(), mreas_path)
        
        data.syllog_conclusion = data.syllog_conclusion.apply(toFormat)
        
        self.data = data
        self.participants_ID = np.unique(data.participant_code.values)
        self.num_sets = num_sets
        self.models = [launch_mreasoner_model() for i in range(num_sets)]
        
        
    def canonical_train_test_split(self):
        train_participants_ID, test_participants_ID = train_test_split(self.participants_ID)
        
        self.train_data = self.data[self.data.participant_code.isin(train_participants_ID)].copy()
        self.test_data  = self.data[self.data.participant_code.isin(test_participants_ID)].copy()


    def clustered_train_test_split(self):
        clustered_data = []
        for i in range(self.num_sets):
            sub_data = self.data[self.data.wm_capacity == i]
            sub_participants_ID = np.unique(sub_data.participant_code.values)
            train_participants_ID, test_participants_ID = train_test_split(sub_participants_ID)
            sub_train_data = sub_data[sub_data.participant_code.isin(train_participants_ID)].copy()
            sub_test_data  = sub_data[sub_data.participant_code.isin(test_participants_ID)].copy()
            clustered_data.append((sub_train_data, sub_test_data))
        
        self.clustered_data = clustered_data
    
    
    def set_capacity(self, feature):
        capacity = dict()
        
        for ID in self.participants_ID:
            capacity[ID] = self.data[self.data.participant_code == ID].feature
        
        self.feature = feature
        self.capacity = capacity
        
    
    def uniform_criterion(self):
        if self.criterion:
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
        
        self.data.wm_capacity = self.data.apply(self.uniform_split, axis = 1)
    
    
    def gaussian_criterion(self):
        if self.criterion:
            scores = list(self.capacity.values())
        else:
            print('Capacity dictionary is missing. Stopping!')
            return
        
        min_score, max_score = min(scores), max(scores)
        mean_score = np.mean(scores)
        std_dev = np.sqrt(np.var(scores))
        
        criterion = [(min_score, mean_score-std_dev),
                     (mean_score-std_dev, mean_score+std_dev),
                     (mean_score+std_dev, max_score)]
        
        self.criterion = criterion
        
        self.data.wm_capacity = self.data.apply(self.split, axis = 1)
    
    
    def split(self, participant):
        score = self.capacity[participant.participant_code]
        
        for i in range(self.num_sets):
            min_cap, max_cap = self.criterion[i]
            if score >= min_cap and score <= max_cap:
                return i
    
    
    def canonical_fit(self, epochs):
        model = self.model[0]
        print('Model fit is starting...')
        start = time.time()
        train_score, optim_param_set = model.fit(self.train_data.syllog.values, 
                                                 self.train_data.syllog_conclusion.values, 
                                                 num_fits = epochs)
        end = time.time()
        print('Time taken to fit %f' %(end-start))
        
        test_score = 1-model._fit_fun(optim_param_set, 
                                      self.test_data.syllog.values, 
                                      self.test_data.syllog_conclusion.values)
        
        self.models = [model]
        self.optim_params = [optim_param_set]
        self.train_scores = [train_score]
        self.test_scores = [test_score]
        
    
    def clustered_fit(self, epochs):
        models = []
        optim_params = []
        train_scores = []
        test_scores = []
        for model, cluster in zip(self.model, self.clustered_data):
            print('Model fit is starting...')
            train_data, test_data = cluster
            start = time.time()
            train_score, optim_param_set = model.fit(self.train_data.syllog.values, 
                                                     self.train_data.syllog_conclusion.values, 
                                                     num_fits = epochs)
            end = time.time()
            print('Time taken to fit %f' %(end-start))
            
            test_score = 1-model._fit_fun(optim_param_set, 
                                          self.test_data.syllog.values, 
                                          self.test_data.syllog_conclusion.values)
            
            models.append(model)
            optim_params.append(optim_param_set)
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        self.models = models
        self.optim_params = optim_params
        self.train_scores = train_scores
        self.test_scores = test_scores
            
# Read data
initial_data = pd.read_csv('../wm_syllog_data.csv', sep = ';', header = 0)

# Create and fit canonical model
w_benchmark = Wrapper(initial_data, 1)
w_benchmark.canonical_train_test_split()
w_benchmark.canonical_fit(epochs = 1)

train_score = w_benchmark.train_scores[0]
test_score = w_benchmark.test_scores[0]

'''
preds = []
for idx, trial in test_data.iterrows():
    res = wrapper.model.query(trial.syllog)
    preds.append(res)
    
individual_scores = []
for pred, truth in zip(preds, test_data.syllog_conclusion.values):
    individual_scores.append(1 if truth in pred else 0)
    
my_test = sum(individual_scores)/len(individual_scores)
'''

# TODO: Create and fit models to uniform split data

# TODO: Create and fit models to gaussian split data

