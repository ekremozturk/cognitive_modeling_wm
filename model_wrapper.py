#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:22:22 2019

@author: ekrem
"""
import time
import pickle

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
        self.optim_params = [m.params for m in self.models]
        
        
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
            capacity[ID] = self.data[self.data.participant_code == ID].iloc[0][feature]
        
        self.feature = feature
        self.capacity = capacity
        
    
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
    
    
    def gaussian_criterion(self):
        if self.capacity:
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
    
    def canonical_random_search(self, epochs):
        model = self.models[0]
        print('Random search is starting...')
        start = time.time()
        error, optim_param_set = model.fit_rnd(self.train_data.syllog.values, 
                                               self.train_data.syllog_conclusion.values, 
                                               num = epochs)
        end = time.time()
        print('Time taken to fit %f' %(end-start))
        
        self.models = [model]
        self.optim_params = [optim_param_set]
        self.search_scores = [1-error]
        
    def canonical_fit(self, epochs):
        model = self.models[0]
        print('Model fit is starting...')
        start = time.time()
        train_score, _ = model.fit(self.train_data.syllog.values, 
                                   self.train_data.syllog_conclusion.values, 
                                   num_fits = epochs)
        end = time.time()
        print('Time taken to fit %f' %(end-start))
        
        test_score = 1-model._fit_fun(self.optim_params[0], 
                                      self.test_data.syllog.values, 
                                      self.test_data.syllog_conclusion.values)
        
        self.models = [model]
        self.train_scores = [train_score]
        self.test_scores = [test_score]
        
    
    def clustered_fit(self, epochs):
        models = []
        optim_params = []
        train_scores = []
        test_scores = []
        for model, cluster in zip(self.models, self.clustered_data):
            print('Model fit is starting...')
            train_data, test_data = cluster
            start = time.time()
            train_score, optim_param_set = model.fit(train_data.syllog.values, 
                                                     train_data.syllog_conclusion.values, 
                                                     num_fits = epochs)
            end = time.time()
            print('Time taken to fit %f' %(end-start))
            
            test_score = 1-model._fit_fun(optim_param_set, 
                                          test_data.syllog.values, 
                                          test_data.syllog_conclusion.values)
            
            models.append(model)
            optim_params.append(optim_param_set)
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        self.models = models
        self.train_scores = train_scores
        self.test_scores = test_scores
            
# Read data
initial_data = pd.read_csv('../wm_syllog_data.csv', sep = ';', header = 0)

# TODO: Create and fit benchmark model to all data
w_benchmark = Wrapper(initial_data, num_sets = 1)
w_benchmark.canonical_train_test_split()
w_benchmark.canonical_random_search(epochs = 10)
w_benchmark.canonical_fit(epochs = 3)

with open('benchmark.model', 'w+') as f:
    pickle.dump(w_benchmark, f)


'''
# TODO: Create and fit models to uniform split data
w_uniform = Wrapper(initial_data, num_sets = 3)
w_uniform.set_capacity('os_max_span')
w_uniform.uniform_criterion()
w_uniform.clustered_train_test_split()
w_uniform.clustered_fit(epochs = 3)

# TODO: Create and fit models to gaussian split data
w_gaussian = Wrapper(initial_data, num_sets = 3)
w_gaussian.set_capacity('os_max_span')
w_gaussian.gaussian_criterion()
w_gaussian.clustered_train_test_split()
w_gaussian.clustered_fit(epochs = 3)
'''

'''
train_score = w_benchmark.train_scores[0]
test_score = w_benchmark.test_scores[0]

optim_params = w_benchmark.optim_params

train_scores = w_uniform.train_scores
test_scores = w_uniform.test_scores

optim_params = w_uniform.optim_params

train_scores = w_gaussian.train_scores
test_scores = w_gaussian.test_scores

optim_params = w_gaussian.optim_params
'''


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