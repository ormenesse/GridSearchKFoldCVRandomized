__author__ = "Vinicius Ormenesse"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Vinicius Ormenesse"
__email__ = "vinicius.ormenesse@gmail.com"
__status__ = "Done"

from itertools import product
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics

class grid_search_regressor_kfoldcv():

    def __init__(self,model,params,error='r2'):
        """
            model : your initialized model.
            params : your params in dict. 
                    Example: 
                    params = {
                    'n_estimators' : [15,30,50,100],
                    'num_leaves' : [2,5,10],
                    'max_depth' : [2,3,5],
                    'learning_rate' : [0.1,0.01,0.5]
                    }
        """
        self.model = model
        self.params = params
        self.error = error
        self.results = []

    def kfoldcv(self,indices, k = 5, seed = 4242):
        """
            k = Number of folds in your cross validation.
        """
        size = len(indices)
        
        subset_size = round(size / k)
        
        random.Random(seed).shuffle(indices)
        
        subsets = [indices[x:x+subset_size] for x in range(0, len(indices), subset_size)]
        
        test = []
        
        train = []
        
        for i in range(k):
        
            test.append(subsets[i])
            
            trainz = np.array([])
            
            for j,subset in enumerate(subsets):
            
                if i != j:
                
                    trainz = np.concatenate((trainz,subset),axis=0)
                    
            train.append(list(trainz))
            
        return train,test

    def fit(self,X,Y,cv=10,seed=42,max_random_iter=-1):
        """
            cv = number of folds.
            max_random_iter = -1 if no randomized search, else the maximum iterations on your randomized search.
        """
        kfold_ensembler = self.kfoldcv(list(X.index),k=cv)
        
        _results = []
        
        if max_random_iter == -1:
        
            model_params_list = [dict(zip(self.params, v)) for v in product(*self.params.values())]
        else:
        
            model_params_list = random.choices([dict(zip(self.params, v)) for v in product(*self.params.values())],k=max_random_iter )
        
        for model_params in tqdm(model_params_list):
            
            self.model.set_params(**model_params)
            
            Rscore_train = 0
            
            Rscore_test = 0
            
            delta = 0
            
            for i in range(cv):
                
                train = kfold_ensembler[0][i]
                
                test = kfold_ensembler[1][i]

                self.model.fit(X.iloc[train], Y.iloc[train])
                
                if self.error == 'rmsle':
                    
                    r2_t = metrics.mean_squared_log_error(Y.iloc[train].abs(), self.model.predict(X.iloc[train].abs()))
                    
                    r2_tt = metrics.mean_squared_log_error(Y.iloc[test].abs(), self.model.predict(X.iloc[test].abs()))
                    
                elif self.error == 'rmse':
                    
                    r2_t = metrics.mean_squared_error(Y.iloc[train], self.model.predict(X.iloc[train]))
                    
                    r2_tt = metrics.mean_squared_error(Y.iloc[test], self.model.predict(X.iloc[test]))
                    
                else:

                    r2_t = metrics.r2_score(Y.iloc[train], self.model.predict(X.iloc[train]))

                    r2_tt = metrics.r2_score(Y.iloc[test], self.model.predict(X.iloc[test]))
                
                
                
                dt = np.abs(r2_t-r2_tt)
                
                Rscore_train = (Rscore_train*(i+1-1)+r2_t)/(i+1)
                
                Rscore_test = (Rscore_test*(i+1-1)+r2_tt)/(i+1)
                
                delta = (delta*(i+1-1)+dt)/(i+1)

            _results.append({**model_params,**{'Train':Rscore_train, 'Test':Rscore_test, 'Delta':delta}})
        
        self.results = _results
    
    def display_results(self,on='DeltaTest',ascending=False):
        
        """
            Choose 'on' between ['Train','Test','Delta', 'DeltaTrain', 'DeltaTest'].
        """
        
        assert on in ['Train','Test','Delta', 'DeltaTrain', 'DeltaTest'], "On option not in ['Train','Test','Delta', 'DeltaTrain', 'DeltaTest']" 
        
        if on == 'Delta':
        
            return pd.DataFrame(self.results).sort_values('Delta',ascending=not ascending)
            
        elif on == 'Test':
        
            return pd.DataFrame(self.results).sort_values('Test',ascending=ascending)
            
        elif on == 'Train':
        
            return pd.DataFrame(self.results).sort_values('Train',ascending=ascending)
            
        elif on == 'DeltaTrain':
        
            return pd.DataFrame(self.results).sort_values(['Delta','Train'],ascending=[not ascending,ascending])
            
        elif on == 'DeltaTest':
        
            return pd.DataFrame(self.results).sort_values(['Delta','Test'],ascending=[not ascending,ascending])
