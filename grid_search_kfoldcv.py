__author__ = "Vinicius Ormenesse"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Vinicius Ormenesse"
__email__ = "vinicius.ormenesse@gmail.com"
__status__ = "Done"

from itertools import product
from sklearn.metrics import roc_auc_score
import random
import pandas as pd
from tqdm import tqdm
import numpy as np

class grid_search_kfoldcv():

    def __init__(self,model,params):
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
        
            model_params_list = random.choices([dict(zip(params, v)) for v in product(*params.values())],k=max_random_iter )
        
        for model_params in tqdm(model_params_list):
            
            self.model.set_params(**model_params)
            
            roc_train = 0
            
            roc_test = 0
            
            delta = 0
            
            for i in range(cv):
                
                train = kfold_ensembler[0][i]
                
                test = kfold_ensembler[1][i]

                self.model.fit(X.iloc[train], Y.iloc[train])

                roc_t = roc_auc_score(Y.iloc[train], self.model.predict_proba(X.iloc[train])[:, 1])

                roc_tt = roc_auc_score(Y.iloc[test], self.model.predict_proba(X.iloc[test])[:, 1])

                dt = np.abs(roc_t-roc_tt)
                
                roc_train = (roc_train*(i+1-1)+roc_t)/(i+1)
                
                roc_test = (roc_test*(i+1-1)+roc_tt)/(i+1)
                
                delta = (delta*(i+1-1)+dt)/(i+1)

            _results.append({**model_params,**{'Train':roc_train, 'Test':roc_test, 'Delta':delta}})
        
        self.results = _results
    
    def display_results(self,on='DeltaTest'):
        
        """
            Choose 'on' between ['Train','Test','Delta', 'DeltaTrain', 'DeltaTest'].
        """
        
        assert on in ['Train','Test','Delta', 'DeltaTrain', 'DeltaTest'], "On option not in ['Train','Test','Delta', 'DeltaTrain', 'DeltaTest']" 
        
        if on == 'Delta':
        
            return pd.DataFrame(self.results).sort_values('Delta',ascending=True)
            
        elif on == 'Test':
        
            return pd.DataFrame(self.results).sort_values('Test',ascending=False)
            
        elif on == 'Train':
        
            return pd.DataFrame(self.results).sort_values('Train',ascending=False)
            
        elif on == 'DeltaTrain':
        
            return pd.DataFrame(self.results).sort_values(['Delta','Train'],ascending=[True,False])
            
        elif on == 'DeltaTest':
        
            return pd.DataFrame(self.results).sort_values(['Delta','Test'],ascending=[True,False])
        
