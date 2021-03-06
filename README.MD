# K Fold Grid Randomized Search                                                      
                                                                                     

An alternative to sklearn GridSearchCV. At the end of the grid search, you are capable to see and choose which parameters are better for your model and purposes.

Simple to use:
```python
import lightgbm as lgb
model = lgb.LGBMClassifier()
params = {'n_estimators' : [15,30,50,100],
          'num_leaves' : [2,5,10],
          'max_depth' : [2,3,5],
          'learning_rate' : [0.1,0.01,0.5]}
```
Initialize the class:
```python
gds = grid_search_kfoldcv(model,params)
```
Fit the desired searches:
```python
gds.fit(X,Y,cv=10)
```
Display the best results in front considering ROC_AUC Delta between train and test (all aternatives are: 'Train','Test','Delta', 'DeltaTrain', 'DeltaTest'])

```python
gds.display_results(on='DeltaTest')
```

|Index    |Delta	|Test	|Train	|learning_rate|max_depth|n_estimators|num_leaves|
| ------- |:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|13	|0.023513	|0.855080	|0.858969	|0.01	|3	|15	|5        |
|67	|0.028224	|0.872607	|0.869483	|0.01	|3	|50	|5        |
|91	|0.029859	|0.859334	|0.865093	|0.01	|2	|100      |5        |
|100|0.029859	|0.859334	|0.865093	|0.01	|2	|100      |10       |
|70	|0.030082	|0.870750	|0.869484	|0.01	|5	|50	|5        |
|27	|0.030166	|0.859007	|0.858587	|0.10	|2	|30	|2        |
