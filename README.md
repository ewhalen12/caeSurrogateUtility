# caeSurrogateUtility
Helper functions for those experimenting with CAE surrogate modeling in python.

### Requirements:
 - numpy
 - sklearn

### Example: 

```
import sys
import numpy as np
sys.path.append('{repo path}/caeSurrogateUtility')
import caeUtility as cu
```

In surrogate modeling, it's common to have a list of numpy arrays representing a set of fields. The fields could represent mechanical stress, fluid velocity, or electric potential for one design in a set. We often want to compare true fields to their corresponding predictions:
```
trueFields = [np.array([0.1, 0.2, 0.3]), 
              np.array([0.1, 0.2, 0.5]),
              np.array([1.7, 0.2, 0.6])]

predictions = [np.array([0.0, 0.2, 0.2]), 
               np.array([0.0, 0.2, 0.9]),
               np.array([1.6, 0.1, 0.6])]
```

There are many possible ways to quantify the quality of the prediction. Depending on the context, you may be interested in error metrics at the point, field or set levels.

#### Point-level metrics:
```
cu.computeFieldLossMetrics(trueFields, predictions, level='point')
```
```
{'errors': [array([0.1, 0. , 0.1]),
  array([ 0.1,  0. , -0.4]),
  array([0.1, 0.1, 0. ])],
 'relErrs': [array([1.        , 0.        , 0.33333333]),
  array([ 1. ,  0. , -0.8]),
  array([0.05882353, 0.5       , 0.        ])]}
```
#### Field-level metrics:
```
cu.computeFieldLossMetrics(trueFields, predictions, level='field')
```
```
{'mse': [0.006666666666666665, 0.05666666666666668, 0.006666666666666658],
 'mae': [0.06666666666666667, 0.16666666666666666, 0.06666666666666662],
 'maxAE': [0.1, 0.4, 0.1],
 'mae/peak': [0.22222222222222224, 0.3333333333333333, 0.03921568627450978],
 'maxAE/peak': [0.33333333333333337, 0.8, 0.05882352941176471],
 'relEAtPeak': [0.33333333333333326, 0.8, 0.05882352941176463]}
```
#### Set-level metrics:
```
cu.computeFieldLossMetrics(trueFields, predictions, level='set')
```
```
{'mae': 0.09999999999999999,
 'mse': 0.02333333333333333,
 'peakR2': 0.8430232558139534}
```

#### Point aggregate-level metrics:
For the special case when a correspondence exists between points across fields (e.g. if all fields have the same mesh), we can define aggregate metrics over each point as follows
```
cu.computeFieldLossMetrics(trueFields, predictions, level='point_agg')
```
```
{'mse': array([0.01      , 0.00333333, 0.05666667]),
 'mae': array([0.1       , 0.03333333, 0.16666667]),
 'r2': array([ 9.82421875e-01, -4.32691405e+30, -2.64285714e+00])}
```
