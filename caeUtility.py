import numpy as np
import sklearn.metrics as skm

###############################################################################
# Takes in two lists of numpy arrays (representing fields) and calculates
# several loss metrics on the user specified level.
def computeFieldLossMetrics(truths, preds, level='field'):
    assert len(truths) > 0, 'truths must be a nonmepty list'
    assert len(preds) > 0, 'preds must be a nonmepty list'
    assert len(truths) == len(preds), 'truths and preds should be the same length'
    assert all([t.shape == p.shape for t, p in zip(truths, preds)]), 'the shape of all predicted fields must match the shape of their corresponding truth'
    assert level in ['point', 'point_agg', 'field', 'set'], 'level must be either \'point\', \'point_agg\', \'field\' or \'set\''
    
    metrics = {}
    
    # ---point-level metrics---
    errorList = [t-p for t, p in zip(truths, preds)]
    
    if level == 'point':
        metrics['errors'] = errorList
        metrics['relErrs'] = [e/t for e, t in zip(errorList, truths)]
        return metrics

    # ---point-aggregate-level metrics---
    if level == 'point_agg':
        assert all([preds[0].shape == p.shape for p in preds]), 'point_agg metrics require that all fields are the same shape'
        stackedErrors = np.stack(errorList)
        metrics['mse'] = np.mean(stackedErrors**2, axis=0)
        metrics['mae'] = np.mean(np.abs(stackedErrors), axis=0)
        metrics['r2'] = _pointAggR2(truths, preds)
        return metrics
        
    
    # ---field-level metrics--- 
    truePeakList = [np.max(np.abs(t)) for t in truths]
    predPeakList = [np.max(np.abs(p)) for p in preds]
    
    if level == 'field':
        metrics['mse'] = [np.mean(e**2) for e in errorList]
        metrics['mae'] = [np.mean(np.abs(e)) for e in errorList]
        metrics['maxAE'] = [np.max(np.abs(e)) for e in errorList]
        metrics['mae/peak'] = [mae/peak for mae, peak in zip(metrics['mae'], truePeakList)]
        metrics['maxAE/peak'] = [maxE/peak for maxE, peak in zip(metrics['maxAE'], truePeakList)]
        metrics['relEAtPeak'] = [np.abs(tp-pp)/tp for tp, pp in zip(truePeakList, predPeakList)]
        return metrics
    
    # ---set-level metrics---
    if level == 'set':
        concatErrors = np.concatenate(errorList)
        metrics['mse'] = np.mean(concatErrors**2)
        metrics['mae'] = np.mean(np.abs(concatErrors))
        metrics['peakR2'] = skm.r2_score(truePeakList, predPeakList)
        metrics['meanAggR2'] = np.mean(_pointAggR2(truths, preds))
        return metrics

###############################################################################
# Calculates loss metrics for a model that always predicts the mean.
# Can be used as a baseline for predictive models. truths is a list of numpy 
# arrays. level is the result level from computeFieldLossMetrics
def baseline(truths, level='field', avgLevel=None):
    assert len(truths) > 0, 'truths must be a nonmepty list'
    assert level in ['point', 'point_agg', 'field', 'set'], 'level must be either \'point\', \'point_agg\', \'field\' or \'set\''
    assert avgLevel in ['point', 'set', None], 'if specified, avgLevel must be either \'point\' or \'set\''

    uniformFields = all([truths[0].shape == t.shape for t in truths])

    if (avgLevel == 'point') or (not avgLevel and uniformFields):
        assert uniformFields, 'point-based averaging requires that all fields are the same shape'
        avgField = np.mean(truths, axis=0)
        preds = [avgField]*len(truths)
    else:
        avgVal = np.mean([np.mean(t) for t in truths])
        preds = [np.full(t.shape, avgVal) for t in truths]
    
    return computeFieldLossMetrics(truths, preds, level=level)


###############################################################################
# helper functions
def _pointAggR2(truths, preds):
    stackedPreds = np.stack(preds)
    stackedTruths = np.stack(truths)
    pointAggR2 = np.zeros(preds[0].shape)
    for ij in np.ndindex(preds[0].shape):
        pointAggR2[ij] = skm.r2_score(stackedTruths[(slice(None),)+ij], 
                                      stackedPreds[(slice(None),)+ij])
    return pointAggR2