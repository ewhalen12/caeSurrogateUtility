import numpy as np
import sklearn.metrics as skm
import torch

########################################################################
# takes in two lists of numpy arrays (representing fields) and calculates
# several loss metrics
def computeLossMetrics(truths, preds, level='part'):
    assert len(truths) > 0, 'truths must be a nonmepty list'
    assert len(preds) > 0, 'preds must be a nonmepty list'
    assert len(truths) == len(preds), 'truths and preds should be the same length'
    
    if type(truths[0]) == torch.Tensor:
        truths = [t.numpy() for t in truths]
        
    if type(preds[0]) == torch.Tensor:
        preds = [p.numpy() for p in preds]
    
    metrics = {}
    
    # ---point-level metrics---
    errorList = [t-p for t, p in zip(truths, preds)]
    
    if level == 'point':
        metrics['errors'] = errorList
        return metrics
    
    # ---part-level metrics--- 
    truePeakList = [np.max(np.abs(t)) for t in truths]
    predPeakList = [np.max(np.abs(p)) for p in preds]
    
    if level == 'part':
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
        metrics['mae'] = np.mean(np.abs(concatErrors))
        metrics['mse'] = np.mean(concatErrors**2)
        metrics['peakR2'] = skm.r2_score(truePeakList, predPeakList)
        
        return metrics
    