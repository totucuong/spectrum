import statsmodels.stats.api as sms


def accuracy(truths, discovered_truths, criteria='exact', epsilon=1000):
    """compute accuracy measure given ground truths and discovered truths.
    
    For now we assume there is only one single truth value for each object.
    
    Parameters
    ----------
    truths: pandas.DatFrame
        a data frame of two columns (object_id, value).
    
    discovered_truths: np.ndarray
        a data frame of two columns (object_id, value).
    
    criteria: string
        a criteria to match up ground truth and discoverd truths. If set to `exact` then a value
        is considered correct if equal to its corresponding ground truth. That is
        truths[object_id] = discovered_truths[object_id].
        
    Returns
    ------
    accuracy: float
    """
    t_df = truths.set_index('object_id').sort_index()
    dt_df = discovered_truths.set_index('object_id').sort_index()

    if t_df.shape[0] != discovered_truths.shape[0]:
        raise ValueError(
            f'The number of truths {t_df.shape[0]} != the number of discoverd truths {discovered_truths.shape[0]}'
        )

    accuracy = (t_df.value == dt_df.value).sum() / t_df.shape[0]
    return accuracy


def confidence_interval_of_accuracy_mean(accuracy):
    """compute confidence interval for a list of accuracy.
    
    Parameters
    ----------
    accuracy: 1D array like
        an array like of accurcy measure
    """
    desc = sms.DescrStatsW(accuracy)
    mean = desc.mean
    lower, _ = desc.tconfint_mean()
    lef_side = mean - lower
    return f'mean: {desc.mean} ± {lef_side}'
