import numpy as np
from scipy.stats import bootstrap

def interquartile_mean(*data, axis=None):
    """
    Note: data will always be tuple, right?
    """
    assert isinstance(data, tuple)
    if len(data) == 1:
        data = data[0]
    if axis is None:
        # Flatten the data if no axis is specified
        if isinstance(data, tuple):
            assert len(data) == 1
            data = data[0]
            
        return interquartile_mean(data.flatten(), axis=0)
    
    def compute_iq_mean(arr):
        arr = np.sort(arr)
        low_incl = (len(arr)-1)//4+1
        high_incl = (3*len(arr))//4
        middle_arr = arr[low_incl:high_incl]
        no_of_wanted = len(arr) / 2  # Number of wanted seeds for the calculation of the mean. 50% of the input length (fractional).
        outer_coef = (no_of_wanted - len(middle_arr))/2
        total_sum = middle_arr.sum()
        total_sum += outer_coef * (arr[low_incl-1] + arr[high_incl])
        return total_sum / no_of_wanted

    # Apply along the specified axis
    def print_return(foo, print=True):
        if print: print(foo)
        return foo
    return np.array(list(map(lambda data: interquartile_mean(data, axis=axis), data))) if isinstance(data, tuple) else print_return(np.apply_along_axis(compute_iq_mean, axis, data), print=False)


def interquartile_confidence_interval(data, axis=0, confidence_level=0.8, method="BCa"):
    if len(data.shape) == 1:
        data = data.reshape((len(data), 1))
    result = bootstrap(
        data = data.T,
        statistic = interquartile_mean,
        n_resamples = 1000,
        axis = axis, 
        confidence_level = confidence_level,
        alternative = "two-sided",
        method = method,
        # vectorized = False,
    )
    return result.confidence_interval