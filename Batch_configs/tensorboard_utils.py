from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes
from scipy.stats import bootstrap

# import pandas as pd

# tfevent = EventAccumulator(
#     ".../events.out.tfevents..."
# )

# print(tfevent.Reload())
# print(tfevent.Tags())
# # print(acc.Scalars("train/loss"))
# print(tfevent.Scalars("val/Weighted_Multiclass_F1_Score"))

def extract_scalar(tfevent, name):
    """
    Parameters: 
        tfevent: EventAccumulator object of the tfevents file.
        name: Name of scalar
    Returns: Iterations(x), ScalarValue(y)
    """
    x, y = [], []
    tfevent.Reload()
    for scalar_event in tfevent.Scalars(name):
        x.append(scalar_event.step)
        y.append(scalar_event.value)
    return x, y

# def interquartile_mean(data, axis=None):
#     if axis is None:
#         # Flatten the data if no axis is specified
#         return interquartile_mean(data.flatten(), axis=0)
    
#     def compute_iq_mean(arr):
#         q1, q3 = np.percentile(arr, [25, 75])
#         interquartile_data = arr[(arr >= q1) & (arr <= q3)]
#         return np.mean(interquartile_data)

#     # Apply along the specified axis
#     return np.apply_along_axis(compute_iq_mean, axis, data)

def interquartile_mean(*data, axis=None):
    """
    Note: data will always be tuple, right?
    """
    # print(f"data from top:  {data}")
    assert isinstance(data, tuple)
    if len(data) == 1:
        data = data[0]
    # print(len(data), axis)
    # print(type(data), isinstance(data, tuple))
    # if isinstance(data, np.ndarray): print(f"Array.shape: {data.shape}")
    # print(f"axis: {axis}")
    # print(len(data))
    # for elem in data:
    #     print(elem.shape)
    if axis is None:
        # Flatten the data if no axis is specified
        if isinstance(data, tuple):
            assert len(data) == 1
            data = data[0]
            # try:
            #     data = np.array(data)
            # except:
            #     print(len(data))
            #     for elem in data:
            #         print(len(elem))
            # print(data)
            # print(type(data))
            # print(data.shape)
            # exit(1)
        # print("print_from_within: ", data, type(data))
        return interquartile_mean(data.flatten(), axis=0)
        # try:
        # except:
        #     exit(1)
    
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
    # print("applying...")
    # print(data)
    # print(np.array(data).shape)
    # return np.apply_along_axis(interquartile_mean, 0, np.array(data)) if isinstance(data, tuple) else np.apply_along_axis(compute_iq_mean, axis, data[0])
    def print_return(foo, print=True):
        if print: print(foo)
        return foo
    return np.array(list(map(lambda data: interquartile_mean(data, axis=axis), data))) if isinstance(data, tuple) else print_return(np.apply_along_axis(compute_iq_mean, axis, data), print=False)


def interquartile_std(data, axis=None):
    if axis is None:
        # Flatten the data if no axis is specified
        return interquartile_std(data.flatten(), axis=0)
    
    def compute_iq_std(arr):
        q1, q3 = np.percentile(arr, [25, 75])
        interquartile_data = arr[(arr >= q1) & (arr <= q3)]
        return np.std(interquartile_data)

    # Apply along the specified axis
    return np.apply_along_axis(compute_iq_std, axis, data)

def interquartile_confidence_interval(data, axis=None, confidence_level=0.8, method="BCa"):
    # print(f"data.shape from i_c_i: {data.shape}")
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
    # result = np.apply_along_axis(
    #     lambda a: print(a) or bootstrap(
    #         data = a,
    #         statistic = interquartile_mean,
    #         n_resamples = 1000,
    #         confidence_level = confidence_level,
    #         alternative = "two-sided",
    #         method = method,
    #     ),
    #     axis=axis, 
    #     arr=data,
    #     )
    # print(f"result.confidence_interval: {result.confidence_interval}")
    return result.confidence_interval

def extract_scalar_std_dev(tfevents:list, name):
    """
    Parameters:
        tfevents: List of EventAccumulator objects with different seeds
        name: Name of scalar
    Returns: Iterations(x), MeanScalarValue(y), StdDevScalarValue(y_err)
    """
    iterations = []
    values = []
    for tfevent in tfevents:
        if not isinstance(tfevent, EventAccumulator): tfevent = EventAccumulator(tfevent.as_posix())
        x, y = extract_scalar(tfevent, name)
        iterations.append(x)
        values.append(y)
    min_iterations = min(map(len, iterations))
    max_iterations = max(map(len, iterations))
    if min_iterations < max_iterations:
        print(f"Reducing the samples from up to {max_iterations} down to {min_iterations}")
        iterations = list(map(lambda a: a[:min_iterations], iterations))
        values = list(map(lambda a: a[:min_iterations], values))
    for i in range(len(iterations)-1):
        assert iterations[i] == iterations[i+1], f"{iterations[i]} & {iterations[i+1]} respectively"
    values = np.array(values)
    # mean_values = interquartile_mean(values, axis=0)
    mean_values = values.mean(axis=0)
    # std_values = interquartile_std(values, axis=0)
    std_values = values.std(axis=0)

    return iterations[0], mean_values, std_values

def extract_scalar_iqm_conf(tfevents:list, name, confidence_level=0.8, bootstrap_method="BCa"):
    """
    Parameters:
        tfevents: List of EventAccumulator objects with different seeds
        name: Name of scalar
    Returns: Iterations(x), IQMeanScalarValue(y), BootstrapConfidenceScalarValue(y_low, y_high)
    """
    iterations = []
    values = []
    for tfevent in tfevents:
        if not isinstance(tfevent, EventAccumulator): tfevent = EventAccumulator(tfevent.as_posix())
        try:
            x, y = extract_scalar(tfevent, name)
        except Exception as e:
            print(e)
            x, y = [], []
        iterations.append(x)
        values.append(y)
    min_iterations = min(map(len, iterations))
    max_iterations = max(map(len, iterations))
    if min_iterations < max_iterations:
        print(f"Reducing the samples from up to {max_iterations} down to {min_iterations}")
        print(f"len(iterations): {list(map(len, iterations))}")
        iterations = list(map(lambda a: a[:min_iterations], iterations))
        values = list(map(lambda a: a[:min_iterations], values))
    for i in range(len(iterations)-1):
        assert iterations[i] == iterations[i+1], f"{iterations[i]} & {iterations[i+1]} respectively"
        assert len(values[i]) == len(values[i+1]), f"{len(values[i])} & {len(values[i+1])} respectively"
    values = np.array(values)
    # print(f"values.shape: {values.shape}")
    mean_values = interquartile_mean(values, axis=0)
    # print("mean_values.shape, mean_values: ", mean_values.shape, mean_values)
    # print(f"type(mean_values): {type(mean_values)}")
    conf_values = interquartile_confidence_interval(values, axis=0, confidence_level=confidence_level, method=bootstrap_method)
    # print(f"conf_values: {conf_values}")

    return iterations[0], mean_values, conf_values

def locate_tfevent(version_dir):
    """
    Locate the tfevents file from within the version directory
    Parameters:
        version_dir: Path to the version directory containing a tfevents file.
    Returns:
        The path to the tvevent's file contained within the version directory
    """
    version_dir = Path(version_dir)
    candidate_files = [file for file in version_dir.iterdir() if file.is_file() and str(file.name).startswith("events.out.tfevents.")]
    if len(candidate_files) > 1:
        print(f"Number of candidate files found exceed 1. These candidate files were found: {candidate_files}, taking the first one..")
    assert len(candidate_files) > 0, "No candidate file found, check the logging directory"
    return candidate_files[0]

def locate_and_open_tfevent(version_dir):
    """
    Locate the tfevents file from within the version directory and returns an EventAccumulator ojbect of said file.
    Parameters:
        version_dir: Path to the version directory containing a tfevents file.
    Returns:
        An EventAccumulator object of the tvevent's file contained within the version directory
    """
    tfevent_path = locate_tfevent(version_dir=version_dir)
    return EventAccumulator(tfevent_path.as_posix())

def plot_with_errors_to_axis(axis: Axes, x, y, y_err, colour=None, alpha=0.6, label=None, linestyle="-", white_region_below=False):
    if isinstance(y_err, tuple):  y_low, y_high = y_err
    else:  y_low, y_high = y-y_err, y+y_err
    axis.plot(x, y, c=colour, label=label, linestyle=linestyle)
    if white_region_below: axis.fill_between(x, y_low, y_high, facecolor="white", zorder=0.75)
    axis.fill_between(x, y_low, y_high, facecolor=colour, alpha=alpha)
    return


# wmf1x, wmf1y = extract_scalar(tfevent, "val/Weighted_Multiclass_F1_Score")
# print(wmf1x, wmf1y)
# wmf1x, wmf1y, _ = extract_scalar_std_dev([tfevent, tfevent], "val/Weighted_Multiclass_F1_Score")
# print(wmf1x, wmf1y)
# print(tfevent.scalars.Keys())