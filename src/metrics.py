import numpy as np

def mean_squared_error(estimates, targets):
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is 
    estimated. The formula is:

    MSE = (1 / n) * \sum_{i=1}^{n} (Y_i - Yhat_i)^2

    Implement this formula here, using numpy and return the computed MSE

    https://en.wikipedia.org/wiki/Mean_squared_error
    """
    n = len(targets)
    diff_array = np.array([])
    for i in range(0, n):
        diff_array = np.append(diff_array, (targets[i]-estimates[i]) ** 2)
    summed = np.sum(diff_array)
    MSE = (1/n) * summed
    return MSE