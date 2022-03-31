# Import Packages
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

import pandas as pd
import numpy as np

# Gather Data
boston_dataset = load_boston()
data = pd.DataFrame(data = boston_dataset.data, columns = boston_dataset.feature_names)

features = data.drop(columns = ['INDUS', 'AGE'])

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns = ['PRICE'])

ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

property_stats = features.mean().values.reshape(1, 11)



regr = LinearRegression().fit(features, target)

fitted_vals = regr.predict(features)
MSE = mse(target, fitted_vals)
RMSE = np.sqrt(MSE)


def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river = False,
                    high_confidence = True):
    
    # Configure Property
    _property_stats = property_stats.copy()
    _property_stats[0][RM_IDX] = nr_rooms
    _property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river:
        _property_stats[0][CHAS_IDX] = 1
    else:
        _property_stats[0][CHAS_IDX] = 0
    
    # Make Prediction
    log_estimate = regr.predict(_property_stats)[0][0]
    
    # Calculate Range
    if high_confidence:
        upper_bound = log_estimate + 2 * RMSE
        lower_bound = log_estimate - 2 * RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    
    return log_estimate, upper_bound, lower_bound, interval


def get_dollar_estimate(rm, ptratio, chas = False, conf_range = True):
    """Estimate the price of a property in Boston.
    
    Keyword arguments:
    rm -- number of rooms in the property
    ptratio -- number of students per teacher in the classroom for the school in the area
    chas -- True if the property is next to the river, False if otherwise
    conf_range -- True for a 95% prediction interval, False for a 68 interval
    
    """
    
    if rm < 1 or ptratio < 1 or ptratio > 100:
        print('That is unrealistic. Try again.')
        return
    
    log_est, log_hi, log_low, conf = get_log_estimate(rm,
                                                      students_per_classroom = ptratio,
                                                      next_to_river = chas,
                                                      high_confidence = conf_range)
    
    dollar_est = SCALE_FACTOR * 1000 * np.e**log_est
    dollar_hi = SCALE_FACTOR * 1000 * np.e**log_hi
    dollar_low = SCALE_FACTOR * 1000 * np.e**log_low
    
    rounded_est = np.around(dollar_est, -3)
    rounded_hi = np.around(dollar_hi, -3)
    rounded_low = np.around(dollar_low, -3)
    
    print(f'The estimated property value is {rounded_est}.')
    print(f'With {conf}% confidence the valuation range is')
    print(f'USD {rounded_low} at the lower end to USD {rounded_hi} at the high end.')