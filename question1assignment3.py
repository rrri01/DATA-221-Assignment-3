import pandas as pd
import numpy as np

crime_data = pd.read_csv("crime1.csv", delimiter = ",")

# find the min and max:
list_of_violent_crimes_per_pop = []
for crime_rate in crime_data["ViolentCrimesPerPop"]:
    list_of_violent_crimes_per_pop.append(crime_rate)

minimum_crime_rate = min(list_of_violent_crimes_per_pop)
maximum_crime_rate = max(list_of_violent_crimes_per_pop)

mean = np.mean(crime_data["ViolentCrimesPerPop"])
median = np.median(crime_data["ViolentCrimesPerPop"])
sd = np.std(crime_data["ViolentCrimesPerPop"])

print(f"mean: {mean}\nmedian: {median}\nstandard deviation: {sd}\nminimum: {minimum_crime_rate}\nmaximum: {maximum_crime_rate}")

"""
the data is slightly skewed. the mean is 0.44119..., while the median is 0.39.

the median represents the halfway mark of the data, and the mean represents the average of all the data.
since the mean is higher than the median, this means that the points above the median are much further from the median than the points below the median.

when there are extreme data points, the mean is more affected than the median.
the value of each data point does not influence the median since the median is whatever point (or average of 2 points) in the center.
the value of each data point influences the mean, because the mean is the average of all the data.
because the median only measure what the middle point is, and the mean measures the average, the mean is more affected by extreme data points than the median is.

"""

