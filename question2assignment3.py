import pandas as pd
import matplotlib.pyplot as plt

crime_data = pd.read_csv("crime1.csv", delimiter = ",")

plt.hist(crime_data["ViolentCrimesPerPop"], edgecolor = "deeppink", color = "pink")
plt.title("Crime Rate Per Population Frequency")
plt.xlabel("Crime Rate Per Population")
plt.ylabel("Frequency")
plt.show()

plt.boxplot(crime_data["ViolentCrimesPerPop"])
plt.title("Boxplot of Violent Crimes Per Population")
plt.xlabel("Population")
plt.ylabel("Violent Crime Rate")
plt.show()

"""
Histograms show the frequency of each value of a dataset.
This histogram shows that the lower crime rates are more frequent in each population compared to the higher crime rates.
We can see that the bars on the lower half of the histogram are higher than the upper half, meaning that lower crime rates per population are more frequent than higher crime rates per popoulation

The box plot does not suggest the presence of outliars, as there are no separate points past the "T" shapes at the ends of the bar graph.
The boxplot shows that the median is close to 0.4. The median is closer to the lower edge of the box than the upper edge.


"""
