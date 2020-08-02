# Human-Activity-Recognition-using-wearable-sensors-and-machine-learning
A human activity recognition model is developed using accelerometer data extracted from sensors and a machine learning approach is used for multi-label activity recognition.

Human activity recognition is the problem of classifying sequences of accelerometer data recorded by specialized harnesses or smart phones into known well-defined movements.
It is a challenging problem given the large number of observations produced each second, the temporal nature of the observations, and the lack of a clear way to relate accelerometer data to known movements.
Classical approaches to the problem involve hand crafting features from the time series data based on fixed-sized windows and training machine learning models, such as ensembles of decision trees. The difficulty is that this feature engineering requires deep expertise in the field.
Here, we extract real time data from a smartphone’s accelerometer (mainly velocity changes across three axes i.e., X, Y and Z). Accordingly, 9 additional features are recorded which include the following:
1. X^2
2. Y^2
3. Z^2
4. X_fd
5. Y_fd
6. Z_fd
7. X_sd
8. Y_sd
9. Z_fd

And the activities recorded are as follows:

![image](https://user-images.githubusercontent.com/66628385/89117424-50ee7880-d4bb-11ea-81ad-42f5316c43ea.png)

# Experimental procedure:


