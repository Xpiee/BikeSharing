# Bike Sharing Demand

Bike Sharing Demand is a Kaggle competition, in which we need to forecast bike rental demand by analyzing and combining historical usage patterns with weather and season data.

## Objective

The objective of my project is to train a model on a challenging mid to large-scale dataset that can predict the number of rented bikes on a day at a particular time. Based problem statement, I understand that it is regression problem on a time series data. Based on my preliminary analysis, regression algorithms like support vector regression, random forest regression, and boosting techniques can be used to forecast the count of rented bikes.

## Exploratory Data Analysis and Feature Engineering
Now, I would try to identify important patterns and would try to extract insights from the dataset so that we can better understand the role and significance of each attribute and its effect on the dependent variables. Before starting the data analysis, lets have a quick look at how our dataset is arranged. For convenience, I have merged the training and test datasets so that each transformation on every attribute is applied to training and test datasets equivalently.

### Extracting Important Information from ‘Datetime’ Column

As we can observe from the first 5 rows of our dataset that the ‘datetime’ column has a lot of information that we could exploit. So, as a first step, we would split the ‘datetime’ and extract information like date, hour, month, and weekday (name of the day of a week e.g., Saturday).

By splitting the ‘datetime’ column we could extract the following new columns:

1. Hour: This column represents the hours on a particular date. The value of this variable varies from 0 – 23. Based on its nature, we can say that this variable is categorical and cyclic in nature since, ‘0’ and ‘23’ are closely related i.e., 12:00am and 11:00pm. So, we would either consider it as a categorical feature or will apply ‘sine’ and ‘cosine’ transformations such that our machine learning algorithm takes this attribute as a cyclic attribute.
2. Month: This column contains the name of the month on which the entry was registered e.g., 1: January, 2: February, etc. This is also a categorical variable.
3. Weekday: This column contains the name of day e.g., Saturday, Sunday, etc. This is a categorical variable.
4. Year: This column contains the year in which the entry was registered. 
 
### Summary of Columns and Checking for Missing and Null Values in the dataset

Now that we have split the ‘datetime’ columns, we will have to check if there are any missing or null values in the dataset. Along with it, we can also, quickly analyze first, last, maximum, and minimum values of each column and their entropy.

Based on the table below, we can observe that there are no missing values in this dataset. However, there are couple of more interesting insights that we can extract from this. These points are mentioned below:

1. There are no missing values in this dataset. However, column ‘windspeed’ as a lot of values as ‘0’. This is unusual since, it is very rare when we have zero windspeed.
2. The following variables has a limited ‘unique values’ and hence they should be considered as categorical: season, holiday, workingday, weather, hour, month, weekday, day, year.
3. There are no negative values in the dataset i.e., there are no false entries in the dataset.

### Predicting Zero Values of ‘windspeed’
As we know that there are zero values in windspeed, we can use Random Forest Regressor with default values to predict these zero values and then replace zero values with them. To predict the values, we would use the attribute that can directly affect the values of windspeed. Hence, we would use the following attributes: "season", "holiday", "workingday", "weather", "weekday", "month", "year", "hour".
We would use the following hyperparameters for our random forest regressors: n_estimators=1200, max_depth=180, max_features='auto'.


### Analysis of Outliers and Skewness in the column ‘count’
After predicting the values in windspeed, we should now check if there are any outliers in the ‘count’ dependent variable. Also, we would check the distribution of the ‘count’ variable.

![Distribution](images/bikedist.png)
Figure a. Distribution of Count variable, b. Distribution of Log transformed Count variable, c. Distribution of Cube root transformed Count variable

The above fig. 1 (a) shows that the dependent variable ‘count’ is highly right skewed i.e., there are a lot of values on the right side of the curve making a tail on the right side. Skewness is an important issue that we need to address as this can make our algorithm make wrong predictions. To deal with the right skewed data we can apply the following transformations:
1. Log Transformation
2. Cube Root Transformation

As we can see, in the fig. 1 (a) and (b), that taking a log transformation improves the distribution of variable, but there is still skewness in the data, meanwhile, the cube root transformation greatly reduce the skewness in the data. However, during modelling, we got better results with log transformation of ‘count’ rather than cube root transformation of ‘count’. Hence, we dropped the cube root transformation root.

Now, we would analyze the outliers in the dependent variable ‘count’. To do that, we would plot the boxplots of count vs different independent variables. As we can see in the plots below that there are outliers in the ‘count’ variable. Also, we can observe following points from the plots below:
1. There are outliers in the variable ‘count’. However, the effect of outliers is greatly reduced when we apply log transformation on ‘count’.
2. The season ‘Spring’ has a smaller number of counts since there is a significant dip in the median. So, we can make a hypothesis that ‘temperature’ would play an important role in forecasting ‘count’ since, in spring outside temperature would not be favorable for bike renting and riders would prefer other modes of transport.
3. There are more outliers on a working day. From this we can conclude that these outliers are due to high demand of bikes on a working day and are not entered in the data erroneously. Hence, we would not remove these outliers as removing them will lead to more information loss. We would use log transformation to reduce the effect of these outliers.

![Boxplot of ‘count’ and ‘log of count’ with ‘season’ and ‘working day’](images/boxplot.png)


## Visualization of Count During a Day with respect to Independent Variables

Now, we would try to visualize that how does the count changes during a day with respect to different independent variables like season, day of a week, holiday, working day. Also, we would try to extract essential patterns and trends from this analysis and engineer new features based on those patterns. These engineered features would be very useful while training our algorithm and would help our algorithm to model these patterns to predict count vales more accurately.
Firstly, we would check how does the ‘season’ affects the count of bike rentals during a day. Note: Complete code of EDA and Feature Engineering can be found in the attached Html file named “Bike Data 4 EDA and Feature Engineering”. Based on the fig. 3, we can say that there is an increase in values of bike rentals during 7-8am and 5-7pm. This could mean that there is an increase in bike rentals during office/school timings. We can further investigate this hypothesis by studying the count with respect to ‘casual’ and ‘registered’ users as well as analyzing the count on weekdays and weekends. Based on our hypothesis, the count should go down on weekends during office/school timings.

![Line plot showing User Counts by hour of a Day across Seasons](images/dayseason.png)
Line plot showing User Counts by hour of a Day across Seasons

Also, as mentioned earlier (and confirming our hypothesis about spring season), it can be seen from the fig. 3 the count is low for season 1 i.e., spring season. Hence, the weather, especially, the temperature and humidity would play an important role in predicting the number of rental bikes since, people would prefer to rent a bike when the weather and temperature is warm or favorable.

![Showing the distribution of counts with working day (and non-working day) during a day](images/workingnon.png)
Showing the distribution of counts with working day (and non-working day) during a day

Based on the fig. 4, we can say that there is peak in bike rentals at 7-8am and 5-7pm on a working day as well as there is a peak between 10am-3pm on non-working days. This is a very important pattern and we can extract this feature into a different column that would have value ‘1’ during peak timings i.e., during 7-8am and 5-7pm on a working day and during 10am-3pm on non-working day. Same insight can be drawn from fig. 5, which shows that the peak timings on weekends and weekdays.


![Distribution of Bike rentals on Days of Week](images/days.png)
Distribution of Bike rentals on Days of Week


Based on our hypothesis that the weather conditions like temperature, humidity, and windspeed would greatly impact the number of bike rentals on that day. Hence, we tried to visualize the distribution of bike rental count with respect to the windspeeds. We observed that, fig. 6, the count of bike rentals decreases significantly if the windspeeds are higher than 30-35. Hence, we created a new feature ‘best condition’ that has value ‘1’ for each entry that has temperature less that 27 and windspeed less than of equal to 30. Also, for a working day, ‘not favorable’ when humidity is more than 60.
