# Term Deposit Marketing
The purpose of this project was to determine whether a customer was going to subscribe to a term deposit product - a product that a bank offers that yields interest and is not accesible until the stipulated time period has ended; usually 3,6, or 9 month time periods. The data contains customer demographics such as age, marital status, job, balance, credit in default, etc., as well as campaign info that includes the last month that they contacted each customer, method of contact, the duration of the call for each customer and the number of times that the bank contacted each customer. Sensative customer data has been kept from the dataset for privacy reasons.  

## Data Exploration
- First, the data looked to be clean - no missing values.
- Second, looking at the unique values, I noticed that to make matters easier for future processes that to convert the 'yes'/'no' categorical values to binary 1/0 values. Along with that I changed the 'object' dtypes to 'category'. Both of these steps nearly halved the memory usage that the dataset was using.
- The next step was to look at the categorical features to see what the distribution of the different values were across each categorical feature. After that step was finished we could see that dataset was highly skewed towards the negative class - the class of customers that did not purchase the term deposit product; over a 9 - 1 ratio.
- Next was to look at the numerical features and see if there were any outliers or any correlations between feature that would constitute removing from the dataset. Only a few features seemed to have heavy outlier presence. As correlations between features was concerned, no noticable linear correlations were found, but exponential correlations between a few features were present and then explored.

## Modeling
- The two sections of modeling were split into looking for 1.) the recall score of the positive class to see how well the model was at identifying customers who were likely to subscribe the the term deposit product and 2.) precision for the positive class to see how well a seperate model would perform at identifying customers who were not likely to subscribe and thusly eliminating unnecessary call time.
- Using Pycaret, models were looked at for recall and precision score with multiple sampling techniques used as well...RandomOversampler, SmoteTomek, SmoteENN
- The models that performed the best on the best sampling technique were chosen to explore further:
- First Model
* Logistic Regression
* Ridge Classifier
* LinearDiscriminantAnalysis
* GaussianNB
* SGDClassifier (SMOTEenn)
- Second Model
* LightGBM (SMOTEenn)
* ExtraTreesClassifer (RandomOverSampler)

- After using Optuna for hyperparameter tuning the SDGClassifier was then used to train on the whole, unsampled dataset where a ~90% recall score was achieved for the positive class

- After the same process the second model (LightGBM using SMOTEenn) achieved a 36% precision score for the positive class

- After performing RecusiveFeatureElimination (RFE) and RFECV the same model was able to classify over 30% more false positives on half the features from the original dummied dataset 

## Conclusion
Using a SGDClassifier model with SMOTEenn for the oversampling technique the first section of modeling was able to classify customers who were likely to subscribe to the term deposit product at over a 90% rate
The second section used a LightGBM model with SMOTEenn for the oversampling technique to achieve a reduction in call time for call center employees by nearly 60 hours while using half of the dataset's dummied features

# Set-up
* python --version 3.9.13

To get all the necessary packages run the command:
 - pip install -r requirements.txt