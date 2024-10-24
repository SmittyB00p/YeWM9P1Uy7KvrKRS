# Term Deposit Marketing

<!-- _includes/youtube.html -->



## About the Dataset
The purpose of this project was to determine whether a customer was going to subscribe to a term deposit product - a product that a bank offers that yields interest and is not accesible until the stipulated time period has ended, which can be from one month to a few years. The data contains customer demographics such as age, marital status, job, balance, credit in default, etc., as well as campaign info that includes the last month that they contacted each customer, method of contact, the duration of the call for each customer and the number of times that the bank contacted each customer. Sensative customer data has been kept from the dataset for privacy reasons. 

## Objectives
* Predict whether the customer will subscribe to the product
* Find which customers are more likely to subscribe as well as the top feature to focus on 

## Data Exploration
- First, the data looked to be clean - no missing values.
- Second, looking at the unique values, I noticed that to make matters easier for future processes that to convert the 'yes'/'no' categorical values to binary 1/0 values. Along with that I changed the 'object' dtypes to 'category'. Both of these steps nearly halved the memory usage that the dataset was using.
- The next step was to look at the categorical features to see what the distribution of the different values were across each categorical feature. After that step was finished we could see that dataset was highly skewed towards the negative class - the class of customers that did not purchase the term deposit product; over a 9 - 1 ratio.
- Next was to look at the numerical features and see if there were any outliers or any correlations between feature that would constitute removing from the dataset. Only a few features seemed to have heavy outlier presence. As correlations between features was concerned, no noticable linear correlations were found, but exponential correlations between a few features were present and then explored.

## Modeling
- The two sections of modeling were split into looking for 1.) the recall score of the positive class to see how well the model was at identifying customers who were likely to subscribe the the term deposit product and 2.) a mix of precision and recall for the positive class as well as f1 macro score and the true negatives to see how well a seperate model would perform at identifying customers who were not likely to subscribe and thus reducing call time.
- Using Pycaret, models were looked at for recall and precision score with multiple sampling techniques used as well...RandomOversampler, SmoteTomek, SmoteENN
- The models that performed the best on the best sampling technique were chosen to explore further:
### First Model
* Logistic Regression
* Ridge Classifier
* LinearDiscriminantAnalysis
* GaussianNB
* SGDClassifier (SMOTEenn)
### Second Model
* LightGBM (SMOTETomek)
* ExtraTreesClassifer (RandomOverSampler)

- After using Optuna for hyperparameter tuning the SDGClassifier was then used to train on the whole, unsampled dataset where a ~90% recall score was achieved for the positive class which equates to roughly 130 hours of saved call time (using median call time)

- After the same process the second model (LightGBM using SMOTETomek) achieved a 36% precision score for the positive class, over 80% for both classes and over 6500 false negatives resulting in over 330 hours of call time saved

- After performing hierarchical and KMeans clustering, the customers that need to be prioritized are those who have secondary or tertiary education levels and those who have either technician or management jobs. The feature that makes the customer subscribe is their balance amount.

## Conclusion
* Using a SGDClassifier model with SMOTEenn for the oversampling technique the first section of modeling was able to classify customers who were likely to subscribe to the term deposit product at over a 90% rate resulting in over 130 hours saved
* The second section used a LightGBM model with SMOTETomek for the oversampling technique to achieve a reduction in call time for call center employees by nearly 300 hours
* The customers that need to be prioritized are those who have secondary or tertiary education levels and those who have balances over €5,000.
* The feature that makes the customer subscribe is their balance amount.

* An interactive dashboard using Tableau can be found at this link: https://public.tableau.com/app/profile/tyler.smith5879/viz/Term_Deposit_Dashboard/TermDepositStory

# Set-up
* python --version 3.9.13

To get all the necessary packages run the command:
 - pip install -r requirements.txt
