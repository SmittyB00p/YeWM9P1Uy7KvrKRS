# Term Deposit Marketing Project
The purpose of this project is to determine whether a customer is going to subscribe to a term deposit product - a product that a bank offers that yields interest and is not accesible until the stipulated time period has ended, which can be from one month to a few years. The data contains customer demographics such as age, marital status, job, balance, credit in default, etc., as well as campaign information that includes the last month that the bank contacted each customer, method of contact, the duration of the call for each customer and the number of times that the bank contacted each customer. Sensative customer data has been kept from the dataset for privacy reasons. 

[![Term Deposit Video](https://img.youtube.com/vi/tyaM6dVxpLQ/0.jpg)](https://www.youtube.com/watch?v=tyaM6dVxpLQ)

Main Goal:
* Predict whether the customer will subscribe to the product, ideally reaching ~81% average accuracy score using 5-fold cross validation.

Secondary Goal:
* Find which customers are more likely to subscribe as well as determining which segments of the customers the bank should prioritize.

* What makes the customers buy? Which feature should be focused on?

### A Brief Note on Project Structure

This project is structured so that most of the exploratory data analysis, model selection, and model experimentation are housed in the `Notebooks` folder. Within the `Notebooks` folder the `Models` folder has the models used for the two parts of this project.

The `Feature Importances` folder is for the final part of the project where we can visualize the most important features selected by the different models and to test the final model on the whole dataset.

## Exploratory Data Analysis
Initially, looking for any missing values and changing data types of features for memory purposes was initiated. The data analysis was then split between demographic features and campaign features.

### Demographic Predictors
The first 8 features of the dataset are related to customer demographics:
* age
* job
* marital
* education
* default
* balance
* housing
* loan

The numerical features `age` and `balance` looked to be correct as there was no one under the age of 19 and balances (in euros) were in the range of -8000 - 100,000+.

The binary variables of `default`, `housing`, and `loan` were all correctly labeled as either having credit in default, a home loan, or personal loan or not.

Categorical features, such as `marital`, `job`, and `education` were for the most part free of incongruities between the bank and what the data showed. For example, the `education` and `job` features had `unknown` as a category within both features. After further investigation, the 'unknowns' were dropped from both features due to the fact that they only comprised ~5% of the dataset and that there was no mention of what to do with such instances.

### Campaign Predictors
The next 5 features were related to the term deposit product campaign:
* contact
* day
* month
* duration
* campaign

The numerical feature of the campaign data, `day`, looked good. 31 days was the range of values. But the other two, `duration` and `campaign`, looked suspicious due to the fact that both features had very high max values: 
* `duration` had a max value that roughly equated to 80 minutes. This variable was 'the duration of the last call that the client made to this customer'.

* `campaign` had a max value of 58. This feature was 'the number of times the client called this customer during the entirety of the campaign'.

After further inspection of the top handful of outlier in these features it looked to be possible that the max values reported could in fact be true. In regard to `duration`, I do not know that if being on the phone with someone for over an hour to pitch them on a product is likely, but since there was only two instances, I left them in the dataset. As for the `campaign` feature, there definitely could be a case where the client has been trying to reach someone, once a week for 11 months and finally in the twelfth month they reach the customer.

### Target Variable

![target distribution](/images/target_distribution.png)

The countplot shows that the distribution is vastly skewed towards the customers who DID NOT subscribe. Because of this the metric that the client has asked to assess the model performance (accuracy 81%+) will not be a good metric to assess by. With no effort at all a null model will predict 93% accuracy, thus, worthless.

A recall score (how many actual subscribers is the model finding), precision score (how many predicted subscribers actually subscribed), and f-1 score (the mean of precision and recall) will be apropos to this problem as well as experimenting with different techniques to deal with the imbalance in the dataset (SMOTETomek, RandomOverSample, SMOTENN).


<!-- ### Correlations
Because the client wants to know the predictor that they should focus on different correlation methods were used to obtain a glimpse at what might be the most relevant predictors in the dataset.

|             Feature            |    F-stat   |   p-value    |
|-------------------------------------------------------------|
| duration_minutes	             | 10431.901211| 0.000000e+00 |
| month_3	                     | 705.764237  | 4.168074e-154|
| month_10	                     | 349.453644  | 1.235608e-77 |
| month_4	                     | 341.081820  | 7.920632e-76 |
| contact_unknown	             | 289.082333  | 1.360355e-64 |
| contact_cellular	             | 275.194708  | 1.373460e-61 |
| marital_married	             | 129.768635  | 5.146244e-30 |
| housing	                     | 117.994511  | 1.906511e-27 |
| marital_single	             | 108.355956  | 2.427728e-25 |
| education_tertiary	         | 85.615706   | 2.294716e-20 |
| campaign	                     | 62.425346   | 2.840836e-15 |
| job_student	                 | 56.675281   | 5.253134e-14 |
| month_5	                     | 56.465990   | 5.842175e-14 |
| month_2	                     | 52.045913   | 5.522056e-13 |
| job_blue-collar	             | 43.843069   | 3.604778e-11 | -->


## Modeling
The modeling phase of the project is split into two sections:
- find the highest recall and precision score using only demographic data
- find the highest recall and precision score using the whole dataset

### Demographic Modeling

![Demographic Models](/images/demographic_models.png)

![Undersampled Models](/images/undersample_models.png)

![Mixed Sampling Models](/images/mixed_models.png)

Using `Pycaret` different models were selected with different techniques to help with the imbalance in the target variable. Looking at the graphs the use of random oversampling or a combination of both over- and undersampling with a few different linear models was providing the best recall scores.

Since most of the models are fairly the same in the way they try to find a decision boundary between classes I will experiment with `LinearSVC` using the RandomOverSampling technique as well as `Logistic Regression` with a SMOTEEnn technique.

#### Results

Coming Soon

|        Model        | Tuned Paramaters   |        Recall   | Precision  | Call Time Saved   |
|---------------------|--------------------|-----------------|------------|-------------------|
| LinearSVC (l2) w/   | Penalty (l1, l2),  | Train	0.855808 | 0.345118   | 372.6 hours       |
| RandomOverSampler   | C                  | Test	0.853211 | 0.332856   |                   |
| Logistic Regression | Penalty (l1, l2),  | Train           |            |                   |
|                     | C                  | Test 


### Post-Campaign Modeling

Coming Soon

<!-- ![Post-Campaign Models](/images/post_campaign_models.png) -->


#### Results

Coming Soon

<!-- |            Model            |    Recall    |    Precision    |
|--------------------------------------------------------------|
| duration_minutes	          | 10431.901211 | 0.000000e+00    | -->












<!-- - The two sections of modeling were split into looking for:
    * 1.) the recall score of the positive class to see how well the model was at identifying customers who were likely to subscribe to the term deposit product and 
    * 2.) a mix of precision and recall for the positive class as well as f1 macro score and the true negatives to see how well a seperate model would perform at identifying customers who were not likely to subscribe and thus reducing call time.
- Using Pycaret, models were looked at for recall and precision score with multiple sampling techniques such as RandomOversampler, SmoteTomek, and SmoteENN
- The models that performed the best on the best sampling technique were chosen to explore further:

### First Section Models
* Logistic Regression
* Ridge Classifier
* LinearDiscriminantAnalysis
* GaussianNB
* SGDClassifier (SMOTEenn)

### Second Section Models
* LightGBM (SMOTETomek)
* ExtraTreesClassifer (RandomOverSampler)

- After using Optuna for hyperparameter tuning the SDGClassifier was then used to train on the whole, unsampled dataset where a ~90% recall score was achieved for the positive class which equates to roughly 130 hours of saved call time (using median call time)

- After the same process the second model (LightGBM using SMOTETomek) achieved a 36% precision score for the positive class, over 80% for both classes and over 6500 false negatives resulting in over 330 hours of call time saved

- After performing hierarchical and KMeans clustering, the customers that need to be prioritized are those who have secondary or tertiary education levels and those who have either technician or management jobs. 

- The feature that is the best indicator of customer subscription is their balance amount. -->

## Conclusion

Coming Soon
<!-- * Using a SGDClassifier model with SMOTEenn for the oversampling technique the first section of modeling was able to classify customers who were likely to subscribe to the term deposit product at over a 90% rate resulting in over 130 hours saved
* The second section used a LightGBM model with SMOTETomek for the oversampling technique to achieve a reduction in call time for call center employees by nearly 300 hours
* The customers that need to be prioritized are those who have secondary or tertiary education levels and those who have balances over €5,000.
* The feature that is the best indicator of customer subscription is their balance amount. -->

* An interactive dashboard using Tableau can be found at this link: https://public.tableau.com/app/profile/tyler.smith5879/viz/Term_Deposit_Dashboard/TermDepositStory

## Set-up
* create a virtual environment (venv) with any name (customary to use .venv for virtual environemnt name): `python3.11 -m venv .venv`
* to activate the virtual environment run the command: `source .venv/bin/activate`
* to install the necessary packages run the command: `pip install -r requirements.txt`
* use `.venv Python3.11.2` kernel for notebook usage
* for reproducability use the seed `4701`
