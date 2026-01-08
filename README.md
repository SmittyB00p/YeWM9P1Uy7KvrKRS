# Term Deposit Marketing Project
This project is aimed at assisting our client to provide machine learning solutions in the European banking market. Primarily, they work in fraud detection, sentiment classification and customer prediction and classification.

Using information coming in from call centers, they are interested in developing machine learning systems that will **improve their sucess rate of calls made** to customers for any products that their clients offer.

The purpose of this project is to determine whether a customer is going to subscribe to a term deposit product - a product that a bank offers that yields interest and is not accesible until the stipulated time period has ended, which can be from one month to a few years. 

The data contains customer demographics such as age, marital status, job, balance, credit in default, etc., as well as campaign information that includes the last month that the bank contacted each customer, method of contact, the duration of the call for each customer and the number of times that the bank contacted each customer. Sensative customer data has been kept from the dataset for privacy reasons. 

[![Term Deposit Video](https://img.youtube.com/vi/tyaM6dVxpLQ/0.jpg)](https://www.youtube.com/watch?v=tyaM6dVxpLQ)

### Main Goal:
* Predict whether the customer will subscribe to the product, ideally reaching ~81% average accuracy score using 5-fold cross validation.

### Secondary Goal:
* Find which customers are more likely to subscribe as well as determining which segments of the customers the bank should prioritize.

* What makes the customers buy? Which feature should be focused on?

### A Brief Note on Project Structure

This project is structured so that most of the exploratory data analysis, model selection, and model experimentation are housed in the `Notebooks` folder. 

```- Notebooks
    |
    - EDA
        |
        - eda.ipynb (initial data exploration)
        - customer_segmentation.ipynb (unsupervised learning techniques looking for customer segments)
    |
    - Models (sequence of models looked at)
        |
        - svm.ipynb
        - logistic_regression.ipynb
        - ensemble.ipynb
        - final_model.ipynb (final model testing)```

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

The numerical features `age` and `balance` looked to be correct as there was no one under the age of 19 and balances (in euros) were in the range of -8,000 to 100,000+.

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

A recall score (how many actual subscribers is the model finding), precision score (how many predicted subscribers actually subscribed), and f-1 score (the mean of precision and recall) will be apropos to this problem as well as experimenting with different techniques to deal with the imbalance in the dataset (SMOTETomek, RandomOverSample, SMOTENN, and Imbalanced Ensembling Techniques).

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

Tree based models and ensembling techniques will be experimented with as well as balanced tree and ensemble techniques.

#### Results

|          Model         |  Tuned Paramaters  |        Recall    | Precision  | Call Time Saved   |
|------------------------|--------------------|------------------|------------|-------------------|
| Balanced Random Forest | n_estimators       | Train	1.000000 | 0.427105   | 383.82 hours      |
| Classifier             | max_depth          | Test	0.891743 | 0.362957   |                   | 
|                        |                    |                  |            |                   |
| LinearSVC (l2) w/      | Penalty (l1, l2),  | Train	0.857143 | 0.344852   | 371.77 hours      |
| RandomOverSampler      | C                  | Test	0.855046 | 0.331909   |                   |
|                        |                    |                  |            |                   |
| Logistic Regression    | Penalty (l1, l2),  | Train   0.872914 | 0.860573   | 294.67 hours      |
| w/ SMOTEEnn            | C                  | Test    0.809174 | 0.229329   |                   |
|                        |                    |                  |            |                   |

The results show that a `BalancedRandomForestClassifier` is performing the best out of the few models we tried.

The call time saved is computed as: 
    
```(true negatives * mean call time of campaign) - (false positives * mean call time of campaign)```

If we substitute terms thus making it A - B, we can say that:
* A = mean call time of true negative predictions (no need to waste the time calling)
* B = mean call time of false positive predictions (wasted call time)

**The main take away from this section is that we have saved call center employees close to 400 hours of time by identifying which customers will and will not subscribe to the term-deposit product.**

## Customer Segmentation

The secondary objective of this project was to find "which customers are more likely to subscribe as well as determining which segments of the customers the bank should prioritize." In trying to find said segments I needed to scale the dataset and then perform principle component analysis (PCA) on the one-hot-encoded dataset that I used for modeling.

Results from the PCA are below:

![PCA](/images/dendrogram.png)

The dendrogram doesn't give us a very intuitive idea of how to cluster the sampled customers, but using a cluster of 4 will give us a less uniform grouping than 8.

#### Sampled Dataset Clustering
|    Hierarchical Clustering    |    0    |    1    |    2    |    3    |
|-------------------------------|---------|---------|---------|---------|
|      Subscription Status      |		  |         |         |         |		
|              0                |	53    |   53    |	31    |   36    |
|              1                |	4	  |   3     |   4     |   8     |

What this table is saying is that for the dendrogram above, the orange and green branches have 4 subscribers; the red and purple branches have 3 subscribers; the brown and pink branches have 4 and the rest have 8 subscribers.

The final step was to look at the whole dataset and obtain a label for each customer and see what the segments show.

#### Total Dataset Clustering
|    Hierarchical Clustering    |    0    |    1    |    2    |    3    |
|-------------------------------|---------|---------|---------|---------|
|      Subscription Status      |		  |         |         |         |		
|              0                |  12047  |  4222   |  12673  |  6601   |
|              1                |	508	  |   625   |   1287  |   372   |

The table shows that cluster 2 has the most subscribers of any of the other three groups. On a relative basis, group 1 has the biggest percentage of subscribers at 14.8%, followed by group 2 at 10.2%, then group 3 at 5.6% and lastly group 0 at over 4%.

Focusing on segments 1 and 2 are suggested since the subscription rate in these two groups are above the average subscription rate of 7%.


## Conclusion

Main goal:

* Predict whether the customer will subscribe to the product, ideally reaching ~81% average accuracy score using 5-fold cross validation.

Achieved:

* Since the dataset was imbalanced the accuracy metric was not going to be the metric to use in assessing model performance. Using a mix of precision and recall we were able to tune a `BalancedRandomForestClassifier` to obtain a test recall score of 89% and a test precision score of 36%. Thusly reducing call time for call center employees by roughly 383 hours.

Secondary Goal:

* Find which customers are more likely to subscribe as well as determining which segments of the customers the bank should prioritize.

* What makes the customers buy? Which feature should be focused on?

Achieved: 

* Using different unsupervised learning techniques I was able to segment the customers into 4 different groups which gave us a lead into which groups were to be prioritized. Groups 1 and 2 look to be the groups to prioritize since their average subscription rates are significantly higher than the total average subscription rate. 

* In regards to which feature makes the customers subscribe there is not a definitive answer that sticks out when looking at the features when grouped by each customer segment. Further exploration might need to be done to understand which features are playing a more integral role than others.


* An interactive dashboard using Tableau can be found at this link: https://public.tableau.com/app/profile/tyler.smith5879/viz/Term_Deposit_Dashboard/TermDepositStory

## Set-up
* create a virtual environment (venv) with any name (customary to use .venv for virtual environemnt name): `python3.11 -m venv .venv`
* to activate the virtual environment run the command: `source .venv/bin/activate`
* to install the necessary packages run the command: `pip install -r requirements.txt`
* use `.venv Python3.11.2` kernel for notebook usage
* for reproducability use the seed `4701`
