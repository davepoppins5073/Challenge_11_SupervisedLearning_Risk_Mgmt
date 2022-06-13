# Module 12 Report Template

## Overview/Summary of Analysis and Findings

When it comes to credit risk there is a central issue. Its a clasffication issue: namely healthy loans easily outnumber risky loans. In other words when it comes to prediction it harder to predict the risky loans because they are not well represented in the datasets available.  In this exercise we first establish that presence of an imbalance in the data set  provided using the value count function in python. Once we establish that there is an overbalance issue we attemp to use an oversampling technique to address this. In the end the the model that used the over-sampled data was more accurate at predicting unhealthy loans

## Methodology:
Our methodology was 3 fold:

#### Split the Data into Training and Testing Sets
<img width="1220" alt="Screen Shot 2022-06-12 at 11 58 12 PM" src="https://user-images.githubusercontent.com/101449950/173277095-7666b191-1e52-4178-91c3-52aca7ac9c35.png">


1. Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.
2. Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.
3. Check the balance of the labels variable (`y`) by using the `value_counts` function.
4. Split the data into training and testing datasets by using `train_test_split`.

#### Create a Logistic Regression Model with the Original Data

1. Fit a logistic regression model by using the training data (`X_train` and `y_train`).
2. Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.
3. Evaluate the model’s performance by doing the following:

    * Calculate the accuracy score of the model.
    * Generate a confusion matrix.
    * Print the classification report.

#### Predict a Logistic Regression Model with Resampled Training Data 
1. Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data.
2. Use the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.
3. Evaluate the model’s performance by doing the following:

    * Calculate the accuracy score of the model.
    * Generate a confusion matrix.
    * Print the classification report.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: Logical Regression
  * What is it? 
  * Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables. Logistic regression predicts the output of a categorical dependent variable.


* Machine Learning Model 2:Logical Regression w. Resampled Data:
  * The RandomOverSampler Modules is a python modules that employs Random oversampling - a technique that duplicates examples from the minority class in the training dataset and can result in overfitting for some models.

## Summary

<img width="710" alt="Screen Shot 2022-06-12 at 11 48 25 PM" src="https://user-images.githubusercontent.com/101449950/173276064-d105d653-0dba-4256-a56f-abe1ee61387d.png">


The balance accuracy score increased from 0.94 to 0.99. The balance accurancy score is the proportion of correctly identified negatives over the total negative prediction made by the model. The increase seems promising but its not until we check out the classification report that we will get a more nuanced understanding of whats happening.

The precision values stayed the same for healthy loans and for high risk loans. To recap that means that out of all the times that the model predicted a testing data observation to be the value 0 (healthy loans), 100% of those predictions were correct. On the other hand, out of all the times that the model predicted a value of 1, only 87% of those predictions were correct.

The next thing we need to look at is recall. Recall is a measure of the classifier's completeness; the ability of a classifier to correctly find all positive instances. For each class, it is defined as the ratio of true positives to the sum of true positives and false negatives. The recall in this case increased from 0.89 to 1. What this increase in the recall means is that the model that used the oversampled data was more accurate at predicting unhealthy loans



