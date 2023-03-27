# covid-19-time-series

Scripts for Kaggle challenge https://www.kaggle.com/c/covid19-global-forecasting-week-4 

The goal is to model how the number of cases and deaths will escalate with the time.  

This script was used to prepare a kaggle notebook which discusses in detail each step of the analysis https://www.kaggle.com/nicolaorlandowork/covid19-global-forecasting-based-on-bayesianridge 

## Introduction

Here we show a simple notebook for a forecasting analysis of COVID19 cases or fatalities based on a Bayesian Ridge model. The idea is to use this simple classification model on a dataset which will be reorganised to re-cast the initial task (time series analysis) into a classification task which can be addressed with the Bayesian Ridge model. This approach is inspired by this post https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/.

The method and model have two main limitations

- The model is not aware of the intrinsic data nature: the cumulative cases/fatalities vs time can only grow, the model is not constrained to reproduce naturally this behaviour.
- The model is based on the available data; all the data is equally important in the model fitting process. This is clearly a limitation; let's for example consider the confirmed cases vs time. They will grow much faster at the beginning of the pandemic while, later on, the grow rate will flatten due to restrictive measures taken by individual countries. As result the model will have a tendency to over-predict the amount of cumulative confirmed cases vs time.

## Comments on the input data

We can easily spot a few properties of the dataset:

- It includes days without confirmed cases and fatalities; these entries are not useful to make a prediction of their time evolution. They will be dropped in the following.
- Some countries will be split by Province. While this makes sense as different provinces will have different evolution of the pandemic (maybe due to non uniform containment policy), the discussion in this notebook doesn't address this possibility, all provinces are all merged based on the country. But the extension to split any prediction by province is technically trivial.

 Here two plots showing the evolution of the pandemic in Italy. The first overlays the number of casualties and cases in the considered time range, the second shows the mortality rate. 

![image](https://user-images.githubusercontent.com/26884030/227993732-a4a98bcc-5b5d-4933-b080-17f89e647775.png)

![image](https://user-images.githubusercontent.com/26884030/227993865-51ae5628-a2ea-4d69-8c45-0fba6584ba44.png)

### Discussion

These two plots can be used to characterise the evolution of the pandemic in Italy. The first plot emphasise the nearly exponential growth in confirmed cases and fatalities as a function of the time during the period between the 21st of February and the 21st of March. The growth rate becames less severe towards the end of April. The last plot shows the mortality rate as a function of the time. Towards the beginning of May the mortality rate is about 15%. The lower mortality rate initially observed might be correlated with the fact that hospitals were not yet in critical conditions. The flattening of the mortality rate towards the end of April hints to a gradual restoration of the hospitals response to the pandemic and a proper recording of the cases and casualties.

## Recast the data

Now I try to prepare the data for making predictions on the evolution of the number of COVID-19 cases. As mentioned in the introduction, I follow the same method developed here https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/.

The method is further illustrated by a graph you can find here below. 

![image](https://user-images.githubusercontent.com/26884030/227995243-1697a316-046a-4d63-96af-c3d927297ed1.png)

For our specific case we will use the simplest approach where N=1, that is we try to predict the confirmed cases for the day N+1 based on the data from the day N and N-1. This approach as some advantages

Minimises the probability of overtraining the model Increase the likelihood that the model has to learn some intrinsic data patterns for our problem, e.g. Value N > Value N-1. 

## Training the model

Now it is time to train our BayesianRidge model. The current setup is not necessarily the best. For example we don't optimise the model and we don't make any educated (based on expected data behaviour) guess of the best possible hyperparameter setting for the model.

## Discussion of data vs fitted model

![image](https://user-images.githubusercontent.com/26884030/227997217-1994da3d-4ac8-4806-95cd-d37d19830043.png)

![image](https://user-images.githubusercontent.com/26884030/227997246-65fcae35-e393-4e2f-bcdc-19ea4c669573.png)

Above you can see two example of plots showing how well the model is able to fit the data, here some considerations

The first plot shows the ratio of the data over the score of the fitted model. Note that all the data shown in this plot is used to train the model. You can see that the model severely over-predicts the number of cases before mid March. This can be caused by lack of flexibility of the fit model and/or severe under-reporting of cases in the initial COVID-19 outbreak whch is not consistent with the overall grow rate of the COVID-19 cases.
The second plot shows overall the linearity response of the fitted model as the number of cases increases.

## Make predictions

Now it is time to make some predictions on the testing set. Here we follow two sequential steps

- Apply scores on testing set that overlaps with the training set.
- Apply scores on the non-overlapping days between training and testing sets.

## Comments and further steps

The test set 'dataframe_merged' incorporates data from the testing set and the model 'predictions' for the days overlapping with the training set. The predictions for the remaining days will have to be filled up based the data available from the previous days. This is done with an iterative procedure as shown below. For each iteration the following operations take place

We split the dataset into two components; one, say ds1, with scores (predictions) available, the other, say ds2, without predictions.
We extract the row in the dataset which is in order the first to not have a prediction.
We apply the model on the row obtained in step 2.
Finally we define a new dataset which glues ds1 with ds2 which now has its first row modified to include the model prediction, with this new dataset we go back to step 1.

## Final steps

The final step is to incorporate the results from the previous section into the test dataset. Let's do it and have a quick look at the results.

Comments

We inspect now our results (incorporated in 'final_output').

In this simple scenario the model seem to be able to predict growing cumulative COVID-19 cases vs time
How does the model prediction compare to the actual data? I am publishing this notebook on 08/05. The model prediction for today is approx 224349 cases to be compared to the actual cases 215858 as reported here https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases. The model is pretty accurate (~95% accuracy) but it is also tuned on a training data which has a strong overlap with the test data.
As expected, according to the discussion in the introduction section, the model tend to over-predict the amount of cases due to the weight of the data from early stage of the pandemic.
