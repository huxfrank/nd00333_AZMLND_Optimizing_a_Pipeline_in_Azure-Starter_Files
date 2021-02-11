# Optimizing an ML Pipeline in Azure
# Table of Contents
1. [Overview]{#overview}
2. [Summary](#summary)
3. [Scikit-Learn Pipeline](#sklpipeline)
4. [AutoML](#auotml)
5. [Pipeline Comparison](#pipeline-compare)
6. [Future Work](#future-work)
 
## Overview <a name="overview" />
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary <a name="summary" />
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
This dataset is related to the direct marketing campaigns (based on phone calls) of a banking institution.
It contains personal data about the applicant (marriage status, age, housing, education etc) as well as if the client was contacted previously either about this campaign or another campaign and how long ago the client was contacted.
This dataset also contains data about social and economic context attributes such as employment variation rate, consumer price index, number of employees, and the consumer confidence index.
The dataset is multivariate and  contains ~45,000 instances spread across 17 attributes.
The goal is to use classification to determine if a client will subscribe to a term deposit (column 'y').

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best performing model was a EnsembleVoting method that used ['XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'RandomForest'] with varying weights. It was the automl model.

## Scikit-learn Pipeline <a name="sklpipeline" />
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
For the SKLearn Hyperdrive pipeline architecture, we first took the raw data and put it into a Tabular dataset so it would be easier to manipulate and then we ran it through a clearning function that took
the information provided and one-hot encoded it to make it easier to see / use, primarily when it came to education level. We then split out the results of the application into their own column.

After I cleaned the data, we split it up into training and validation sets and then the training script set up logging for Accuracy as well as parameters for the run itself such as regularization strength
and max iterations at 1.0 and 100 respectively. The training script then ran a logistic regression on the training data and scored it against the validation set, logging the result as Accuracy. It also
established a place for models to be saved as well.

Now that the training script is complete, it was time to set up the SKLearn estimator and Hyperdrive configuration. The SKLearn estimator config was then run using the compute i created for this with train.py
as the entry script and the hyperdrive config was set up to maximize accuracy (logged from the training script) over 12 runs with 4 max concurrent runs at a time using random parameter sampling with a early 
termination BanditPolicy using evaluation_interval=2 and slack_factor=0.1.

**What are the benefits of the parameter sampler you chose?**
For learning rate, I chose normal(10,3) with 10 and 3 as my bounds because very large learnings rates result in unstable training and very small learning rates result in an inability to train. 
I tried running the same model with 20,2 as expanded bounds but saw little change in the results.
The best model with 10,3 was ['--keep_probability', '0.06240641242744088', '--learning_rate', '9.513857773108702'] with accuracy = 0.9133.
The best model with expanded parameters was ['--keep_probability', '0.05106288981937711', '--learning_rate', '18.58767652434821'] with accuracy = 0.9133. If we can achieve the same accuracy with smaller bounds, it
will speed up our runs and not have to go through as many parameters to achieve the best result. 

I went with random parameter sampling because it supports early termination of low performance runs which optimizes your time spent and its good for initial searches. It was the better choice over Grid sampling because grid sampling 
only supports discrete hyperparameters and we want to sweep over a range of continuous values for some of our hyperparameters. Random parameter sampling is also the better choice over Bayesian sampling because bayesian sampling
requires max_#_of_runs >= 20*(# of hyperparameters tuned) for best results and we aren't doing nearly enough runs to fully utilize this sampling method. 
Both Grid sampling and bayesian sampling work best if there is enough budget to exhaustively search over the search space and that isn't necessary for the purposes of this project so Random sampling is the better choice in terms of 
saving money as well. 

**What are the benefits of the early stopping policy you chose?**
The bandit policy is good because it will cancel any run where the primary metric is smaller than (Metric + Metric*Slack_Factor). This allows for early termination of low performance runs as well.
Bandit is the better choice over the other two methods because it is the most aggressive about saving money but doesn't terminate promising jobs; if you can save money without incurring a loss on your primary metric, why spend the extra money?
Bandit is also the best choice here because oftem automl runs are used as a tool to determine a starting point for your hyperparameters / experiment and it's a good idea to not blow a large percentage of your budge on just finding your
starting point. It's better to maximize savings at this early step so if we require it later on, it's available.

## AutoML <a name="auotml" />
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
The AutoML config i used was classification with primary metric accuracy and 5 cross validations. 
The model it returned with the best accuracy was a VotingEnsemble which means it took multiple different classifiers and then weighted them to get the best result. With accuracy = 0.916, this method ended up being slightly better than the Hyperdrive run which had accuracy = 0.913. 

Here are the other hyperparameters:
ensemble_weights : [0.3333333333333333, 0.13333333333333333, 0.2, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.13333333333333333]
ensembled_algorithms : ['XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'RandomForest']
ensembled_iterations : [1, 0, 28, 14, 11, 15, 4]

These hyperparameters indicate that results of 7 different runs utilizing three types of models were aggregated together to determine the best result. From what is given, we can see that the 1st iteration run utilizing XGBoostClassifier was the biggest contributor to the final model 
and result, followed by the 28th iterative run of XGBoostClassifier and then the 4th iterative run of RandomForest and the 0th iterative run of LightGBM. As these are the biggest contributors to the final result, if I were to go in and do tweaks to improve performance, I would start with those models.

## Pipeline comparison <a name="pipeline-compare" />
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The hyperdrive model gave accuracy = 0.9133 and the automl run gave accuracy = 0.916 which performed slightly better but on average the automl run took almost 10x the time to arrive at that result. The differences may have been caused by the fact that the hyperdrive was running a set training script and only a single algorithm whereas the automl ran multiple algorithms and then aggregated / ensembled them to derive the best result. The latter process is more involved and intensive and naturally takes more time and resources but it provided a better result, albeit only by a slight margin.

## Future work <a name="future-work" />
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Some areas of improvement are in the code and framework itself. There were large chunks of the code that I had to comment out / remove because they were throwing errors. There is also a warning that SKLearn is deprecated and it recommended I do it another way. Pinning dependencies would be greatly beneficial for these issues because then it would insure that the execution of this project works as planned regardless of new updates to the frameworks. There was also an issue with importing OnxxConverter that I could not resolve.

Usually automl runs are used to establish a starting point. Future improvements to model accuracy would require me tweaking different hyperparameters myself and further cleaning / preparing the data in order to minimize that last 9% of accuracy error.


