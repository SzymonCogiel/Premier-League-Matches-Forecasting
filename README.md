# Football Result Forecasting

## Overview
A machine learning project created in the free time to predict the results of Premier League matches.
The model is based on data on English league match results from the 2009/2010 to 2018/2019 seasons downloaded from the page below:

http://www.football-data.co.uk/englandm.php

and other calculated statistics.
Logistic Regression and Random Forest performed the best during the classification. Finally, after tuning the hyperparameters, I chose multiclass Logistic Regression which achieved an accuray of 61.6% for the test set.
The model has achieved really satisfactory results, considering that the completely random model gives us a precision of 33%.
## Model overview

![kedro-pipeline (4)](https://user-images.githubusercontent.com/81774440/156662253-c02ed9bb-7a7c-445c-9fa5-0641d259da14.png)


List of pipeline nodes with a description:
- [Combined Seasons]()

- [Preprocess Categorical Node](https://github.com/SzymonCogiel/Premier-League-Matches-Forecasting/tree/master/src/Football_Result_Forecasting/pipelines/data_processing)

- [Preprocess Numerical Node](https://github.com/SzymonCogiel/Premier-League-Matches-Forecasting/tree/master/src/Football_Result_Forecasting/pipelines/data_processing)

- [Split Data Node](https://github.com/SzymonCogiel/Premier-League-Matches-Forecasting/tree/master/src/Football_Result_Forecasting/pipelines/data_science)

- [Train Model Node](https://github.com/SzymonCogiel/Premier-League-Matches-Forecasting/tree/master/src/Football_Result_Forecasting/pipelines/data_science)

- [Evaluate Model Node](https://github.com/SzymonCogiel/Premier-League-Matches-Forecasting/tree/master/src/Football_Result_Forecasting/pipelines/data_science)

