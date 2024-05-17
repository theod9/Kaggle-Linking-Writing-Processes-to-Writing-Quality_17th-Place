Competition: https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality <br>
Efficiency Track Leaderboard: https://www.kaggle.com/code/ryanholbrook/writing-processes-efficiency-lb

This repository contains the files for my solution to the competition. I focused exclusively on the efficiency track, as the challenge of balancing accuracy and speed was very compelling.

The preprocessing.py file contains functions that handle a variety of data preprocessing tasks using both Polars and Pandas libraries, focusing on reconstructing texts and extracting features from logs. It includes detailed functions for counting values, calculating statistics, reconstructing essays from logs, and aggregating data on word, sentence, and paragraph levels. 

The model_pipeline.py script integrates functions from preprocessing.py to load, preprocess, and combine a variety of features derived from raw data. It has functions to standardize and clean data, and employs feature selection using SelectKBest, which proved to be very efficient. Finally, it includes functionalities for training multiple models, making predictions, and combining these predictions through an ensemble method.

The notebook12.ipynb serves as the project's main execution file. It contains several models, primarily gradient-boosted models but also a Ridge regression, wrapped in a pipeline with a SplineTransformer for non-linear feature transformations. Hyperparameters were fine-tuned using the Optuna library (not shown here). The notebook implements ensemble to leverage the strengths of each model and finally generates the submission file.

