This program is based on a subset of NetFlix dataset, which is not allowed to be distributed
So the data parsing part of the grogram can be ignored
Prepare: Put the hw4.py file in the same folder of all HW4 data and eval data and script
Please notice that the program requires a scipy version of 0.17.0 and a numpy version of 1.10.4 (or higher)
Usage: python hw4.py
Output:
The code will generate corpstat.txt file for corpus statistic analysis

It will generate output files for experiment 1-3 based on k = 10, for each experiment, there are prediction score files for:
dot product similarity and normal averaging
dot product similarity and weight averaging
cosine similarity and normal averaging
cosine similarity and weight averaging

It will also generate prediction score file for experiment 4 based on D = 5

After each file generation, the code will call the eval python script to calculate RMSE for that file

In the code, there are three boolean parameters that can be set to be True to perform the complete set of experiments
__sweepK__ for enabling the sweep of k with values 10, 100, 500
__sweepD__ for enabling the sweep of D with values of 2, 5, 10, 20, 50
__generateTest__ for enabling the generation of score prediction file based on test set file test.csv
By default, they are set to be False