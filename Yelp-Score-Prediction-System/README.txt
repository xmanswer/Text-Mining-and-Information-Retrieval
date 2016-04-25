This is a supervised learning system using multi-class logistic regression 
and SVM for predicting Yelp restaurant scores based on comments. It is trained on
2GB of Yelp restaurant scores & comments dataset in JSON format.

Please put all four python files under the same folder as the JSON dataset of YELP
as well as stop list files. 

The main routine is main_script.py. To run the experiment,
just do python main_script.py. This will call necessary scripts to parse the data.
Notice that it will generate some temporary files for the data processing part. 
The purpose of this is to accelerate the experiment speed so that I do not have
to parse the raw dataset every time I do some experiment. The temporary files 
have the size of around 400 MB. 

For RMLR, the script will call train_RMLR and classify_RMLR to do training and
classification; for SVM, the script will call sklearn.svm.LinearSVC to do the
same task. So please make sure the python environment has the necassary package
such as scipy and sklearn.

The experiment switches in main_script.py can be modified to conduct certain experiement. 
By default only the tfidf one is turned on since it gives the best results.
For each experiemnt, it will generate result files, with one stores the training
accuracy, the other one stores the prediction of the scores (with the name of the
format like dev_trainAndClassify_$CLASSIFIERNAME_$FEATURENAME_$LAMBDA)