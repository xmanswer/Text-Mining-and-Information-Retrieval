The src folder contains 6 python files. Please locate all of them in the same directory as Netlifx dataset.
For all scripts running correctly, scipy version of 0.17.0 is required. In additional, numpy package is required
for dense matrix operation, and sklearn package is required for SVM (used LinearSVC, which is the same as the one
in LIBLINEAR).

baseline.py:
original code from hw 4, baseline performances of memory based mthod, model based
method and PMF are generated using this script

main_script.py:
Used to conduct experiment and generate predicting results
for movie recommendation from a subset of Netflix dataset, locate this script
with other python files as well as data set files in the same folder. Intermediate
.npy files may be generated.  Two switches can be enabled in order to conduct 
development experiment (__dev__)and generate final test set predictions (__test__)

preprocess.py:
contains functions used to parse traing set, development set 
and test set. It also contains functions to generate prediction results files
for development and test

PMF.py:
This is a class for PMF. contains necessary fields such as rating matrix, 
latent feature matrices, and learning parameters such as lambdas for U and V, 
learning rate, etc. It also contains necessary methods for constructing the 
latent feature matrices U and V, and predicted rating matrix R_hat, through 
gradient descent. It also provides the method for constructing training samples 
from rating matrix R.


RMLR.py:
This class provides functions to do training and classification. It pack three 
different gradient ascent methods (aka gradeint ascent, stochastic gradient ascent and 
batched-stochastic gradeint ascent). It also contains functions to calculate 
the graident and objective function for given sparse matrix and weight vectors.

eval_ndcg_mod.py:
modified version of the eval_ndcg, run function can be called by other script
the print contents are also disabled
