# script to take a .pkl file containing a trained model with CP TF<2.14, 
# and convert it to a numpy format that is readable even if the TF version is >=2.14,
# namely, avoiding pickle.
# You only need to run this once using TF<2.14 (was tested on TF=2.13), 
# and this creates the numpy file replacing the pickle file.
# After running this script you should be able to run CPJ with TF>=2.14.
# Inputs are name of the pkl file to convert, and whether pca was used or not.

import numpy as np
import pickle
import sys, ast

# specify the pickle filename in trained_models without .pkl extension,
# either by passing it as first argument or hardcoding it here
try:
    pkl_filename = sys.argv[1]
except:
    print('Pickle filename not passed as input, so falling back to file already in trained models')
    pkl_filename = 'tf_cmb_template' # this is one of the SPT models available from CP

# also select if PCA was used, as in that case the file had a different structure
try:
    pca = ast.literal_eval(sys.argv[2])
except:
    # if nothing is given as input, just assume no pca is needed
    print('No PCA flag specified, will assume the pkl file had no PCA involved')
    pca = False 

with open(f'./cosmopower_jax/trained_models/{pkl_filename}.pkl', 'rb') as f:
      pickle_file = pickle.load(f)

# you can change the list of variable names below in case your model is different
if pca:
    variable_names = ['weights_', 'biases_', 'alphas_', 'betas_', \
                      'param_train_mean', 'param_train_std', \
                      'feature_train_mean', 'feature_train_std', \
                      'training_mean', 'training_std', \
                      'parameters', 'n_parameters', \
                      'modes', 'n_modes', \
                      'n_pcas', 'pca_matrix', \
                      'n_hidden', 'n_layers', 'architecture']
else:
    variable_names = ['weights_', 'biases_', 'alphas_', 'betas_', \
                  'param_train_mean', 'param_train_std', \
                  'feature_train_mean', 'feature_train_std', \
                  'n_parameters', 'parameters', \
                  'n_modes', 'modes', \
                  'n_hidden', 'n_layers', 'architecture']

# check that the correct variables are being saved
assert len(variable_names) == len(pickle_file), "Length of loaded variables is inconsistent, make sure the PCA flag is used only if loading a PCA model"
# create the new dictionary, and save it with the same name (but different extension)
new_dict = {name: value for name, value in zip(variable_names, pickle_file)}
np.savez(f'./cosmopower_jax/trained_models/{pkl_filename}', new_dict)
