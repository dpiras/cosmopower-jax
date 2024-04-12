# script to take a .pkl file containing a trained model with CP TF<2.14, 
# and convet it to a numpy format that is readable even if the TF version is >=2.14,
# namely, avoiding pickle.
# You only need to run this once using TF<2.14 (was tested on TF=2.13), 
# and this creates the numpy file replacing the pickle file.
# After running this script you should be able to run CPJ with TF>=2.14.

import numpy as np
import pickle
import sys

# specify the pickle filename in trained_models without .pkl extension,
# either by passing it as first argument or hardcoding it here
try:
    pkl_filename = sys.argv[1]
except:
    print('Pickle filename not passed as input, so falling back to file already in trained models')
    pkl_filename = 'tf_cmb_template' # this is one of the SPT models available from CP

with open(f'./cosmopower_jax/trained_models/{pkl_filename}.pkl', 'rb') as f:
      pickle_file = pickle.load(f)

variable_names = ['weights_', 'biases_', 'alphas_', 'betas_', \
                  'param_train_mean', 'param_train_std', \
                  'feature_train_mean', 'feature_train_std', \
                  'n_parameters', 'parameters', \
                  'n_modes', 'modes', \
                  'n_hidden', 'n_layers', 'architecture']

# create the new dictionary, and save it with the same name (but different extension)
new_dict = {name: value for name, value in zip(variable_names, pickle_file)}
np.savez(f'./cosmopower_jax/trained_models/{pkl_filename}', new_dict)
