import numpy as np
import jax.numpy as jnp
from jax.nn import sigmoid

class CosmoPowerJAX:
    """Predict cosmological power spectra using pre-trained neural networks.
    All done in JAX. Can predict the linear and non-linear boost power spectrum, 
    as well as CMB probes.
    
    Parameters
    ----------
    
    Attributes
    ----------

    """
    def __init__(self, probe): 
        if probe not in ['cmb_tt', 'cmb_ee', 'cmb_te', 'cmb_pp', 'mpk_lin', 'mpk_boost']:
            raise ValueError(f"Probe not known. It should be one of "
                         f"'cmb_tt', 'cmb_ee', 'cmb_te', 'cmb_pp', 'mpk_lin', 'mpk_boost'; found '{probe}'") 
        
        if probe in ['cmb_tt', 'cmb_ee', 'mpk_lin', 'mpk_boost']:
            self.log == True            
            
        # Load pre-trained model
        with open(f"./trained_models/{probe}.pkl", 'rb') as f:
            weights, hyper_params, \
            param_train_mean, param_train_std, \
            feature_train_mean, feature_train_std, \
            n_parameters, parameters, \
            n_modes, modes, \
            n_hidden, n_layers, architecture = pickle.load(f)
            
        self.weights = weights
        self.hyper_params = hyper_params
        self.param_train_mean = param_train_mean
        self.param_train_std = param_train_std
        self.feature_train_mean = feature_train_mean
        self.feature_train_std = feature_train_std
        self.modes = modes

               
    def _activation(self, x, a, b):
        """Non-linear activation function.
        Based on the original CosmoPower paper.
        """
        return jnp.multiply(jnp.add(b, jnp.multiply(sigmoid(jnp.multiply(a, x)), jnp.subtract(1., b))), x)

    def predict(self, input_vec):
        """ Forward pass through pre-trained network.
        In its current form, it does not make use of high-level frameworks like
        FLAX et similia; rather, it simply loops over the network layers.
        In future work this can be improved, especially if speed is a problem.
        """
        act = []
        # Standardise
        layer_out = [(input_vec - self.param_train_mean)/self.param_train_std]

        # Loop over layers
        for i in range(len(self.weights[:-1])):
            w, b = self.weights[i]
            alpha, beta = self.hyper_params[i]
            act.append(jnp.dot(layer_out[-1], w.T) + b)
            layer_out.append(self._activation(act[-1], alpha, beta))

        # Final layer prediction (no activations)
        w, b = self.weights[-1]
        preds = jnp.dot(layer_out[-1], w.T) + b[-1]

        # Undo the standardisation
        preds = preds * self.feature_train_std + self.feature_train_mean
        if self.log == True:
            preds = 10**preds
        else
            pass # do PCA
        return preds.squeeze()
