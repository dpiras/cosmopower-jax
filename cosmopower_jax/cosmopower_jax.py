import pickle
import numpy as np
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import jacfwd, jacrev

# to deal with the pre-saved models and PCA attributes
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to python <3.7 `importlib_resources`.
    import importlib_resources as pkg_resources
from . import trained_models  # relative-import the *package* containing the templates
from . import pca_utils  # relative-import the *package* containing the templates


class CosmoPowerJAX:
    """Predict cosmological power spectra using pre-trained neural networks.
    All done in JAX. Can predict the linear and non-linear (boost) power spectrum, 
    as well as CMB probes.
    
    Parameters
    ----------
    probe : string
        The probe being considered to make predictions. 
        Must be one of (the names are hopefully self-explanatory):
        'cmb_tt', 'cmb_ee', 'cmb_te', 'cmb_pp', 'mpk_lin', 'mpk_boost', 'mpk_nonlin', custom_log', 'custom_pca'
    filename : string, default=None
        In case you want to restore from a custom file with the same pickle format
        as the provided ones, indicate the name to the .pkl file here.
        The .pkl file should be placed in the `cosmopower_jax/trained_models/` folder.
        You can specify either a .pkl file for a model trained on log-spectra ('custom_log'),
        or for a model trained with PCAplusNN ('custom_pca').
        This is generally to upload models trained with the original CP, 
        so you will also probably need to pip install tensorflow.
        If you are using `TF>=2.14`, make sure you run the `convert_tf214.py` script first
        and follow the instructions there.
    filepath: string, default=None
        If you do not specify a filename, you can specify the full path where to upload the 
        pre-trained model from. Note that you cannot specify both `filename` and `filepath`, 
        and that if you specify `filepath` you need to ensure that the format is the correct one.
        CPJ will try its best to load it depending on the format and the specified `probe`.
    """
    def __init__(self, probe, filename=None, filepath=None): 
        if probe not in ['cmb_tt', 'cmb_ee', 'cmb_te', 'cmb_pp', 'mpk_lin', 'mpk_boost', 'mpk_nonlin', 'custom_log', 'custom_pca']:
            raise ValueError(f"Probe not known. It should be one of "
                         f"'cmb_tt', 'cmb_ee', 'cmb_te', 'cmb_pp', 'mpk_lin', 'mpk_boost', 'mpk_nonlin', custom_log', 'custom_pca'; found '{probe}'") 
        
        if probe in ['cmb_tt', 'cmb_ee', 'mpk_lin', 'mpk_boost', 'mpk_nonlin', 'custom_log']:
            self.log = True
        else:
            self.log = False
            if probe == 'custom_pca':
                # we deal with this case below
                pass
            else:
                # prepare for PCA: load pre-trained PCA matrix, and mean and std dev of the training data
                pca_matrix_file = pkg_resources.open_binary(pca_utils, f'{probe}_pca_transform_matrix.npy')
                pca_matrix = np.load(pca_matrix_file)
                self.pca_matrix = pca_matrix

                training_mean_file = pkg_resources.open_binary(pca_utils, f'{probe}_training_mean.npy')
                training_mean = np.load(training_mean_file)
                self.training_mean = training_mean

                training_std_file = pkg_resources.open_binary(pca_utils, f'{probe}_training_std.npy')
                training_std = np.load(training_std_file)
                self.training_std = training_std

        if probe == 'mpk_nonlin':
            # here we need to combine the linear and non-linear one, so it is a big less elegant
            # Load pre-trained models of linear power spectrum and boost
            probe_file = pkg_resources.open_binary(trained_models, 'mpk_lin.pkl')
            self.weights_l, self.hyper_params_l, \
            self.param_train_mean_l, self.param_train_std_l, \
            self.feature_train_mean_l, self.feature_train_std_l, \
            n_parameters, parameters, \
            n_modes, modes_l, \
            n_hidden, n_layers, architecture = pickle.load(probe_file) 
            
            probe_file = pkg_resources.open_binary(trained_models, 'mpk_boost.pkl')
            weights, hyper_params, \
            param_train_mean, param_train_std, \
            feature_train_mean, feature_train_std, \
            n_parameters, parameters, \
            n_modes, modes, \
            n_hidden, n_layers, architecture = pickle.load(probe_file) 
        
        elif probe == 'custom_pca':
            try:
                if filename is not None:
                    assert filepath == None, f'Specified filename {filename}, but also filepath {filepath};' \
                                             f' to avoid ambiguities only specify one of them.'
                    probe_file = pkg_resources.open_binary(trained_models, filename)
                    # in this case hyperparams and weights/biases were loaded separately
                    # so we have to zip them
                    weights_, biases_, alphas_, betas_, \
                    param_train_mean, param_train_std, \
                    feature_train_mean, feature_train_std, \
                    self.training_mean, self.training_std, \
                    parameters, n_parameters, \
                    modes, n_modes, \
                    n_pcas, self.pca_matrix, \
                    n_hidden, n_layers, architecture = pickle.load(probe_file)
                elif filepath is not None:
                    with open(filepath, 'rb') as probe_file:
                        # in this case hyperparams and weights/biases were loaded separately
                        # so we have to zip them
                        weights_, biases_, alphas_, betas_, \
                        param_train_mean, param_train_std, \
                        feature_train_mean, feature_train_std, \
                        self.training_mean, self.training_std, \
                        parameters, n_parameters, \
                        modes, n_modes, \
                        n_pcas, self.pca_matrix, \
                        n_hidden, n_layers, architecture = pickle.load(probe_file)
                else:
                    raise ValueError('You specified `custom_pca` as the probe, but no `filename` or `filepath`.')
            except:
                # in this case, we fall back to the dictionary that is created
                # when running the convert_tf214.py script, available in the root folder
                print('Tried to load pickle file from pre-trained model, but failed.')
                print('This usually means that you have TF>=2.14, or that you are loading a model' \
                      ' that was trained on PCA but loaded with the log (or viceversa), or that' \
                      ' you are loading a non-standard model from the cosmopower-organization repo.')
                print('Falling back to the dictionary, if case this also fails or does not output the right shape' \
                      ' make sure you ran the `convert_tf214.py` script, and that a `.npz` file exists among' \
                      ' the trained models, and that you ran `pip install .`. Also make sure' \
                      ' that you are asking for the right probe between `custom_log` and `custom_pca`.')
                # the [:-4] should ensure we remove the .pkl suffix,
                # ensuring backward compatibility
                if filename is not None:
                    loaded_variable_dict = pkg_resources.open_binary(trained_models, f'{filename[:-4]}.npz')
                elif filepath is not None:
                    loaded_variable_dict = filepath
                else:
                    raise ValueError('You specified `custom_pca` as the probe, but no `filename` or `filepath`.')
                loaded_variable_dict = np.load(loaded_variable_dict, allow_pickle=True)
                if 'arr_0' in loaded_variable_dict:
                    loaded_variable_dict = loaded_variable_dict['arr_0'].tolist()
                # boring, but needed as the exec approach did not work here, and we need to assign some properties
                n_parameters = loaded_variable_dict['n_parameters']
                parameters = loaded_variable_dict['parameters']
                n_modes = loaded_variable_dict['n_modes']
                modes = loaded_variable_dict['modes']
                n_hidden = loaded_variable_dict['n_hidden']
                n_layers = loaded_variable_dict['n_layers']
                architecture = loaded_variable_dict['architecture']
                n_pcas = loaded_variable_dict['n_pcas']
                try: self.pca_matrix = loaded_variable_dict['pca_matrix']
                except: self.pca_matrix = loaded_variable_dict['pca_transform_matrix']
                if "weights_" in loaded_variable_dict:
                    # assign the list of weight arrays from 'weights_' directly
                    weights_ = loaded_variable_dict["weights_"]
                else:
                    # use individual weight arrays if available
                    weights_ = [loaded_variable_dict[f"W_{i}"] for i in range(n_layers)]
                # repeat for biases, alphas and betas
                if "biases_" in loaded_variable_dict:
                    biases_ = loaded_variable_dict["biases_"]
                else:
                    biases_ = [loaded_variable_dict[f"b_{i}"] for i in range(n_layers)]
                if "alphas_" in loaded_variable_dict:
                    alphas_ = loaded_variable_dict["alphas_"]
                else:
                    alphas_ = [loaded_variable_dict[f"alphas_{i}"] for i in range(n_layers-1)]
                if "betas_" in loaded_variable_dict:
                    betas_ = loaded_variable_dict["betas_"]
                else:
                    betas_ = [loaded_variable_dict[f"betas_{i}"] for i in range(n_layers-1)]  
                # attempt to load 'parameters_mean' or fall back to 'param_train_mean' (and analogous)
                try: param_train_mean = loaded_variable_dict['parameters_mean']
                except: param_train_mean = loaded_variable_dict['param_train_mean']
                try: param_train_std = loaded_variable_dict['parameters_std']
                except: param_train_std = loaded_variable_dict['param_train_std']
                # some shenanigans to make sure we load the correct things...
                if 'pca_mean' in loaded_variable_dict and 'feature_train_mean' in loaded_variable_dict:
                    # this should be Boris' case
                    self.training_mean = loaded_variable_dict['feature_train_mean']
                    self.training_std = loaded_variable_dict['feature_train_std']
                    feature_train_mean = loaded_variable_dict['pca_mean']
                    feature_train_std = loaded_variable_dict['pca_std']             
                elif 'pca_mean' in loaded_variable_dict and 'feature_train_mean' not in loaded_variable_dict:
                    # this should be Hidde's case
                    self.training_mean = loaded_variable_dict['features_mean']
                    self.training_std = loaded_variable_dict['features_std']
                    feature_train_mean = loaded_variable_dict['pca_mean']
                    feature_train_std = loaded_variable_dict['pca_std']  
                else:
                    # this is the typical CP file
                    self.training_mean = loaded_variable_dict['training_mean']
                    self.training_std = loaded_variable_dict['training_std']
                    feature_train_mean = loaded_variable_dict['feature_train_mean']
                    feature_train_std = loaded_variable_dict['feature_train_std']                      
            hyper_params = list(zip(alphas_, betas_))
            # we also have to transpose all weights, since in JAX we did differently
            weights_ = [w.T for w in weights_]
            weights = list(zip(weights_, biases_))
            
        else:
            # Load pre-trained model
            if probe == 'custom_log':
                # first we try the standard approach, which will fail if TF>=2.14
                # since TF removed support for pickle
                try:
                    if filename is not None:
                        assert filepath == None, f'Specified filename {filename}, but also filepath {filepath};' \
                                                 f' to avoid ambiguities only specify one of them.'
                        # in this case hyperparams and weights/biases were loaded separately
                        # so we have to zip them
                        probe_file = pkg_resources.open_binary(trained_models, filename)
                        weights_, biases_, alphas_, betas_, \
                        param_train_mean, param_train_std, \
                        feature_train_mean, feature_train_std, \
                        n_parameters, parameters, \
                        n_modes, modes, \
                        n_hidden, n_layers, architecture = pickle.load(probe_file)
                    elif filepath is not None:
                        with open(filepath, 'rb') as probe_file:
                            weights_, biases_, alphas_, betas_, \
                            param_train_mean, param_train_std, \
                            feature_train_mean, feature_train_std, \
                            n_parameters, parameters, \
                            n_modes, modes, \
                            n_hidden, n_layers, architecture = pickle.load(probe_file)
                    else:
                        raise ValueError('You specified `custom_log` as the probe, but no `filename` or `filepath`.')
                except:
                    # in this case, we fall back to the dictionary that is created
                    # when running the convert_tf214.py script, available in the root folder
                    print('Tried to load pickle file from pre-trained model, but failed.')
                    print('This usually means that you have TF>=2.14, or that you are loading a model' \
                          ' that was trained on PCA but loaded with the log (or viceversa), or that' \
                          ' you are loading a non-standard model from the cosmopower-organization repo.')
                    print('Falling back to the dictionary, if case this also fails or does not output the right shape' \
                          ' make sure you ran the `convert_tf214.py` script, and that a `.npz` file exists among' \
                          ' the trained models, and that you ran `pip install .`. Also make sure' \
                          ' that you are asking for the right probe between `custom_log` and `custom_pca`.')
                    # the [:-4] should ensure we remove the .pkl suffix, 
                    # ensuring backward compatibility
                    if filename is not None:
                        loaded_variable_dict = pkg_resources.open_binary(trained_models, f'{filename[:-4]}.npz')
                    elif filepath is not None:
                        loaded_variable_dict = filepath
                    else:
                        raise ValueError('You specified `custom_log` as the probe, but no `filename` or `filepath`.')
                    loaded_variable_dict = np.load(loaded_variable_dict, allow_pickle=True)
                    if 'arr_0' in loaded_variable_dict:
                        loaded_variable_dict = loaded_variable_dict['arr_0'].tolist()
                    # boring, but needed as the exec approach did not work here
                    n_parameters = loaded_variable_dict['n_parameters']
                    parameters = loaded_variable_dict['parameters']
                    n_modes = loaded_variable_dict['n_modes']
                    modes = loaded_variable_dict['modes']
                    n_hidden = loaded_variable_dict['n_hidden']
                    n_layers = loaded_variable_dict['n_layers']
                    architecture = loaded_variable_dict['architecture']
                    if "weights_" in loaded_variable_dict:
                        # assign the list of weight arrays from 'weights_' directly
                        weights_ = loaded_variable_dict["weights_"]
                    else:
                        # use individual weight arrays if available
                        weights_ = [loaded_variable_dict[f"W_{i}"] for i in range(n_layers)]
                    # repeat for biases, alphas and betas
                    if "biases_" in loaded_variable_dict:
                        biases_ = loaded_variable_dict["biases_"]
                    else:
                        biases_ = [loaded_variable_dict[f"b_{i}"] for i in range(n_layers)]
                    if "alphas_" in loaded_variable_dict:
                        alphas_ = loaded_variable_dict["alphas_"]
                    else:
                        alphas_ = [loaded_variable_dict[f"alphas_{i}"] for i in range(n_layers-1)]
                    if "betas_" in loaded_variable_dict:
                        betas_ = loaded_variable_dict["betas_"]
                    else:
                        betas_ = [loaded_variable_dict[f"betas_{i}"] for i in range(n_layers-1)]                        
                    # attempt to load 'parameters_mean' or fall back to 'param_train_mean' (and analogous)
                    try: param_train_mean = loaded_variable_dict['parameters_mean']
                    except: param_train_mean = loaded_variable_dict['param_train_mean']
                    try: param_train_std = loaded_variable_dict['parameters_std']
                    except: param_train_std = loaded_variable_dict['param_train_std']
                    try: feature_train_mean = loaded_variable_dict['features_mean']
                    except: feature_train_mean = loaded_variable_dict['feature_train_mean']
                    try: feature_train_std = loaded_variable_dict['features_std']
                    except: feature_train_std = loaded_variable_dict['feature_train_std']
                        
                hyper_params = list(zip(alphas_, betas_))
                # we also have to transpose all weights, since in JAX we did differently
                weights_ = [w.T for w in weights_]
                weights = list(zip(weights_, biases_))
            else:
                # most general case
                if filepath is None:
                    probe_file = pkg_resources.open_binary(trained_models, f'{probe}.pkl')
                    weights, hyper_params, \
                    param_train_mean, param_train_std, \
                    feature_train_mean, feature_train_std, \
                    n_parameters, parameters, \
                    n_modes, modes, \
                    n_hidden, n_layers, architecture = pickle.load(probe_file)
                else:
                    try:
                        with open(filepath, 'rb') as probe_file:
                            weights, hyper_params, \
                            param_train_mean, param_train_std, \
                            feature_train_mean, feature_train_std, \
                            n_parameters, parameters, \
                            n_modes, modes, \
                            n_hidden, n_layers, architecture = pickle.load(probe_file)
                    except:
                        # in this case, we fall back to the dictionary that is created
                        # when running the convert_tf214.py script, available in the root folder
                        print('Tried to load pickle file from pre-trained model, but failed.')
                        print('This usually means that you have TF>=2.14, or that you are loading a model' \
                              ' that was trained on PCA but loaded with the log (or viceversa), or that' \
                              ' you are loading a non-standard model from the cosmopower-organization repo.')
                        print('Falling back to the dictionary, if case this also fails or does not output the right shape' \
                              ' make sure you ran the `convert_tf214.py` script, and that a `.npz` file exists among' \
                              ' the trained models, and that you ran `pip install .`. Also make sure' \
                              ' that you are asking for the right probe between `custom_log` and `custom_pca`.')
                        # the [:-4] should ensure we remove the .pkl suffix,
                        # ensuring backward compatibility
                        loaded_variable_dict = np.load(f'{filepath[:-4]}.npz', allow_pickle=True)
                        if 'arr_0' in loaded_variable_dict:
                            loaded_variable_dict = loaded_variable_dict['arr_0'].tolist()

                        n_parameters = loaded_variable_dict['n_parameters']
                        parameters = loaded_variable_dict['parameters']
                        n_modes = loaded_variable_dict['n_modes']
                        modes = loaded_variable_dict['modes']
                        n_hidden = loaded_variable_dict['n_hidden']
                        n_layers = loaded_variable_dict['n_layers']
                        architecture = loaded_variable_dict['architecture']
                        # in this case weights, biases, alphas and betas should be already all set up
                        weights = loaded_variable_dict['weights']
                        hyper_params = loaded_variable_dict['hyper_params']
                        # and although this should not be needed, let's be conservative here
                        try: param_train_mean = loaded_variable_dict['parameters_mean']
                        except: param_train_mean = loaded_variable_dict['param_train_mean']
                        try: param_train_std = loaded_variable_dict['parameters_std']
                        except: param_train_std = loaded_variable_dict['param_train_std']
                        try: feature_train_mean = loaded_variable_dict['features_mean']
                        except: feature_train_mean = loaded_variable_dict['feature_train_mean']
                        try: feature_train_std = loaded_variable_dict['features_std']
                        except: feature_train_std = loaded_variable_dict['feature_train_std']
                    else:
                        raise ValueError(f'You specified a custom probe {probe}, but no filepath,' \
                                          ' or the filepath you specified is not a pickle file or' \
                                          ' a npz dictionary.')

       
        # save useful attributes  
        self.probe = probe
        self.weights = weights
        self.hyper_params = hyper_params
        self.param_train_mean = param_train_mean
        self.param_train_std = param_train_std
        self.feature_train_mean = feature_train_mean
        self.feature_train_std = feature_train_std
        self.n_parameters = n_parameters
        self.parameters = parameters
        if probe in ['cmb_pp', 'cmb_te']:
            # in this case, the modes are the PCA ones, so we have to replace them
            self.modes = np.arange(2, 2509)
        else:
            self.modes = modes

    def _dict_to_ordered_arr_np(self,
                               input_dict,
                               ):
        """
        Sort input parameters. Takend verbatim from CP 
        (https://github.com/alessiospuriomancini/cosmopower/blob/main/cosmopower/cosmopower_NN.py#LL291C1-L308C73)

        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted

        Returns:
            numpy.ndarray:
                parameters sorted according to desired order
        """
        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)

    def _activation(self, x, a, b):
        """Non-linear activation function. Based on the original CosmoPower paper, Eq. A1.
        x is the input of each layer, while a and b are the trainable hyper-parameters. 
        These correspond to beta and gamma in Eq. A1 in https://arxiv.org/pdf/2106.03846v2.pdf.
        """
        return jnp.multiply(jnp.add(b, jnp.multiply(sigmoid(jnp.multiply(a, x)), jnp.subtract(1., b))), x)

    def _predict(self, weights, hyper_params, param_train_mean, param_train_std,
                 feature_train_mean, feature_train_std, input_vec):
        """ Forward pass through pre-trained network.
        In its current form, it does not make use of high-level frameworks like
        FLAX et similia; rather, it simply loops over the network layers.
        In future work this can be improved, especially if speed is a problem.
        
        Parameters
        ----------
        weights : array
            The stored weights of the neural network.
        hyper_params : array
            The stored hyperparameters of the activation function for each layer.
        param_train_mean : array
            The stored mean of the training cosmological parameters.
        param_train_std : array
            The stored standard deviation of the training cosmological parameters.
        feature_train_mean : array
            The stored mean of the training features.
        feature_train_std : array
            The stored  standard deviation of the training features.
        input_vec : array of shape (n_samples, n_parameters) or (n_parameters)
            The cosmological parameters given as input to the network.
            
        Returns
        -------
        predictions : array
            The prediction of the trained neural network.
        """        
        act = []
        # Standardise
        layer_out = [(input_vec - param_train_mean)/param_train_std]

        # Loop over layers
        for i in range(len(weights[:-1])):
            w, b = weights[i]
            alpha, beta = hyper_params[i]
            act.append(jnp.dot(layer_out[-1], w.T) + b)
            layer_out.append(self._activation(act[-1], alpha, beta))

        # Final layer prediction (no activations)
        w, b = weights[-1]
        if self.probe == 'custom_log' or self.probe == 'custom_pca':
            # in original CP models, we assumed a full final bias vector...
            preds = jnp.dot(layer_out[-1], w.T) + b
        else:   
            # ... unlike in cpjax, where we used only a single bias vector
            preds = jnp.dot(layer_out[-1], w.T) + b[-1]

        # Undo the standardisation
        preds = preds * feature_train_std + feature_train_mean
        if self.log == True:
            preds = 10**preds
        else:
            preds = (preds@self.pca_matrix)*self.training_std + self.training_mean
            if self.probe == 'cmb_pp':
                preds = 10**preds
        predictions = preds.squeeze()
        return predictions
    
    def predict(self, input_vec):
        """ Emulate cosmological power spectrum, based on the probe specified as input.
        Need to provide in input the array (or the dictionary) of cosmological parameters.
        If input is a dictionary, we to convert it to an array internally.
        
        Parameters
        ----------
        input_vec : array of shape (n_samples, n_parameters) or (n_parameters); else, dict
            The cosmological parameters given as input to the network.
            The order has to be:
            (for CMB) omega_b, omega_cdm, h, tau, n_s, ln10^10A_s
            (for mPk) omega_b, omega_cdm, h, n_s, ln10^{10}A_s, (c_min, eta0), z 
            Alternatively, a dictionary can be passed, and we take care of the conversion internally.
            
        Returns
        -------
        predictions : array
            The cosmological power spectrum as required by input probe.
        """
        # convert dict to array, if needed
        if isinstance(input_vec, dict):
            input_vec = self._dict_to_ordered_arr_np(input_vec)  
   
        if len(input_vec.shape) == 1:
            input_vec = input_vec.reshape(-1, self.n_parameters)
        assert len(input_vec.shape) == 2
        predictions = self._predict(self.weights, self.hyper_params, self.param_train_mean, 
                                    self.param_train_std, self.feature_train_mean, self.feature_train_std,
                                    input_vec)
        if self.probe == 'mpk_nonlin':
            # multiply by linear power spectrum
            input_vec = jnp.concatenate((input_vec[:, :5],input_vec[:, 7:8]), axis=1)  
            predictions *= self._predict(self.weights_l, self.hyper_params_l, self.param_train_mean_l, 
                                         self.param_train_std_l, self.feature_train_mean_l, self.feature_train_std_l,
                                         input_vec
                                         )
        return predictions
    
    def derivative(self, input_vec, mode='forward'):
        """ Derivative of the cosmological power spectra with respect to cosmological parameters.
        All done automatically by JAX with autodiff.
        
        Parameters
        ----------
        input_vec : array of shape (n_samples, n_parameters) or (n_parameters)
            The cosmological parameters at which to compute the derivatives.
        mode : string, default='forward'
            The differentiation mode. It must be either 'forward' or 'reverse', with the former
            being a bit faster. The answers should be in agreement within machine precision.
            A detailed discussion is available here: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html.
       
        Returns
        -------
        derivatives : array
            The derivatives of the cosmological power spectrum with respect to the input cosmological parameters.
        """
        # convert dict to array, if needed
        if isinstance(input_vec, dict):
            input_vec = self._dict_to_ordered_arr_np(input_vec)  
            
        if len(input_vec.shape) == 1:
            input_vec = input_vec.reshape(1, self.n_parameters)
        assert len(input_vec.shape) == 2
                
        if mode == 'forward':
            # shape trick, to return what we actually care about
            if input_vec.shape[0] == 1:
                derivatives = np.swapaxes(jacfwd(self.predict)(input_vec), 1, 2)
            else:
                derivatives = np.diagonal(np.swapaxes(jacfwd(self.predict)(input_vec), 1, 2))
        elif mode == 'reverse':
            if input_vec.shape[0] == 1:
                derivatives = np.swapaxes(jacfwd(self.predict)(input_vec), 1, 2)
            else:
                derivatives = np.diagonal(np.swapaxes(jacrev(self.predict)(input_vec), 1, 2))            
        else:  
            raise ValueError(f"Differentiation mode not known. It should be either "
                         f"'forward' or 'reverse'; found '{mode}'")         
            
        return derivatives

