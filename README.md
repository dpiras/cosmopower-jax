# CosmoPower-JAX


<p align="center">
  <img src="https://user-images.githubusercontent.com/25639122/235351711-39be2b50-dbcb-4964-adbf-f38ffc74ef5f.jpeg" width="300" height="202.5"
 alt="CPJ_logo"/>
</p>
<div align="center">
  
![](https://img.shields.io/badge/Python-181717?style=plastic&logo=python)
![](https://img.shields.io/badge/Author-Davide%20Piras%20-181717?style=plastic)
![](https://img.shields.io/badge/Installation-pip%20install%20cosmopower--jax-181717?style=plastic)
[![arXiv](https://img.shields.io/badge/arXiv-2305.06347-b31b1b.svg)](https://arxiv.org/abs/2305.06347)


</div>

     
`CosmoPower-JAX` in an extension of the [CosmoPower](https://github.com/alessiospuriomancini/cosmopower) framework to emulate cosmological power spectra in a differentiable way. With `CosmoPower-JAX` you can efficiently run Hamiltonian Monte Carlo with hundreds of parameters (for example, nuisance parameters describing systematic effects), on CPUs and GPUs, in a fraction of the time which would be required with traditional methods. We provide some examples on how to use the neural emulators below, and more applications [in our paper](https://arxiv.org/abs/2305.06347). You can also have a look at [our poster](https://github.com/dpiras/dpiras.github.io/blob/master/assets/images/poster_CPJ.pdf) presented at the [ML-IAP/CCA-2023](https://indico.iap.fr/event/1/overview) conference, which includes a video on how to use `CosmoPower-JAX`.

Of course, with `CosmoPower-JAX` you can also obtain efficient and differentiable predictions of cosmological power spectra. We show how to achieve this in less than 5 lines of code below.

## Installation

To install `CosmoPower-JAX`, you can simply use `pip`:

    pip install cosmopower-jax

We recommend doing it in a fresh `conda` environment, to avoid clashes (e.g. `conda create -n cpj python=3.9 && conda activate cpj`). 

Alternatively, you can:

    git clone https://github.com/dpiras/cosmopower-jax.git
    cd cosmopower-jax
    pip install . 

The latter will also give you access to a Jupyter notebook with some examples.

## Usage & example

After the installation, getting a cosmological power spectrum prediction is as simple as (e.g. for the CMB temperature power spectrum):

    import numpy as np
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
    # omega_b, omega_cdm, h, tau, n_s, ln10^10A_s
    cosmo_params = np.array([0.025, 0.11, 0.68, 0.1, 0.97, 3.1])
    emulator = CPJ(probe='cmb_tt')
    emulator_predictions = emulator.predict(cosmo_params)

Similarly, we can also compute derivatives like:

    emulator_derivatives = emulator.derivative(cosmo_params)

Rather than passing an array, as in the original `CosmoPower` syntax you can also pass a dictionary:

    cosmo_params = {'omega_b': [0.025],
                    'omega_cdm': [0.11],
                    'h': [0.68],
                    'tau_reio': [0.1],
                    'n_s': [0.97],
                    'ln10^{10}A_s': [3.1],
                    }
    emulator = CPJ(probe='cmb_tt')
    emulator_predictions = emulator.predict(cosmo_params)

We also support reusing original `CosmoPower` models, which you can now use in JAX without retraining. In that case, you should: 

```
   git clone https://github.com/dpiras/cosmopower-jax.git
   cd cosmopower-jax
```

and move your model(s) `.pkl` files into the folder `cosmopower_jax/trained_models`. At this point:

- if you can call your models from the `cosmopower-jax` folder you are in, you should be good to go;
- otherwise, run first `pip install .`, and then you should be able to call your custom models from anywhere.
 
To finally call a custom model, you can run:

```
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
emulator_custom = CPJ(probe='custom_log', filename='<custom_filename>.pkl')
```

where `<custom_filename>.pkl` is the filename (only, no path) with your custom model, and `custom_log` indicates that your model was trained on log-spectra, so all predictions will be returned elevated to the power of 10. Alternatively, you can pass `custom_pca`, and you will automatically get the predictions for a model trained with `PCAplusNN`. In this case the parameter dictionary should of course contain the parameter keys corresponding to your trained model. We also allow the full `filepath` of the trained model to be indicated: in this case, do not specify `filename` and only indicate the full `filepath` including the suffix.

We provide a full walkthrough and all instructions in the accompanying [Jupyter notebook](https://github.com/dpiras/cosmopower-jax/blob/main/notebooks/emulators_example.ipynb), and we describe `CosmoPower-JAX` in detail in the release paper. We currently do not provide the code to train a neural-network model in JAX; if you would like to re-train a JAX-based neural network on different data, [raise an issue](https://github.com/dpiras/cosmopower-jax/issues) or contact [Davide Piras](mailto:davide.piras@unige.ch).

### Note if you are using `TensorFlow>=2.14`
If you are reusing a model trained with `CosmoPower` and have a `TensorFlow` version higher or equal to 2.14, you might get an error when trying to load the model, even in `CosmoPower-JAX`. This is [a known issue](https://github.com/alessiospuriomancini/cosmopower/issues/22). In this case, you should run the `convert_tf214.py` script available in this repository to transform your `.pkl` file into a different format (based on `NumPy`) that will then be read by `CosmoPower-JAX`. You only have to do the conversion once for each `.pkl` file you have, make sure you `pip install .` after the conversion, and everything else should remain unchanged.


## Contributing and contacts

Feel free to [fork](https://github.com/dpiras/cosmopower-jax/fork) this repository to work on it; otherwise, please [raise an issue](https://github.com/dpiras/cosmopower-jax/issues) or contact [Davide Piras](mailto:davide.piras@unige.ch).

## Citation
If you use `CosmoPower-JAX` in your work, please cite both papers as follows:

    @article{Piras23,
             author = {{Piras}, Davide and {Spurio Mancini}, Alessio},
             title = "{CosmoPower-JAX: high-dimensional Bayesian inference 
             with differentiable cosmological emulators}",
             journal = {The Open Journal of Astrophysics},
             keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, 
             Astrophysics - Instrumentation and Methods for Astrophysics, 
             Computer Science - Machine Learning},
             year = 2023,
             month = jul,
             volume = {6},
             eid = {20},
             pages = {20},
             doi = {10.21105/astro.2305.06347},
             archivePrefix = {arXiv},
             eprint = {2305.06347},
             primaryClass = {astro-ph.CO}
             }
    
    @article{SpurioMancini2022,
             title={CosmoPower: emulating cosmological power spectra for 
             accelerated Bayesian inference from next-generation surveys},
             volume={511},
             ISSN={1365-2966},
             url={http://dx.doi.org/10.1093/mnras/stac064},
             DOI={10.1093/mnras/stac064},
             number={2},
             journal={Monthly Notices of the Royal Astronomical Society},
             publisher={Oxford University Press (OUP)},
             author={Spurio Mancini, Alessio and Piras, Davide and 
             Alsing, Justin and Joachimi, Benjamin and Hobson, Michael P},
             year={2022},
             month={Jan},
             pages={1771–1788}
             }
             

## License

`CosmoPower-JAX` is released under the GPL-3 license - see [LICENSE](https://github.com/dpiras/cosmopower-jax/blob/main/LICENSE)-, subject to 
the non-commercial use condition - see [LICENSE_EXT](https://github.com/dpiras/cosmopower-jax/blob/main/LICENSE_EXT).

     CosmoPower-JAX     
     Copyright (C) 2023 Davide Piras & contributors

     This program is released under the GPL-3 license (see LICENSE), 
     subject to a non-commercial use condition (see LICENSE_EXT).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
