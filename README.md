# CosmoPower-JAX


<p align="center">
  <img src="https://user-images.githubusercontent.com/25639122/235351711-39be2b50-dbcb-4964-adbf-f38ffc74ef5f.jpeg" width="300" height="202.5"
 alt="CPJ_logo"/>
</p>

     
`CosmoPower-JAX` in an extension of the [CosmoPower](https://github.com/alessiospuriomancini/cosmopower) framework to emulate cosmological power spectra in a differentiable way. With `CosmoPower-JAX` you can efficiently run Hamiltonian Monte Carlo with hundreds of parameters (for example, nuisance parameters describing systematic effects), on CPUs and GPUs, in a fraction of the time which would be required with traditional methods. We provide some examples on how to use the neural emulators below, and more applications [in our paper](https://arxiv.org/abs/2305.06347).

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

We provide a full walkthrough in the accompanying [Jupyter notebook](https://github.com/dpiras/cosmopower-jax/blob/main/notebooks/emulators_example.ipynb), and we describe `CosmoPower-JAX` in detail in the release paper. We currently do not provide the code to train a neural-network model in JAX; if you would like to re-train a JAX-based neural network on different data, [raise an issue](https://github.com/dpiras/cosmopower-jax/issues) or contact [Davide Piras](mailto:davide.piras@unige.ch).

## Contributing and contacts

Feel free to [fork](https://github.com/dpiras/cosmopower-jax/fork) this repository to work on it; otherwise, please [raise an issue](https://github.com/dpiras/cosmopower-jax/issues) or contact [Davide Piras](mailto:davide.piras@unige.ch).

## Citation
If you use `CosmoPower-JAX` in your work, please cite both papers as follows:

    @article{Piras23,
             title={CosmoPower-JAX: high-dimensional Bayesian inference with
             differentiable cosmological emulators},
             author = {Piras, Davide and Spurio Mancini, Alessio},
             journal = {arXiv e-prints},
             year = 2023,
             month = may,
             eid = {arXiv:2305.06347},
             pages = {arXiv:2305.06347},
             archivePrefix = {arXiv},
             eprint = {2305.06347},
             adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230506347P},
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
