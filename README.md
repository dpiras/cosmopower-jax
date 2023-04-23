# CosmoPower-JAX

(We will add a logo soon!)

`CosmoPower-JAX` in an extension of the [CosmoPower](https://github.com/alessiospuriomancini/cosmopower) framework to emulate cosmological power spectra in a differentiable way. With `CosmoPower-JAX` you can efficiently run Hamiltonian Monte Carlo with hundreds of parameters (for example, nuisance parameters describing systematic effects), on CPUs and GPUs, in a fraction of the time which would be required with traditional methods. We provide some examples on how to use the neural emulators below, and more applications in our paper (coming soon).

Of course, with `CosmoPower-JAX` you can also obtain efficient and differentiable predictions of cosmological power spectra. We show how to achieve this in 3 lines of code below.

## Installation

To install `CosmoPower-JAX`, you can simply use `pip`:

        pip install cosmopower_jax

We recommend doing it in a fresh `conda` environment, to avoid clashes (e.g. `conda create -n cpj python=3.9 && conda activate cpj`). Alternatively, you can:

        git clone https://github.com/dpiras/cosmopower-jax.git
        cd cosmopower-jax
        pip install . 

The latter will also give you access to a Jupyter notebook with some examples.

## Usage & example

After the installation, getting a prediction is as simple as:

. In the meantime, if you want to use the emulators of the matter power spectrum or of the CMB temperature and polarisation spectra, you can do as follows (coming soon).


We currently do not provide the code to train a neural-network model in JAX; if you would like to re-train a JAX-based neural network on different data, [raise an issue](https://github.com/dpiras/cosmopower-jax/issues) or contact [Davide Piras](mailto:davide.piras@unige.ch).

## Contributing and contacts

Feel free to [fork](https://github.com/dpiras/cosmopower-jax/fork) this repository to work on it; otherwise, please [raise an issue](https://github.com/dpiras/cosmopower-jax/issues) or contact [Davide Piras](mailto:davide.piras@unige.ch).

## Citation
If you use `CosmoPower-JAX` in your work, please cite both papers as follows:

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
             
    @article{Piras23,
             title={CosmoPower-JAX: high-dimensional Bayesian inference with 
             differentiable cosmological emulators},
             volume={TBC},
             ISSN={TBC},
             url={TBC},
             DOI={TBC},
             number={TBC},
             journal={TBC},
             publisher={TBC},
             author={Piras, Davide and Spurio Mancini, Alessio},
             year={2023},
             month={TBC},
             pages={TBC}
             }

## License

`CosmoPower-JAX` is released under the GPL-3 license - see [LICENSE](https://github.com/dpiras/cosmopower-jax/blob/main/LICENSE.txt)-, subject to 
the non-commercial use condition - see [LICENSE_EXT](https://github.com/dpiras/cosmopower-jax/blob/main/LICENSE_EXT.txt).

     CosmoPower-JAX     
     Copyright (C) 2023 Davide Piras & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
