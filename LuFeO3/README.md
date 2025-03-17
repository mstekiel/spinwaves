# LuFeO3

`lufeo3.py`
Full example of magnon plotting, fitting and compating with measurements.

## Feature request

- [ ] Add filters to the data.
- [ ] Better diagnostic tools.

## Code structure

Loading of libraries. You should set up your virtual pyhon environment properly.

Definition of data. These are the results I got from Dnyaneshwar, then I load them into arrays.

Finally, there are many functions building the program:
 - `load_system` the engine of the calculation that creates a full `SpinW` model from the `parameters`. It defines the symmetry, lattice, nuclear and magnetic structure of the system, then creates the coupling scheme as we have developed for LuFeO3.
 - `plot_spectrum` makes the data vs simulation plots.
 - `lfo_residuals`, `fit_lfo` fitting of the model to the data.
 - `load_lfo_parameters` function calling the set of parameters we are fitting to the data.
 - `main` what is done when we run the script.


## How to

1. `main` function can quickly switch between fitting and plotting models. What I usually do is I build the starting model in the `load_lfo_parameters` and plot it. Once I have reasonable similarity between data and model I let it fit.
2. Building a model. You have to define all parameters in your model in `load_lfo_parameters`. Those parameters are instance of `lmfit.Parameters` and get the full functionality of the `lmfit` library. For example, to bound the `J2a` value to the same of `J2b` see line 497. There are many things you can do with `lmfit.Parameters`, see what I've done in the program or the [documentation](https://lmfit.github.io/lmfit-py/parameters.html)
3. The code is open for any tweak I can imagine. We should discuss how to proceed. Maybe the model itself can use update. 