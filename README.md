Calibration of a trophic network model for Lake Võrtsjärv under different noise assumptions, using maximum likelihood estimation and evolutionary optimization.


These codes use Julia version 1.7.0 and dependences from atn_vorts/Project.toml


Includes separate code packages for environmental and observation noise.

"code_package_for_environmental_noise_with_feeding_link_12_8_and_cannibalism_release" includes codes for fitting the bioenergetic model with ENVIRONMENTAL noise. 
The default setting includes cannibalism of perch and pike and white bream as a resource for eel.

"code_package_for_observation_noise" includes codes for fitting the bioenergetic model with OBSERVATION noise. 
The default setting does not include cannibalism of perch and pike nor white bream as a resource for eel.


Both of the code packages include the following scripts:

fit_empirical_data.jl : module for fitting the bioenergetic model with environmental (or observation) noise to the Võrts biomass time series

fit_foodweb_parameters.jl : module for optimization functions, called by fit_empirical_data.jl

foodweb_fitting_options.jl : module for setting options for parameter estimation

foodweb_loss_functions.jl : module for different loss functions used in parameter estimation

foodweb_model_discretized.jl : implements the underlying deterministic bioenergetic model, discretized 

foodweb_utils.jl : functions for performing transformations for parameters and writing initial settings, parameter estimates etc. to file

plot_results_for_foodwebs.jl : analyze and visualize the results 

set_foodweb_parameters.jl : set parameters, such as feeding links, for the Võrts food web model


In addition, the code package for environmental noise includes the following script, used only in visualization:

set_foodweb_parameters_ODE.jl : set parameters for the Võrts food web, for the observation  noise model


To study the impact of
    #  activity respiration: 
        include a coefficient for activity respiration inside the function bioenergetic_discretized_detrital_loop of module DiscreteFoodWebModel (file "foodweb_model_discretized.jl")
    # different feeding matrices:
        type the feeding links inside the function initialize_vorts_foodweb (or initialize_vorts_foodweb_ODE_disc) of module SetFoodWebParameters (file "set_foodweb_parameters.jl")
    # different thresholds for penalization of negative values:
        change the value of "extinction_penalty_threshold" in the function initialize_MLE_options of module FittingOptions (file "foodweb_fitting_options.jl)
