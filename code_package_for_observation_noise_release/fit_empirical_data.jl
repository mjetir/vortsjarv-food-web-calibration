
# Maria Tirronen 2025
# https://orcid.org/0000-0001-6052-3186



module FitEmpiricalData

using Statistics, Random
using Dates
using DelimitedFiles

include("set_foodweb_parameters.jl")
include("foodweb_fitting_options.jl")
include("fit_foodweb_parameters_EO.jl")
include("plot_results_for_foodwebs.jl")
include("foodweb_model_discretized.jl")
include("foodweb_utils.jl")


# FIT THE MODEL TO THE VORTS DATA TIME SERIES
function fit_vorts_data(method,model_type,n_iter_restart, n_offspring,q_init)
    
    vorts_data, fw_init_vorts = SetFoodWebParameters.initialize_vorts_foodweb_ODE_disc(model_type,q_init)
        
    f_opt = FittingOptions.initialize_MLE_options_ODE_disc(fw_init_vorts,vorts_data,method,n_iter_restart,n_offspring)    

    vorts_results = FitFoodWebModel.fit_foodweb(fw_init_vorts,f_opt)   

    if(q_init == 0.0)
        q_fix = "0"
    elseif(q_init == 0.3)
        q_fix = "0_3"
    elseif(q_init == 0.5)
        q_fix = "0_5"
    elseif(q_init == 0.7)
        q_fix = "0_7"
    elseif(q_init == 1.0)
        q_fix = "1"
    else
        return 0 
    end

    if(cmp(model_type,"TypeIIIDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)  
                results_parent_folder = "results_vorts_BOUNDED_T_TLG24_1_extinction_penalty_const_separate_normal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"
                
    elseif(cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)  
                results_parent_folder = "results_vorts_BOUNDED_T_TLG24_1_extinction_penalty_const_fixed_q_"*q_fix*"_separate_normal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"
                                    
    elseif(cmp(model_type,"FixedTypeDetritalLoopRelativeNormalNoiseExtinctionPenaltyConst")==0)  
                results_parent_folder = "results_vorts_BOUNDED_T_TLG24_1_extinction_penalty_const_fixed_q_"*q_fix*"_relative_normal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"
                                        
    else         
        return 0
    end

    FoodWebUtils.write_initial_parameters_to_file(f_opt,results_parent_folder*"EO_initial_setting")
  
    FoodWebUtils.write_parameters_to_file(vorts_results,results_parent_folder*"EO_parameter_estimates")
        
    FoodWebUtils.write_losses_to_file(vorts_results,results_parent_folder*"EO_loss")

    FoodWebPlots.plot_losses(vorts_results,results_parent_folder*"EO_loss")

    return vorts_results, fw_init_vorts
end


end #module



### absolute normal observation noise
## "separate" refers to absolute

## qs fixed

#res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE_ODE_disc","FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst", 999, 200, 0.0);

#res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE_ODE_disc","FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst", 999, 200, 0.3);

#res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE_ODE_disc","FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst", 999, 200, 0.5);

#res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE_ODE_disc","FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst", 999, 200, 0.7);

#res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE_ODE_disc","FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst", 999, 200, 1.0);


## more iteration
#res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE_ODE_disc","FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst", 2999, 200, 0.3);


## qs estimated

# res, fw = 
# FitEmpiricalData.fit_vorts_data("MLE_ODE_disc","TypeIIIDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst", 999, 200, 0.3);




### relative normal observation noise
## qs fixed 

#res, fw = 
# FitEmpiricalData.fit_vorts_data("MLE_ODE_disc","FixedTypeDetritalLoopRelativeNormalNoiseExtinctionPenaltyConst", 999, 200, 0.3);





### to study the impact of
    ##  activity respiration: 
        ## implement a coefficient for activity respiration in the function bioenergetic_discretized_detrital_loop of module DiscreteFoodWebModel (file "foodweb_model_discretized.jl")
    ## different feeding matrices:
        ## implement them in the function initialize_vorts_foodweb_ODE_disc of module SetFoodWebParameters (file "set_foodweb_parameters.jl")

