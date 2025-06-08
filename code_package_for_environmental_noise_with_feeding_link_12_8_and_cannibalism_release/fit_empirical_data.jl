
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


# FIT THE MODEL TO THE VORTS TIME SERIES
function fit_vorts_data(method,model_type,n_iter_restart, n_offspring,q_init)

    vorts_data, fw_init_vorts = SetFoodWebParameters.initialize_vorts_foodweb(model_type,q_init)

    f_opt = FittingOptions.initialize_MLE_options(fw_init_vorts,vorts_data,method,n_iter_restart,n_offspring)

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

    if(cmp(model_type,"TypeIIIDetritalLoopSeparateNormalNoise")==0)  
        results_parent_folder = "results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_separate_normal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_discrete_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"
        
    elseif(cmp(model_type,"TypeIIIDetritalLoopRelativeNormalNoise")==0)  
            results_parent_folder = "results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_relative_normal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_discrete_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"
            
    elseif(cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoise")==0)  
                results_parent_folder = "results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_fixed_q_"*q_fix*"_separate_normal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_discrete_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"
                    
    elseif(cmp(model_type,"FixedTypeDetritalLoopRelativeNormalNoise")==0)  
                results_parent_folder = "results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_fixed_q_"*q_fix*"_relative_normal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_discrete_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"
                
    else         
        return 0
    end

    FoodWebUtils.write_initial_parameters_to_file(f_opt,results_parent_folder*"EO_initial_setting")
  
    FoodWebUtils.write_parameters_to_file(vorts_results,results_parent_folder*"EO_parameter_estimates")
        
    FoodWebUtils.write_losses_to_file(vorts_results,results_parent_folder*"EO_loss")

    FoodWebPlots.plot_losses(vorts_results,results_parent_folder*"EO_loss")

    return vorts_results, fw_init_vorts
end


# FIT THE MODEL TO THE VORTS TIME SERIES, CONTINUE FROM PREVIOUSLY OBTAINED VALUES
# read the initial parameter values from a file 
function fit_vorts_data_from_last_estimates(method,model_type,n_iter_restart, n_offspring,q_init,results_from)

    vorts_data, fw_init_vorts = SetFoodWebParameters.initialize_vorts_foodweb(model_type,q_init)

    # read the best parameter estimates so far from a file and set them for the initial food web
    param_est = 0
    try
            # try to read the untransformed (in R^n) parameter estimates
            param_est = readdlm(results_from*"/EO_parameter_estimates/parameter_estimates.txt",Float64)
    catch
            println("No appropriate data files in the folder "*results_from*"/EO_parameter_estimates.")
            return 0
    end
        
    if(             cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyInf")==0 || 
                    cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0 || 
                    cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConstQuant")==0 )
    
                    transformed_param = FoodWebUtils.rho_transform(param_est)
                                
                    fw_init_vorts.r = transformed_param[1:fw_init_vorts.nGuilds - 1]
        
                    fw_init_vorts.J[fw_init_vorts.I_ind] = transformed_param[fw_init_vorts.nGuilds:fw_init_vorts.nGuilds + fw_init_vorts.nLinks - 1]
                    fw_init_vorts.B0[fw_init_vorts.I_ind] = transformed_param[fw_init_vorts.nGuilds + fw_init_vorts.nLinks:fw_init_vorts.nGuilds + 2*fw_init_vorts.nLinks - 1]
    
                    fw_init_vorts.std = transformed_param[fw_init_vorts.nGuilds + 2*fw_init_vorts.nLinks:2*fw_init_vorts.nGuilds + 2*fw_init_vorts.nLinks - 1]
        
                    fw_init_vorts.K = transformed_param[end]                        

    elseif(cmp(model_type,"FixedTypeDetritalLoopSeparateLognormalNoise")==0 ||
           cmp(model_type,"FixedTypeDetritalLoopRelativeLognormalNoise")==0 )
    
                    transformed_param = FoodWebUtils.rho_transform(param_est)
                                
                    fw_init_vorts.r = transformed_param[1:fw_init_vorts.nGuilds - 1]
        
                    fw_init_vorts.J[fw_init_vorts.I_ind] = transformed_param[fw_init_vorts.nGuilds:fw_init_vorts.nGuilds + fw_init_vorts.nLinks - 1]
                    fw_init_vorts.B0[fw_init_vorts.I_ind] = transformed_param[fw_init_vorts.nGuilds + fw_init_vorts.nLinks:fw_init_vorts.nGuilds + 2*fw_init_vorts.nLinks - 1]
                
                    fw_init_vorts.K = transformed_param[end]                        

    else 
            println("The given model not defined.")
    end
    

    f_opt = FittingOptions.initialize_MLE_options(fw_init_vorts,vorts_data,method,n_iter_restart,n_offspring)

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


    if(cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyInf")==0)  
                results_parent_folder = "results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_extinction_penalty_inf_using_last_estimates_fixed_q_"*q_fix*"_separate_normal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_discrete_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"

    elseif(cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)  
                results_parent_folder = "results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_extinction_penalty_const_using_last_estimates_fixed_q_"*q_fix*"_separate_normal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_discrete_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"

    elseif(cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConstQuant")==0)  
                results_parent_folder = "results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_extinction_penalty_const_quant_using_last_estimates_fixed_q_"*q_fix*"_separate_normal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_discrete_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"
                
    elseif(cmp(model_type,"FixedTypeDetritalLoopSeparateLognormalNoise")==0)  
                results_parent_folder = "results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_using_last_estimates_fixed_q_"*q_fix*"_separate_lognormal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_discrete_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"
                        
    elseif(cmp(model_type,"FixedTypeDetritalLoopRelativeLognormalNoise")==0)  
                results_parent_folder = "results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_using_last_estimates_fixed_q_"*q_fix*"_relative_lognormal_noise_"*string(n_offspring)*"_offspring_"*string(n_iter_restart+1)*"_iterations_"*method*"_discrete_BIOEN_DETRITAL_LOOP_CATCHES_"*string(Dates.now())*"/"
                                    
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



### absolute normal environmental noise
## "separate" refers to absolute

## qs fixed

# res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE","FixedTypeDetritalLoopSeparateNormalNoise", 999, 200, 0.0);

#res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE","FixedTypeDetritalLoopSeparateNormalNoise", 999, 200, 0.3);

#res, fw = 
# FitEmpiricalData.fit_vorts_data("MLE","FixedTypeDetritalLoopSeparateNormalNoise", 999, 200, 0.5);

# res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE","FixedTypeDetritalLoopSeparateNormalNoise", 999, 200, 0.7);

# res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE","FixedTypeDetritalLoopSeparateNormalNoise", 999, 200, 1.0);


## estimate qs

#res, fw = 
#   FitEmpiricalData.fit_vorts_data("MLE","TypeIIIDetritalLoopSeparateNormalNoise", 999, 200, 0.3);




### relative normal environmental noise 

## qs fixed

#res, fw = 
# FitEmpiricalData.fit_vorts_data("MLE","FixedTypeDetritalLoopRelativeNormalNoise", 999, 200, 0.3);


## estimate qs

# res, fw = 
# FitEmpiricalData.fit_vorts_data("MLE","TypeIIIDetritalLoopRelativeNormalNoise", 999, 200, 0.3);




### continue parameter estimation from previously obtained values

## penalization of negative values

#res, fw = 
# FitEmpiricalData.fit_vorts_data_from_last_estimates("MLE","FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst", 1999, 200, 0.3,
#    "results_vorts_environmental_noise/results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_fixed_q_0_3_separate_normal_noise_200_offspring_1000_iterations_MLE_discrete_BIOEN_DETRITAL_LOOP_CATCHES");

#res, fw = 
# FitEmpiricalData.fit_vorts_data_from_last_estimates("MLE","FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConstQuant", 1999, 200, 0.3,
#    "results_vorts_environmental_noise/results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_fixed_q_0_3_separate_normal_noise_200_offspring_1000_iterations_MLE_discrete_BIOEN_DETRITAL_LOOP_CATCHES");




## lognormal noise

#res, fw = 
#    FitEmpiricalData.fit_vorts_data_from_last_estimates("MLE","FixedTypeDetritalLoopSeparateLognormalNoise", 999, 200, 0.3,
#        "results_vorts_environmental_noise/results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_extinction_penalty_const_using_last_estimates_fixed_q_0_3_separate_normal_noise_200_offspring_2000_iterations_MLE_discrete_BIOEN_DETRITAL_LOOP_CATCHES");

#res, fw = 
#    FitEmpiricalData.fit_vorts_data_from_last_estimates("MLE","FixedTypeDetritalLoopRelativeLognormalNoise", 999, 200, 0.3,
#       "results_vorts_environmental_noise/results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_extinction_penalty_const_using_last_estimates_fixed_q_0_3_separate_normal_noise_200_offspring_2000_iterations_MLE_discrete_BIOEN_DETRITAL_LOOP_CATCHES");
   



### to study the impact of
    ##  activity respiration: 
        ## implement a coefficient for activity respiration in the function bioenergetic_discretized_detrital_loop of module DiscreteFoodWebModel (file "foodweb_model_discretized.jl")
    ## different feeding matrices:
        ## implement them in the function initialize_vorts_foodweb of module SetFoodWebParameters (file "set_foodweb_parameters.jl")
    ## different thresholds for penalization of negative values:
        ## change the value of "extinction_penalty_threshold" in the function initialize_MLE_options of module FittingOptions (file "foodweb_fitting_options.jl)

