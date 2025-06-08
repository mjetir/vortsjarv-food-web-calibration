
# Maria Tirronen 2025
# https://orcid.org/0000-0001-6052-3186


module FittingOptions


using LinearAlgebra
using Random, Distributions
using Statistics, Missings

include("foodweb_model_discretized.jl")
include("foodweb_loss_functions.jl")
include("foodweb_utils.jl")

# STRUCT FOR FITTING OPTIONS
# nPar: the number of parameters to be fitted
# p_init: initial values for the parameters (untransformed, in R^n)
# q_init: initial or set value of q
# param_min, param_max: the lower and upper bounds of transformed parameters
# bc_min, bc_max: the lower and upper bounds of untransformed parameters
# model: function of the model
# model_type: name or type of the model
# loss_function: function of loss
# extinction_penalty_threshold: used for penalization based on quantiles
# training_data: recorded time series
# n_iter_restarts: number of restarts in optimization
# n_offspring: the number of offspring used for evolutionary optimization
# method: here MLE
struct MLEOptions
    nPar::Int64 
    p_init::Vector{Float64}
    q_init::Float64 
    param_min::Vector{Float64}
    param_max::Vector{Float64}
    bc_min::Vector{Float64}
    bc_max::Vector{Float64}
    model
    model_type
    loss_function
    extinction_penalty_threshold::Float64
    training_data
    n_iter_restart::Int64 
    n_offspring::Int64 
    method::String
end


# INITIALIZE AN MLEOPTIONS STRUCT
function initialize_MLE_options(fw_init,training_data,method,n_iter_restart,n_offspring;kwargs...)
    
    model = DiscreteFoodWebModel.bioenergetic_discretized_detrital_loop

    if(cmp(fw_init.model_type,"TypeIIIDetritalLoopRelativeNormalNoise")==0 ||
        cmp(fw_init.model_type,"TypeIIIDetritalLoopSeparateNormalNoise")==0 )

        nPar = 3*fw_init.nLinks + 2*fw_init.nGuilds 

        var_ini = FoodWebUtils.inverse_rho_transform(vcat(fw_init.r,
        fw_init.J[fw_init.I_ind],
        fw_init.B0[fw_init.I_ind],fw_init.q[fw_init.I_ind],
        fw_init.cv,fw_init.K))

    elseif(cmp(fw_init.model_type,"FixedTypeDetritalLoopRelativeNormalNoise")==0 ||
        cmp(fw_init.model_type,"FixedTypeDetritalLoopSeparateNormalNoise")==0 || 
        cmp(fw_init.model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0 || 
        cmp(fw_init.model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConstQuant")==0 || 
        cmp(fw_init.model_type,"FixedTypeDetritalLoopRelativeLognormalNoise")==0 || 
        cmp(fw_init.model_type,"FixedTypeDetritalLoopSeparateLognormalNoise")==0)

        nPar = 2*fw_init.nLinks + 2*fw_init.nGuilds 

        var_ini = FoodWebUtils.inverse_rho_transform(vcat(fw_init.r,
        fw_init.J[fw_init.I_ind],
        fw_init.B0[fw_init.I_ind],
        fw_init.cv,fw_init.K))

    else 
        println("The given model not defined.")
        return 0
    end 

    if(length(var_ini)!=nPar)
            return 0
    end
        
    # initialize:
    param_min = zeros(nPar)        
    param_max = Inf.*ones(nPar)
    
    bc_min = copy(param_min)
    bc_max = copy(param_max)

    bc_min .= -Inf
    bc_max .= Inf

    if(cmp(fw_init.model_type,"TypeIIIDetritalLoopRelativeNormalNoise")==0)

        loss_function = LossFunctions.loss_MLE_TypeIII_detrital_loop_carrying_capacity_relative_normal_noise
    
    elseif(cmp(fw_init.model_type,"TypeIIIDetritalLoopSeparateNormalNoise")==0)
    
        loss_function = LossFunctions.loss_MLE_TypeIII_detrital_loop_carrying_capacity_separate_normal_noise
        
    elseif(cmp(fw_init.model_type,"FixedTypeDetritalLoopRelativeNormalNoise")==0)

        loss_function = LossFunctions.loss_MLE_FixedType_detrital_loop_carrying_capacity_relative_normal_noise
        
    elseif(cmp(fw_init.model_type,"FixedTypeDetritalLoopSeparateNormalNoise")==0)
        
        loss_function = LossFunctions.loss_MLE_FixedType_detrital_loop_carrying_capacity_separate_normal_noise

    elseif(cmp(fw_init.model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)
        
        loss_function = LossFunctions.loss_MLE_FixedType_detrital_loop_carrying_capacity_separate_normal_noise_extinction_penalty_const

    elseif(cmp(fw_init.model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConstQuant")==0)
        
        loss_function = LossFunctions.loss_MLE_FixedType_detrital_loop_carrying_capacity_separate_normal_noise_extinction_penalty_const_quant

    elseif(cmp(fw_init.model_type,"FixedTypeDetritalLoopRelativeLognormalNoise")==0)
    
        loss_function = LossFunctions.loss_MLE_FixedType_detrital_loop_carrying_capacity_relative_lognormal_noise
                    
    elseif(cmp(fw_init.model_type,"FixedTypeDetritalLoopSeparateLognormalNoise")==0)
    
        loss_function = LossFunctions.loss_MLE_FixedType_detrital_loop_carrying_capacity_separate_lognormal_noise
    
    else

        loss_function = 0

    end 

    return MLEOptions(nPar,var_ini,fw_init.q[fw_init.I_ind][1],param_min,param_max,bc_min,bc_max,model,fw_init.model_type,loss_function,0.01,training_data,
            n_iter_restart,n_offspring,method)
    ### to study the impact of different thresholds for penalization of negative values, change the extinction penalty threshold above (currently 0.01)

end


# variables for avoiding temporary allocations during estimation
mutable struct MLEContainersTypeIII
    transf_par_tmp::Vector{Float64}
    model_tmp::Matrix{Float64}
    pred_distr::Matrix{Normal{Float64}}
    pred_quantiles::Matrix{Float64}
    F
    denom_sum::Matrix{Float64}
end

# INITIALIZE A CONTAINER FOR MLE
# fw_param: a foodweb struct
# f_opt: a fitting options struct
function initialize_MLE_container(fw_param,f_opt)
    transf_par_tmp = zeros(f_opt.nPar)
    model_tmp = zeros(fw_param.nGuilds,length(fw_param.time_ind)-1)
    pred_distr = [Normal(0.0,1.0) for i=1:fw_param.nGuilds,t=1:length(fw_param.time_ind)-1]
    pred_quantiles = [0.0 for i=1:fw_param.nGuilds,t=1:length(fw_param.time_ind)-1]

    F = zeros(fw_param.nGuilds,fw_param.nGuilds,length(fw_param.time_ind)-1)
    denom_sum = zeros(fw_param.nGuilds,length(fw_param.time_ind)-1)

    return MLEContainersTypeIII(transf_par_tmp,model_tmp,pred_distr,pred_quantiles,F,denom_sum)
end


end # module
