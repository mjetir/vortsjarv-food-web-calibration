
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
# method: here referred to as MLE_ODE_disc
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
function initialize_MLE_options_ODE_disc(fw_init,training_data,method,n_iter_restart,n_offspring;kwargs...)
    
    model = DiscreteFoodWebModel.bioenergetic_ODE_discretized
    
    if(cmp(fw_init.model_type,"TypeIIIDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)
    
            nPar = 3*fw_init.nLinks + 3*fw_init.nGuilds 
    
            var_ini = FoodWebUtils.inverse_rho_transform(vcat(fw_init.r,
            fw_init.J[fw_init.I_ind],
            fw_init.B0[fw_init.I_ind],fw_init.q[fw_init.I_ind],
            fw_init.std,fw_init.K,fw_init.u0))
    
    elseif(cmp(fw_init.model_type,"FixedTypeDetritalLoopRelativeNormalNoiseExtinctionPenaltyConst")==0 )

        nPar = 2*fw_init.nLinks + 3*fw_init.nGuilds 

        var_ini = FoodWebUtils.inverse_rho_transform(vcat(fw_init.r,
        fw_init.J[fw_init.I_ind],
        fw_init.B0[fw_init.I_ind],
        fw_init.cv,fw_init.K,fw_init.u0))

    elseif(cmp(fw_init.model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)

        nPar = 2*fw_init.nLinks + 3*fw_init.nGuilds 

        var_ini = FoodWebUtils.inverse_rho_transform(vcat(fw_init.r,
        fw_init.J[fw_init.I_ind],
        fw_init.B0[fw_init.I_ind],
        fw_init.std,fw_init.K,fw_init.u0))

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

    if(cmp(fw_init.model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)
    
        loss_function = LossFunctions.loss_MLE_FixedType_detrital_loop_carrying_capacity_separate_normal_noise_ODE_disc_extinction_penalty_const

    elseif(cmp(fw_init.model_type,"TypeIIIDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)
    
        loss_function = LossFunctions.loss_MLE_TypeIII_detrital_loop_carrying_capacity_separate_normal_noise_ODE_disc_extinction_penalty_const

    elseif(cmp(fw_init.model_type,"FixedTypeDetritalLoopRelativeNormalNoiseExtinctionPenaltyConst")==0)
    
        loss_function = LossFunctions.loss_MLE_FixedType_detrital_loop_carrying_capacity_relative_normal_noise_ODE_disc_extinction_penalty_const

    else

        loss_function = 0

    end 

    return MLEOptions(nPar,var_ini,fw_init.q[fw_init.I_ind][1],param_min,param_max,bc_min,bc_max,model,fw_init.model_type,loss_function,0.01,training_data,
            n_iter_restart,n_offspring,method)

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

