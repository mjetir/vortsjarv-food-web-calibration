
# Maria Tirronen 2025
# https://orcid.org/0000-0001-6052-3186



# Minimize the loss function w.r.t. the unknown model 
# parameters using evolutionary optimization
module FitFoodWebModel

using StatsPlots
using LinearAlgebra
using Random, Distributions
using CSV, DataFrames, Statistics
using Evolutionary
using Metaheuristics

include("set_foodweb_parameters.jl")
include("foodweb_model_discretized.jl")
include("foodweb_utils.jl")
include("plot_results_for_foodwebs.jl")
include("foodweb_fitting_options.jl")
include("foodweb_loss_functions.jl")


# STRUCT FOR SAVING RESULTS
# loss: initial loss and losses after running EO
# param: estimated parameter values during iteration
# param_final: parameter values corresponding to the minimum of loss, untransformed values
# f_options: fitting options struct
struct FittingResults
    loss::Vector{Float64}
    param::Matrix{Float64}
    param_final::Vector{Float64}
    f_options # MLEOptions struct
end


# MINIMIZE THE LOSS FUNCTION USING CMAES, STARTING FROM A GIVEN VALUE OF PARAMETERS
function EO_minimize(fw_tmp,f_opt,f_cont,param_init)

    bc = BoxConstraints(f_opt.bc_min, f_opt.bc_max)
    
    res = Evolutionary.optimize(x->f_opt.loss_function(x,fw_tmp,f_opt,f_cont),bc,param_init,CMAES(lambda=f_opt.n_offspring),
        Evolutionary.Options(iterations=3000))

    println("EO result:")
    println(res)
            
    p_tmp = Evolutionary.minimizer(res);

    loss = f_opt.loss_function(p_tmp,fw_tmp,f_opt,f_cont)

    return p_tmp, loss
end

# MINIMIZE THE LOSS FUNCTION USING CMAES, STARTING FROM A GIVEN VALUE OF PARAMETERS
# restart using the result
function EO_minimize_with_restart(fw_tmp,f_opt,f_cont)
            
    Random.seed!(1526436)

    func_minimize = EO_minimize

    # initialize, includes initial loss
    loss_in_iteration::Vector{Float64}=zeros(f_opt.n_iter_restart+2)

    # initialize, includes initial guess
    parameters_in_iteration::Matrix{Float64}=zeros(f_opt.n_iter_restart+2,length(f_opt.p_init))

    # initialize
    min_loss = f_opt.loss_function(f_opt.p_init,fw_tmp,f_opt,f_cont)

    # intialize
    best_solution=copy(f_opt.p_init)

    loss_in_iteration[1] = copy(min_loss) # initial loss
    parameters_in_iteration[1,:] = copy(best_solution)

    print("Initial loss: ")
    println(min_loss)    

    res = 0
    p_tmp= 0

    p_tmp, loss_in_iteration[2] = func_minimize(fw_tmp,f_opt,f_cont,f_opt.p_init)
    parameters_in_iteration[2,:] = copy(p_tmp)
    if(loss_in_iteration[2]<min_loss)
        best_solution=copy(p_tmp)
        min_loss=copy(loss_in_iteration[2])
    end

    print("=== First round of optimization executed. ")
    print("Loss: ")
    println(loss_in_iteration[2])

    for i=1:f_opt.n_iter_restart
        p_tmp, loss_in_iteration[i+2] = func_minimize(fw_tmp,f_opt,f_cont,p_tmp)
        parameters_in_iteration[i+2,:] = copy(p_tmp)

        print("=== Restarted, the proportion of iterations executed: ")
        print(i/f_opt.n_iter_restart)
        print(". Loss: ")
        println(loss_in_iteration[i+2])
        
        if(loss_in_iteration[i+2]<min_loss)
            best_solution=copy(p_tmp)
            min_loss=copy(loss_in_iteration[i+2])
            println("    ###### Restart provided better fit.")
        end        
    end
    return best_solution, parameters_in_iteration, min_loss, loss_in_iteration
end

# FIT A FOODWEB MODEL 
# fw_init: initial food web struct
# f_opt: fitting options struct
function fit_foodweb(fw_init,f_opt)
        
    f_cont = FittingOptions.initialize_MLE_container(fw_init,f_opt)
    
    fw_tmp = SetFoodWebParameters.copy_foodweb(fw_init)

    param_init = copy(f_opt.p_init)

    best_solution, solutions, min_loss, losses = EO_minimize_with_restart(fw_tmp,f_opt,f_cont)

    # for returning the results:    
    param = solutions

    param_final = best_solution 

    return FittingResults(losses,
    param,
    param_final, 
    f_opt)
end


end #module


