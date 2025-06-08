
# Maria Tirronen 2025
# https://orcid.org/0000-0001-6052-3186



# Functions for plotting
module FoodWebPlots

using StatsPlots
using Colors
using Random, Distributions
using Statistics
using CSV
using DataFrames
using DelimitedFiles
using LinearAlgebra


gr(dpi=600)

include("foodweb_utils.jl")
include("foodweb_model_discretized.jl")
include("set_foodweb_parameters.jl")
include("foodweb_fitting_options.jl")
include("set_foodweb_parameters_ODE.jl")
include("foodweb_loss_functions.jl")

my_green = RGB(0.6549, 0.77255, 0.09412)
my_pink = RGB(0.996,0.008,0.482)
my_gray = RGB(0.9216,0.9255,0.9412)
my_yellow = RGB(0.914,1.0,0.533)
my_hot_pink = RGB(0.996,0.039,0.718)
my_fuchsia = RGB(0.898,0.004,0.522)

my_gray_1 = RGB(0.749,0.749,0.749)
my_gray_2 = RGB(0.498,0.498,0.498)
my_gray_3 = RGB(0.251,0.251,0.251)

my_blue = RGB(0,53/255,197/255)


### FOR PLOTTING RESULTS OF MODEL FITTING 
function plot_and_analyze_predictions(results_from,figures_to,model_type,q_init)
        
    Random.seed!(8758743)

    vorts_data, fw_vorts = SetFoodWebParameters.initialize_vorts_foodweb(model_type,q_init)
    f_opt = FittingOptions.initialize_MLE_options(fw_vorts,vorts_data,"",1,1)    
    f_cont = FittingOptions.initialize_MLE_container(fw_vorts,f_opt)
    
    ### Read parameter estimates
    param_est = 0
    losses = 0
    try
        param_est = readdlm(results_from*"/EO_parameter_estimates/parameter_estimates.txt",Float64)
        losses = readdlm(results_from*"/EO_loss/losses.txt",Float64)
    catch
        println("No appropriate data files in the folder "*results_from*"/EO_parameter_estimates.")
        return 0
    end

    fw_pred = SetFoodWebParameters.copy_foodweb(fw_vorts) 
    
    min_neg_likelihood = 0

    println("")

    println(model_type)

    println("q (init/fixed): ")
    println(q_init)


    if(cmp(model_type,"TypeIIIDetritalLoopSeparateNormalNoise")==0  )

        transformed_param = FoodWebUtils.rho_transform(param_est)
                    
        fw_pred.r = transformed_param[1:fw_vorts.nGuilds - 1]

        fw_pred.J[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds:fw_vorts.nGuilds + fw_vorts.nLinks - 1]
        fw_pred.B0[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + fw_vorts.nLinks:fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]
        fw_pred.q[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + 2*fw_vorts.nLinks:fw_vorts.nGuilds + 3*fw_vorts.nLinks - 1]

        fw_pred.std = transformed_param[fw_vorts.nGuilds + 3*fw_vorts.nLinks:2*fw_vorts.nGuilds + 3*fw_vorts.nLinks - 1]

        fw_pred.K = transformed_param[end]                        
        

    elseif(cmp(model_type,"TypeIIIDetritalLoopRelativeNormalNoise")==0  )

            transformed_param = FoodWebUtils.rho_transform(param_est)
                        
            fw_pred.r = transformed_param[1:fw_vorts.nGuilds - 1]

            fw_pred.J[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds:fw_vorts.nGuilds + fw_vorts.nLinks - 1]
            fw_pred.B0[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + fw_vorts.nLinks:fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]
            fw_pred.q[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + 2*fw_vorts.nLinks:fw_vorts.nGuilds + 3*fw_vorts.nLinks - 1]

            fw_pred.cv = transformed_param[fw_vorts.nGuilds + 3*fw_vorts.nLinks:2*fw_vorts.nGuilds + 3*fw_vorts.nLinks - 1]

            fw_pred.K = transformed_param[end]                        
            
    
    ### fixed q
    elseif(cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoise")==0 ||
                cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0 || 
                cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConstQuant")==0 || 
                cmp(model_type,"FixedTypeDetritalLoopSeparateLognormalNoise")==0 )
    
                transformed_param = FoodWebUtils.rho_transform(param_est)
                            
                fw_pred.r = transformed_param[1:fw_vorts.nGuilds - 1]
    
                fw_pred.J[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds:fw_vorts.nGuilds + fw_vorts.nLinks - 1]
                fw_pred.B0[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + fw_vorts.nLinks:fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]

                fw_pred.std = transformed_param[fw_vorts.nGuilds + 2*fw_vorts.nLinks:2*fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]
    
                fw_pred.K = transformed_param[end]                        
                    
            
    elseif(cmp(model_type,"FixedTypeDetritalLoopRelativeNormalNoise")==0 || 
                    cmp(model_type,"FixedTypeDetritalLoopRelativeLognormalNoise")==0 )
        
                    transformed_param = FoodWebUtils.rho_transform(param_est)
                                
                    fw_pred.r = transformed_param[1:fw_vorts.nGuilds - 1]
        
                    fw_pred.J[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds:fw_vorts.nGuilds + fw_vorts.nLinks - 1]
                    fw_pred.B0[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + fw_vorts.nLinks:fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]
        
                    fw_pred.cv = transformed_param[fw_vorts.nGuilds + 2*fw_vorts.nLinks:2*fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]
        
                    fw_pred.K = transformed_param[end]                        
                            
    
    else 
        println("Model not defined.")
        return 0
    end


    println("")

    prediction_mean =  f_opt.model(vorts_data,fw_pred,f_cont) 

    println("CV of internal fluctuations:")
    println(sum(Statistics.mean(prediction_mean,dims=2)./Statistics.std(prediction_mean,dims=2))/size(prediction_mean,1))

    min_neg_likelihood = minimum(losses)

    if(cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)
        min_neg_likelihood -= 100.0*sum(prediction_mean .<= 0.0)
    end

    if(cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConstQuant")!=0)
        println("")
        println("Maximum log-likelihood:")
        println(round(-min_neg_likelihood,digits=5))  
    
        AIC = 2*length(param_est)+2*min_neg_likelihood
        println("")
        println("AIC: ")
        println(round(AIC,digits=5))
    end


    if(cmp(model_type,"TypeIIIDetritalLoopSeparateNormalNoise")==0 ||
        cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoise")==0 ||
        cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0 ||
        cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConstQuant")==0
        )

        posterior_predictives = [Normal(prediction_mean[i,t],fw_pred.std[i]) 
            for i=1:fw_vorts.nGuilds,t=1:length(fw_vorts.time_ind)-1]

    elseif(cmp(model_type,"TypeIIIDetritalLoopRelativeNormalNoise")==0 ||
        cmp(model_type,"FixedTypeDetritalLoopRelativeNormalNoise")==0)

        posterior_predictives = [Normal(prediction_mean[i,t],fw_pred.cv[i]*vorts_data[i,t]) 
            for i=1:fw_vorts.nGuilds,t=1:length(fw_vorts.time_ind)-1]   
            
    elseif(cmp(model_type,"FixedTypeDetritalLoopSeparateLognormalNoise")==0 )
    
                posterior_predictives = [exp.(rand(Normal(log.(max.(0.00000001,prediction_mean[i,t])),fw_pred.std[i]),1000000)) 
                    for i=1:fw_vorts.nGuilds,t=1:length(fw_vorts.time_ind)-1]
    
    elseif(cmp(model_type,"FixedTypeDetritalLoopRelativeLognormalNoise")==0)
     
        posterior_predictives = [exp.(rand(Normal(log.(max.(0.00000001,prediction_mean[i,t])),fw_pred.cv[i]*vorts_data[i,t]),1000000))
                    for i=1:fw_vorts.nGuilds,t=1:length(fw_vorts.time_ind)-1]    
    else  
        posterior_predictives = 0                
    end 

    lower_quantiles = quantile.(posterior_predictives,0.05)
    upper_quantiles = quantile.(posterior_predictives,0.95)

    quantiles_for_penalty = quantile.(posterior_predictives,0.01)

    if(cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConstQuant")==0)
            min_neg_likelihood -= 100.0*sum(quantiles_for_penalty .<= 0.0)
            println("")
            println("Maximum log-likelihood:")
            println(round(-min_neg_likelihood,digits=5))  
    
            AIC = 2*length(param_est)+2*min_neg_likelihood
            println("")
            println("AIC: ")
            println(round(AIC,digits=5))
    end

    println("")
    println("Number of negative mean predictions guild-wise:")
    neg_pred = 0
    neg_excl_det = 0
    for j=1:fw_vorts.nGuilds
        print("Guild"*string(j)*":")
        neg = sum(prediction_mean[j,1:end-1] .<= 0.0)
        println(neg)
        neg_pred += neg
        if(neg > 0)
            println(prediction_mean[j,1:end-1] .<= 0.0)
            println(prediction_mean[j,1:end-1])
        end
        if(j < fw_vorts.nGuilds)
            neg_excl_det += neg
        end
    end 
    println("")
    println("Total number of negative mean predictions:")
    println(neg_pred)
    println("Total number of negative mean predictions excluding detritus:")
    println(neg_excl_det)


    data_for_pred = vorts_data[:,2:end]
    non_missing = data_for_pred .>= 0.0

    pred = prediction_mean[:,1:end-1]

    bray_curtis_sum = sum(min.(pred[non_missing], data_for_pred[non_missing]))/(sum(pred[non_missing]) + sum(data_for_pred[non_missing]))

    bray_curtis = 1-2*bray_curtis_sum

    print("Bray-Curtis: ")
    println(bray_curtis)

    CPI_coverage = sum((lower_quantiles[non_missing] .<= data_for_pred[non_missing]) .&& (upper_quantiles[non_missing] .>= data_for_pred[non_missing]))/sum(non_missing)

    print("90 % CPI coverage: ")
    println(CPI_coverage)


    #########################


    println("")
    println("Plotting to "*figures_to)
    if ~ispath(results_from*"/"*figures_to)
        mkpath(results_from*"/"*figures_to)
    end


    # init
    plot_ = StatsPlots.plot(zeros(3),ones(3))
    plots = [plot_ for i=1:fw_vorts.nGuilds+1]

    ylims_upper = [210,16,4,40,12.5,6,3,2.3,22,0.5,1.2,4,7,1.5,40]
    ylims_lower = [-53,-5,-0.3,-18.0,-10,-2,-1.5,-0.8,-15,-0.4,-0.3,-1.5,-3,-0.5,-10]
#    ylims_upper = [500,25,6,130,70,20,8,14,20,2500,6,1.2*10^7,12,2.0,70] # for log-normal noise
#    ylims_lower = zeros(15) # for log-normal noise

    guild_names = ["Phytopl.","Protozoopl.","Metazoopl.","Benthos","Ruffe","Roach","Bleak","White bream","Bream",
        "Smelt","Perch","Eel","Pikeperch","Pike","Detritus"]

    i=1 

    plots[i] = StatsPlots.plot(fw_vorts.time[2:end], lower_quantiles[i,:], fillrange = upper_quantiles[i,:],
    ylim=(ylims_lower[i],ylims_upper[i]), 
    xlim = (1994.5,2012.5),
    fillalpha = 1.0, linealpha=0, color = my_gray_1 , 
    label = "",  
    ylabel=guild_names[i],
    title = "Environmental noise",
    fontfamily=:"Computer Modern",
    left_margin = 20Plots.mm, right_margin = 1Plots.mm)
    StatsPlots.plot!(plots[i],fw_vorts.time[2:end],prediction_mean[i,1:end-1],color=:black, label = "", linewidth=2.5)
    StatsPlots.scatter!(plots[i],fw_vorts.time[2:end],vorts_data[i,2:end], label = "", color=my_blue)
    StatsPlots.plot!(plots[i],fw_vorts.time[2:end],zeros(length(fw_vorts.time[2:end])),color=:white, label = "", linewidth=2.0)


    for i=2:fw_vorts.nGuilds

        plots[i] = StatsPlots.plot(fw_vorts.time[2:end], lower_quantiles[i,:], fillrange = upper_quantiles[i,:],
        ylim=(ylims_lower[i],ylims_upper[i]), 
        xlim = (1994.5,2012.5),
        fillalpha = 1.0, linealpha=0, color = my_gray_1 , 
        label = "",  
        ylabel=guild_names[i],
        fontfamily=:"Computer Modern",
        left_margin = 20Plots.mm, right_margin = 1Plots.mm)
        StatsPlots.plot!(plots[i],fw_vorts.time[2:end],prediction_mean[i,1:end-1],color=:black, label = "", linewidth=2.5)
        StatsPlots.scatter!(plots[i],fw_vorts.time[2:end],vorts_data[i,2:end], label = "", color=my_blue)
        StatsPlots.plot!(plots[i],fw_vorts.time[2:end],zeros(length(fw_vorts.time[2:end])),color=:white, label = "", linewidth=2.0)

    end

    imputation_years = findall(i->in(i,[1999,2001,2002,2006,2010,2011]),fw_vorts.time)

    StatsPlots.scatter!(plots[fw_vorts.nGuilds],fw_vorts.time[imputation_years],vorts_data[fw_vorts.nGuilds,imputation_years], color=:white, label = "")

    i = fw_vorts.nGuilds

    plots[i+1] = StatsPlots.plot(fw_vorts.time[2:end], lower_quantiles[i,:], fillrange = upper_quantiles[i,:], 
        fillalpha = 1.0, linealpha=0, color = my_gray_1, 
        label = "Prediction of true biomass",  
        ylabel="",xlabel="",
        fontfamily=:"Computer Modern",
        left_margin = 20Plots.mm, right_margin = 1Plots.mm)
    StatsPlots.plot!(plots[i+1],fw_vorts.time[2:end],prediction_mean[i,1:end-1],color=:black, label = "Median of prediction", linewidth=2.5)
    StatsPlots.scatter!(plots[i+1],fw_vorts.time[2:end],vorts_data[i,2:end], label = "Recorded data", color=my_blue)

    StatsPlots.scatter!(plots[i+1],fw_vorts.time[imputation_years],vorts_data[fw_vorts.nGuilds,imputation_years], color=:white, label = "Interpolated data")


    plot(plots[1],plots[9],plots[2],plots[10],plots[3],plots[11],plots[4],plots[12],plots[5],
        plots[13],plots[6],plots[14],plots[7],plots[15],plots[8],plots[16],
        layout=grid(8, 2,heights=0.12.*ones(16),widths=0.4.*ones(16)),
        size=(800,900))

    savefig(results_from*"/"*figures_to*"/posterior_predictives_all_guilds.png")
    savefig(results_from*"/"*figures_to*"/posterior_predictives_all_guilds.svg")

end



function plot_losses(f_res,folder_name)
    println("Plotting to "*folder_name)
    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    l = @layout [a ; b]
    p1 = StatsPlots.plot(1:length(f_res.loss),f_res.loss,lw=2,color=:black,xlabel="Iteration",
     ylabel="Loss", label = "", xticks = 0:5:length(f_res.loss)-1)
    p2 = StatsPlots.plot(30:length(f_res.loss),f_res.loss[30:end],lw=2,color=:black,xlabel="Iteration",
    ylabel="Loss", label = "", xticks = 30:5:length(f_res.loss))
    plot(p1, p2, layout = l)
    savefig(folder_name*"/loss_in_iteration.png")
end



function plot_sensitivity_of_parameter_estimates(results_from_list,figures_to)
    
    param_est = 0
    losses = 0

    guild_names = ["Phytopl.","Protozoopl.","Metazoopl.","Benthos","Ruffe","Roach","Bleak","White bream","Bream",
    "Smelt","Perch","Eel","Pikeperch","Pike","Detritus"]

    vorts_data, fw_vorts = SetFoodWebParameters.initialize_vorts_foodweb("FixedTypeDetritalLoopSeparateNormalNoise",0.3)
    
    fw_pred = SetFoodWebParameters.copy_foodweb(fw_vorts) 

    #### ODE ####################

    vorts_data_ODE, fw_vorts_ODE = SetFoodWebParametersODE.initialize_vorts_foodweb_ODE_disc("FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst",0.3)

    fw_pred_ODE = SetFoodWebParametersODE.copy_foodweb_ODE(fw_vorts_ODE) 

    #############################

    param_table = zeros(length(results_from_list)-1,2*fw_vorts.nGuilds + 2*fw_vorts.nLinks)

    param_table_ODE = zeros(2*fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks)


    stds_data = zeros(fw_vorts.nGuilds)

    stds_log_data = zeros(fw_vorts.nGuilds)

    min_data = zeros(fw_vorts.nGuilds)

    max_data = zeros(fw_vorts.nGuilds)


    for i in 1:fw_vorts.nGuilds
        if(i == 12 || i == 15)
            stds_data[i] = Statistics.std(vorts_data[i,1:end-1])
            stds_log_data[i] = Statistics.std(log.(vorts_data[i,1:end-1]))
            min_data[i] = minimum(vorts_data[i,1:end-1])
            max_data[i] = maximum(vorts_data[i,1:end-1])
        else
            stds_data[i] = Statistics.std(vorts_data[i,:])
            stds_log_data[i] = Statistics.std(log.(vorts_data[i,:]))
            min_data[i] = minimum(vorts_data[i,:])
            max_data[i] = maximum(vorts_data[i,:])
        end
    end

    res_guild_min = repeat(min_data',fw_vorts.nGuilds)
    res_guild_max = repeat(max_data',fw_vorts.nGuilds)

    ################ Js from literature#######################

    max_body_weights = [0, 0, 0, 0, 0.4, 1.8, 0.06, 1.0, 6.0, 0.178, 4.8, 6.6, 20.0, 28.4, 0.0]

    J_lit = copy(fw_vorts.J)
    J_lit .= 0.0
    for i in 5:14
        for j in 1:fw_vorts.nGuilds
            if in(CartesianIndex(i,j),fw_vorts.I_ind) 
                J_lit[i,j] = 0.2*8.9*(max_body_weights[i]^(-0.25))/fw_vorts.e[i,j]
            end
        end
    end


    ###################################


    for j=1:length(results_from_list)-1
            
            ### Read parameter estimates
            try
                # try to read the untransformed (in R^n) parameter estimates
                param_est = readdlm(results_from_list[j]*"/EO_parameter_estimates/parameter_estimates.txt",Float64)
                losses = readdlm(results_from_list[j]*"/EO_loss/losses.txt",Float64)
            catch
                println("No appropriate data files in the folder "*results_from_list[j]*"/EO_parameter_estimates.")
                return 0
            end
        
            param_table[j,:] = FoodWebUtils.rho_transform(param_est)

            transformed_param = param_table[j,:]
                                
            fw_pred.r = transformed_param[1:fw_vorts.nGuilds - 1]

            fw_pred.J[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds:fw_vorts.nGuilds + fw_vorts.nLinks - 1]

            fw_pred.B0[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + fw_vorts.nLinks:fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]

            fw_pred.std = transformed_param[fw_vorts.nGuilds + 2*fw_vorts.nLinks:2*fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]

            fw_pred.K = transformed_param[end]         
            
            #### Compare to data ################### 

            println("Estimated max. ingestion rates compared to literature:")
            println("Model "*string(j))
            println(sum((fw_pred.J[fw_vorts.I_ind] .>= J_lit[fw_vorts.I_ind]) .& (J_lit[fw_vorts.I_ind] .> 0.0))./sum(J_lit[fw_vorts.I_ind] .> 0.0))
            println("")

            println("Estimated half-sat. constants compared to data:")

            println("Model "*string(j))
            
            println("Within data:")
            println(sum((fw_pred.B0[fw_vorts.I_ind] .>= res_guild_min[fw_vorts.I_ind]) .& (fw_pred.B0[fw_vorts.I_ind] .<= res_guild_max[fw_vorts.I_ind])  )./length(fw_vorts.I_ind) )

            println("Higher than data:")
            println(sum( fw_pred.B0[fw_vorts.I_ind] .> res_guild_max[fw_vorts.I_ind] )./length(fw_vorts.I_ind) )

            println("Lower than data:")
            println(sum( fw_pred.B0[fw_vorts.I_ind] .< res_guild_min[fw_vorts.I_ind] )./length(fw_vorts.I_ind) )
            println("")

    end


    try
        # try to read the untransformed (in R^n) parameter estimates
        param_est = readdlm(results_from_list[end]*"/EO_parameter_estimates/parameter_estimates.txt",Float64)
        losses = readdlm(results_from_list[end]*"/EO_loss/losses.txt",Float64)
    catch
        println("No appropriate data files in the folder "*results_from_list[end]*"/EO_parameter_estimates.")
        return 0
    end

    
    transformed_param = FoodWebUtils.rho_transform(param_est)
                    
    fw_pred_ODE.r = transformed_param[1:fw_vorts_ODE.nGuilds - 1]

    fw_pred_ODE.J[fw_vorts_ODE.I_ind] = transformed_param[fw_vorts_ODE.nGuilds:fw_vorts_ODE.nGuilds + fw_vorts_ODE.nLinks - 1]

    fw_pred_ODE.B0[fw_vorts_ODE.I_ind] = transformed_param[fw_vorts_ODE.nGuilds + fw_vorts_ODE.nLinks:fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks - 1]

    fw_pred_ODE.std = transformed_param[fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks:2*fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks - 1]

    fw_pred_ODE.K = transformed_param[2*fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks]
    
    fw_pred_ODE.u0 = transformed_param[2*fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks+1:end]

    param_table_ODE = FoodWebUtils.rho_transform(param_est)


    #### Compare to data ################### 

    println("Estimated max. ingestion rates compared to literature:")
    println("Obs. noise")
    println(sum( (fw_pred_ODE.J[fw_vorts_ODE.I_ind] .>= J_lit[fw_vorts_ODE.I_ind]) .& (J_lit[fw_vorts_ODE.I_ind] .> 0.0) )./sum(J_lit[fw_vorts_ODE.I_ind] .> 0.0))  
    println("")

    println("Estimated half-sat. constants compared to data:")

    println("Obs. noise")
    
    println("Within data:")
    println(sum((fw_pred_ODE.B0[fw_vorts_ODE.I_ind] .>= res_guild_min[fw_vorts_ODE.I_ind]) .& (fw_pred_ODE.B0[fw_vorts_ODE.I_ind] .<= res_guild_max[fw_vorts_ODE.I_ind])  )./length(fw_vorts_ODE.I_ind) )

    println("Higher than data:")
    println(sum( fw_pred_ODE.B0[fw_vorts_ODE.I_ind] .> res_guild_max[fw_vorts_ODE.I_ind] )./length(fw_vorts_ODE.I_ind) )

    println("Lower than data:")
    println(sum( fw_pred_ODE.B0[fw_vorts_ODE.I_ind] .< res_guild_min[fw_vorts_ODE.I_ind] )./length(fw_vorts_ODE.I_ind) )
    println("")


    ##########################################################

    println("Plotting to "*figures_to)
    if ~ispath(figures_to)
        mkpath(figures_to)
    end


    r_ind = 1

    T_ind = 2:fw_vorts.nGuilds - 1
    
    J_ind = fw_vorts.nGuilds:fw_vorts.nGuilds + fw_vorts.nLinks - 1

    B0_ind = fw_vorts.nGuilds + fw_vorts.nLinks:fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1

    std_ind = fw_vorts.nGuilds + 2*fw_vorts.nLinks:2*fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1

    K_ind = 2*fw_vorts.nGuilds + 2*fw_vorts.nLinks

    nPar = 2*fw_vorts.nGuilds + 2*fw_vorts.nLinks


    J_ind_ODE = fw_vorts_ODE.nGuilds:fw_vorts_ODE.nGuilds + fw_vorts_ODE.nLinks - 1

    B0_ind_ODE = fw_vorts_ODE.nGuilds + fw_vorts_ODE.nLinks:fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks - 1

    std_ind_ODE = fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks:2*fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks - 1

    K_ind_ODE = 2*fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks

    u0_ind_ODE = 2*fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks+1:3*fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks+1

    nPar_ODE = 3*fw_vorts_ODE.nGuilds + 2*fw_vorts_ODE.nLinks


    #### PLOT r_1, K, T_i, std ###############


    j = 1

    p1 = plot([0,0],[0,0],label="")
    
    label_ind = 1
    
    my_labels = ["Absolute normal environmental noise", "Lognormal environmental noise", "Absolute normal observation noise", "", "Literature", "Range of data (resource)"]
    
    range_max = 0
    
    
    for i=1
    
            j = j + 3
    
            StatsPlots.annotate!(p1, j, -0.8, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 90), ylabel = "Intrinsic growth rate [1/year]")
    
            range_max = maximum([param_table[1,i],param_table[2,i],param_table_ODE[i]])
    
            StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(0,5),legend=:none)
    
            if(i == 1)
                        my_label = my_labels[1]
            else 
                        my_label = my_labels[4]
            end
    
            StatsPlots.scatter!(p1,[j,j], [param_table[1,i],param_table[1,i]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=5,
                        fontfamily="Computer Modern",
                        left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 25Plots.mm)
        
    
            if(i == 1)
                        my_label = my_labels[2]
            else 
                        my_label = my_labels[4]
            end
    
            StatsPlots.scatter!(p1,[j,j], [param_table[2,i],param_table[2,i]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=4)
                        
            if(i == 1)
                        my_label = my_labels[3]
            else 
                        my_label = my_labels[4]
            end
    
                    #plot ODE results
            StatsPlots.scatter!(p1,[j,j], [param_table_ODE[i],param_table_ODE[i]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                            label=my_label,markersize=3)
    
    
            j = j +1
    
    end

    j = 1

    p2 = plot([0,0],[0,0],label="")
    
    label_ind = 1
    
    my_labels = ["Absolute normal environmental noise", "Lognormal environmental noise", "Absolute normal observation noise", "", "Literature", "Range of data"]
    
    range_max = 0
    
    
    for i=1
    
            j = j + 3
    
            StatsPlots.annotate!(p2, j, -40.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 90), ylabel = "Carrying capacity [tonnes/km2]")
    
            range_max = maximum([param_table[1,K_ind],param_table[2,K_ind],param_table_ODE[K_ind_ODE], max_data[i]])
    
            StatsPlots.plot!(p2,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(-0.3,250),legend=:topleft)
    
            if(i == 1)
                        my_label = my_labels[1]
            else 
                        my_label = my_labels[4]
            end
    
            StatsPlots.scatter!(p2,[j,j], [param_table[1,K_ind],param_table[1,K_ind]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=5,
                        fontfamily="Computer Modern",
                        left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 25Plots.mm)

    
            if(i == 1)
                        my_label = my_labels[2]
            else 
                        my_label = my_labels[4]
            end
    
            StatsPlots.scatter!(p2,[j,j], [param_table[2,K_ind],param_table[2,K_ind]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=4)
                        
            if(i == 1)
                        my_label = my_labels[3]
            else 
                        my_label = my_labels[4]
            end
    
                    #plot ODE results
            StatsPlots.scatter!(p2,[j,j], [param_table_ODE[K_ind_ODE],param_table_ODE[K_ind_ODE]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                            label=my_label,markersize=3)
            
            if(i == 1)
                                my_label = my_labels[6]
            else 
                                my_label = my_labels[4]
            end
        
            range_min = min_data[i]
            range_max = max_data[i]


            StatsPlots.plot!(p2,[j,j], [range_min,range_max], color=:black,linewidth=2, label = my_label, xticks = [])

    
            j = j +1
    
    end

    
    plot(p1,p2, layout = grid(1,2,heights=ones(2), widths=[0.2,0.78]), size=(500,500))
    
    savefig(figures_to*"/sensitivity_of_r_and_K.png")
    savefig(figures_to*"/sensitivity_of_r_and_K.svg")    


    j = 1

    p1 = plot([0,0],[0,0],label="")
    
    label_ind = 1
    
    my_labels = ["Absolute normal environmental noise", "Lognormal environmental noise", "Absolute normal observation noise", "", "Literature", "Range of data (resource)"]
    
    range_max = 0
    
    
    for i=2:fw_vorts.nGuilds-1
    
            j = j + 3
    
            StatsPlots.annotate!(p1, j, -3.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 90), ylabel = "Mass-specific respiration rate [1/year]")
    
            range_max = maximum([param_table[1,i],param_table[2,i],param_table_ODE[i]])
    
            StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(-0.3,10),legend=:topleft)
    
            if(i == 2)
                        my_label = my_labels[1]
            else 
                        my_label = my_labels[4]
            end
    
            StatsPlots.scatter!(p1,[j,j], [param_table[1,i],param_table[1,i]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=5,
                        fontfamily="Computer Modern",
                        left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 40Plots.mm)

   
            if(i == 2)
                        my_label = my_labels[2]
            else 
                        my_label = my_labels[4]
            end
    
            StatsPlots.scatter!(p1,[j,j], [param_table[2,i],param_table[2,i]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=4)
                        
            if(i == 2)
                        my_label = my_labels[3]
            else 
                        my_label = my_labels[4]
            end
    
                    #plot ODE results
            StatsPlots.scatter!(p1,[j,j], [param_table_ODE[i],param_table_ODE[i]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                            label=my_label,markersize=3)
    
    
            j = j +1
    
    end
    
    plot(p1,size=(750,500))
    
    savefig(figures_to*"/sensitivity_of_respiration_rates.png")
    savefig(figures_to*"/sensitivity_of_respiration_rates.svg")    


    j = 1

    p1 = plot([0,0],[0,0],label="")
    
    label_ind = 1
    
    my_labels = ["Absolute normal environmental noise", "Lognormal environmental noise", "Absolute normal observation noise", "", "Literature", "Range of data (resource)", "Data", "Log-transformed data"]
    
    range_max = 0
    
    
    for i=1
    
            j = j + 3
    
            StatsPlots.annotate!(p1, j, -15.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 90), ylabel = "Standard deviation [tonnes/km2]")
    
            range_max = maximum([param_table[1,std_ind[i]],param_table[2,std_ind[i]],param_table_ODE[std_ind_ODE[i]]])
    
            StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(-0.3,50),legend=:none)
    
            if(i == 2)
                        my_label = my_labels[1]
            else 
                        my_label = my_labels[4]
            end
    
            StatsPlots.scatter!(p1,[j,j], [param_table[1,std_ind[i]],param_table[1,std_ind[i]]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=5,
                        fontfamily="Computer Modern",
                        left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 40Plots.mm)
                        
    
            if(i == 2)
                        my_label = my_labels[2]
            else 
                        my_label = my_labels[4]
            end
    
            StatsPlots.scatter!(p1,[j,j], [param_table[2,std_ind[i]],param_table[2,std_ind[i]]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=4)
            
            
            if(i == 2)
                        my_label = my_labels[3]
            else 
                        my_label = my_labels[4]
            end
    
                    #plot ODE results
            StatsPlots.scatter!(p1,[j,j], [param_table_ODE[std_ind_ODE[i]],param_table_ODE[std_ind_ODE[i]]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                            label=my_label,markersize=3)
            
            if(i == 2)
                                my_label = my_labels[7]
            else 
                                my_label = my_labels[4]
            end
        
            StatsPlots.scatter!(p1, [j,j], [stds_data[i], stds_data[i]], markersize=4, 
                            color =:white, markerstrokecolor = my_gray_1,label=my_label)
    
            if(i == 2)
                                my_label = my_labels[8]
            else 
                                my_label = my_labels[4]
            end
        
            StatsPlots.scatter!(p1, [j,j], [stds_log_data[i], stds_log_data[i]], markersize=4, 
                            color =:white, markerstrokecolor = my_fuchsia,label=my_label)
                            
            j = j +1
    
    end

    p2 = plot([0,0],[0,0],label="")
    
    label_ind = 1
    
    my_labels = ["Absolute normal environmental noise", "Lognormal environmental noise", "Absolute normal observation noise", "", "Literature", "Range of data (resource)", "Data", "Log-transformed data"]
    
    range_max = 0
    
    
    for i=2:fw_vorts.nGuilds
    
            j = j + 3
    
            StatsPlots.annotate!(p2, j, -4.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 90), ylabel = "")
    
            range_max = maximum([param_table[1,std_ind[i]],param_table[2,std_ind[i]],param_table_ODE[std_ind_ODE[i]]])
    
            StatsPlots.plot!(p2,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(-0.3,13),legend=:topleft)
    
            if(i == 2)
                        my_label = my_labels[1]
            else 
                        my_label = my_labels[4]
            end
    
            StatsPlots.scatter!(p2,[j,j], [param_table[1,std_ind[i]],param_table[1,std_ind[i]]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=5,
                        fontfamily="Computer Modern",
                        left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 40Plots.mm)


            if(i == 2)
                        my_label = my_labels[2]
            else 
                        my_label = my_labels[4]
            end
    
            StatsPlots.scatter!(p2,[j,j], [param_table[2,std_ind[i]],param_table[2,std_ind[i]]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=4)
                        
            if(i == 2)
                        my_label = my_labels[3]
            else 
                        my_label = my_labels[4]
            end
    
                    #plot ODE results
            StatsPlots.scatter!(p2,[j,j], [param_table_ODE[std_ind_ODE[i]],param_table_ODE[std_ind_ODE[i]]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                            label=my_label,markersize=3)
            
            if(i == 2)
                                my_label = my_labels[7]
            else 
                                my_label = my_labels[4]
            end
        
            StatsPlots.scatter!(p2, [j,j], [stds_data[i], stds_data[i]], markersize=4, 
                            color =:white, markerstrokecolor = my_gray_1,label=my_label)
    
            if(i == 2)
                                my_label = my_labels[8]
            else 
                                my_label = my_labels[4]
            end
        
            StatsPlots.scatter!(p2, [j,j], [stds_log_data[i], stds_log_data[i]], markersize=4, 
                            color =:white, markerstrokecolor = my_fuchsia,label=my_label)

                            
            j = j +1
    
    end

    
    plot(p1,p2, layout = grid(1, 2,heights=ones(2),widths=[0.1,0.87]), size=(750,500))
    
    savefig(figures_to*"/sensitivity_of_stds.png")
    savefig(figures_to*"/sensitivity_of_stds.svg")    


    #### PLOT MAXIMUM INGESTION RATES ######################

    j = 1

    p1 = plot([0,0],[0,0],label="")

    label_ind = 1

    range_max = 0

    StatsPlots.annotate!(p1, -4, -5.0, text("Resource", :black, :left, 10,"Computer Modern",rotation = 0))

    StatsPlots.annotate!(p1, -4, -20.0, text("Consumer", :black, :left, 10,"Computer Modern",rotation = 0))

    for i=2:4

        j = j + 3

        StatsPlots.annotate!(p1, j, -20.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 0), ylabel = "Maximum ingestion rate [1/year]")

        for k=1:length(fw_vorts.I_ind)

            par_ind = J_ind[k]

            par_ind_ODE = 0

            if(fw_vorts.I_ind[k][1]==i)

                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    index_ODE = findfirst(item -> item == fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    par_ind_ODE = J_ind_ODE[index_ODE]
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind],param_table_ODE[par_ind_ODE]])
                else
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind]])
                end

                StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(0,55),legend=:topleft)

                if(k == 1)
                    my_label = my_labels[1]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[1,par_ind],param_table[1,par_ind]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=5,
                    fontfamily="Computer Modern",
                    left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 40Plots.mm)
                

                if(k == 1)
                    my_label = my_labels[2]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[2,par_ind],param_table[2,par_ind]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=4)

                StatsPlots.annotate!(p1, j, -1.0, text(guild_names[fw_vorts.I_ind[k][2]], :black, :right, 8,"Computer Modern", rotation = 90))
                
                if(k == 1)
                    my_label = my_labels[3]
                else 
                    my_label = my_labels[4]
                end

                #plot ODE results
                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    StatsPlots.scatter!(p1,[j,j], [param_table_ODE[par_ind_ODE],param_table_ODE[par_ind_ODE]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                        label=my_label,markersize=3)
                end 

                if(i == 5)
                    my_label = my_labels[5]
                else 
                    my_label = my_labels[4]
                end

                if(J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]] > 0.0)
                    StatsPlots.scatter!(p1, [j,j], [J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]],J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]]], markersize=3, 
                        color =:white, markerstrokecolor = my_gray_1,label=my_label)
                end


                j = j +1

            end
        end
    end

    plot(p1,size=(350,500))

    savefig(figures_to*"/sensitivity_of_max_ing_rates.png")
    savefig(figures_to*"/sensitivity_of_max_ing_rates.svg")


    j = 1

    p1 = plot([0,0],[0,0],label="")

    label_ind = 1

    range_max = 0

    StatsPlots.annotate!(p1, -3, -12.0, text("Resource", :black, :left, 10,"Computer Modern",rotation = 0))

    StatsPlots.annotate!(p1, -3, -20.0, text("Consumer", :black, :left, 10,"Computer Modern",rotation = 0))

    for i=5:9

        j = j + 3

        StatsPlots.annotate!(p1, j, -20.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 0), ylabel = "Maximum ingestion rate [1/year]")

        for k=1:length(fw_vorts.I_ind)

            par_ind = J_ind[k]

            par_ind_ODE = 0

            if(fw_vorts.I_ind[k][1]==i)

                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    index_ODE = findfirst(item -> item == fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    par_ind_ODE = J_ind_ODE[index_ODE]
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind],param_table_ODE[par_ind_ODE]])
                else
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind]])
                end

                StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(-1.5,55),legend=:topleft)

                if(k == 10)
                    my_label = my_labels[1]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[1,par_ind],param_table[1,par_ind]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=5,
                    fontfamily="Computer Modern",
                    left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 25Plots.mm)
                    

                if(k == 10)
                    my_label = my_labels[2]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[2,par_ind],param_table[2,par_ind]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=4)

                StatsPlots.annotate!(p1, j, -4.0, text(guild_names[fw_vorts.I_ind[k][2]], :black, :right, 8,"Computer Modern", rotation = 90))
                
                if(k == 10)
                    my_label = my_labels[3]
                else 
                    my_label = my_labels[4]
                end

                #plot ODE results
                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    StatsPlots.scatter!(p1,[j,j], [param_table_ODE[par_ind_ODE],param_table_ODE[par_ind_ODE]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                        label=my_label,markersize=3)
                end 

                if(k == 10)
                    my_label = my_labels[5]
                else 
                    my_label = my_labels[4]
                end

                if(J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]] > 0.0)
                    StatsPlots.scatter!(p1, [j,j], [J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]],J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]]], markersize=3, 
                        color =:white, markerstrokecolor = my_gray_1,label=my_label)
                end


                j = j +1

            end
        end
    end

    plot(p1,size=(600,400))

    savefig(figures_to*"/sensitivity_of_max_ing_rates_b.png")
    savefig(figures_to*"/sensitivity_of_max_ing_rates_b.svg")


    j = 1

    p1 = plot([0,0],[0,0],label="")

    label_ind = 1

    range_max = 0

    StatsPlots.annotate!(p1, -2, -5.0, text("Resource", :black, :left, 10,"Computer Modern",rotation = 0))

    StatsPlots.annotate!(p1, -2, -20.0, text("Consumer", :black, :left, 10,"Computer Modern",rotation = 0))

    for i=10:12

        j = j + 3

        StatsPlots.annotate!(p1, j, -20.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 0), ylabel = "Maximum ingestion rate [1/year]")

        for k=1:length(fw_vorts.I_ind)

            par_ind = J_ind[k]

            par_ind_ODE = 0

            if(fw_vorts.I_ind[k][1]==i)

                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    index_ODE = findfirst(item -> item == fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    par_ind_ODE = J_ind_ODE[index_ODE]
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind],param_table_ODE[par_ind_ODE]])
                else
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind]])
                end

                StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(0,55),legend=:topleft)

                if(k == 30)
                    my_label = my_labels[1]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[1,par_ind],param_table[1,par_ind]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=5,
                    fontfamily="Computer Modern",
                    left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 40Plots.mm)
                

                if(k == 30)
                    my_label = my_labels[2]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[2,par_ind],param_table[2,par_ind]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=4)

                StatsPlots.annotate!(p1, j, -1.0, text(guild_names[fw_vorts.I_ind[k][2]], :black, :right, 8,"Computer Modern", rotation = 90))
                
                if(k == 30)
                    my_label = my_labels[3]
                else 
                    my_label = my_labels[4]
                end

                #plot ODE results
                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    StatsPlots.scatter!(p1,[j,j], [param_table_ODE[par_ind_ODE],param_table_ODE[par_ind_ODE]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                        label=my_label,markersize=3)
                end 

                if(k == 30)
                    my_label = my_labels[5]
                else 
                    my_label = my_labels[4]
                end

                if(J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]] > 0.0)
                    StatsPlots.scatter!(p1, [j,j], [J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]],J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]]], markersize=3, 
                        color =:white, markerstrokecolor = my_gray_1,label=my_label)
                end


                j = j +1

            end
        end
    end

    plot(p1,size=(750,500))

    savefig(figures_to*"/sensitivity_of_max_ing_rates_c.png")
    savefig(figures_to*"/sensitivity_of_max_ing_rates_c.svg")


    j = 1

    p1 = plot([0,0],[0,0],label="")

    label_ind = 1

    range_max = 0

    StatsPlots.annotate!(p1, -2, -5.0, text("Resource", :black, :left, 10,"Computer Modern",rotation = 0))

    StatsPlots.annotate!(p1, -2, -20.0, text("Consumer", :black, :left, 10,"Computer Modern",rotation = 0))

    for i=13:fw_vorts.nGuilds-1

        j = j + 3

        StatsPlots.annotate!(p1, j, -20.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 0), ylabel = "Maximum ingestion rate [1/year]")

        for k=1:length(fw_vorts.I_ind)

            par_ind = J_ind[k]

            par_ind_ODE = 0

            if(fw_vorts.I_ind[k][1]==i)

                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    index_ODE = findfirst(item -> item == fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    par_ind_ODE = J_ind_ODE[index_ODE]
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind],param_table_ODE[par_ind_ODE]])
                else
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind]])
                end

                StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(0,55),legend=:topright)

                if(k == 40)
                    my_label = my_labels[1]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[1,par_ind],param_table[1,par_ind]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=5,
                    fontfamily="Computer Modern",
                    left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 40Plots.mm)


                if(k == 40)
                    my_label = my_labels[2]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[2,par_ind],param_table[2,par_ind]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=4)

                StatsPlots.annotate!(p1, j, -1.0, text(guild_names[fw_vorts.I_ind[k][2]], :black, :right, 8,"Computer Modern", rotation = 90))
                
                if(k == 40)
                    my_label = my_labels[3]
                else 
                    my_label = my_labels[4]
                end

                #plot ODE results
                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    StatsPlots.scatter!(p1,[j,j], [param_table_ODE[par_ind_ODE],param_table_ODE[par_ind_ODE]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                        label=my_label,markersize=3)
                end 

                if(k == 40)
                    my_label = my_labels[5]
                else 
                    my_label = my_labels[4]
                end

                if(J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]] > 0.0)
                    StatsPlots.scatter!(p1, [j,j], [J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]],J_lit[fw_vorts.I_ind[k][1],fw_vorts.I_ind[k][2]]], markersize=3, 
                        color =:white, markerstrokecolor = my_gray_1,label=my_label)
                end


                j = j +1

            end
        end
    end

    plot(p1,size=(750,500))

    savefig(figures_to*"/sensitivity_of_max_ing_rates_d.png")
    savefig(figures_to*"/sensitivity_of_max_ing_rates_d.svg")



    ###### PLOT B0s ##########################################


    j = 1

    p1 = plot([0,0],[0,0],label="")

    label_ind = 1

    range_max = 0

    StatsPlots.annotate!(p1, -4, -17.0, text("Resource", :black, :left, 10,"Computer Modern",rotation = 0))

    StatsPlots.annotate!(p1, -4, -55.0, text("Consumer", :black, :left, 10,"Computer Modern",rotation = 0))


    for i=2:4

        j = j + 3

        StatsPlots.annotate!(p1, j-1, -56.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 0), ylabel = "Half-saturation constant [tonnes/km2]")

        for k=1:length(fw_vorts.I_ind)

            par_ind = B0_ind[k]

            par_ind_ODE = 0

            if(fw_vorts.I_ind[k][1]==i)

                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    index_ODE = findfirst(item -> item == fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    par_ind_ODE = J_ind_ODE[index_ODE]
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind],param_table_ODE[par_ind_ODE],max_data[fw_vorts.I_ind[k][2]]])
                else
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind],max_data[fw_vorts.I_ind[k][2]]])
                end

                StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(-1,230),legend=:topright)

                range_min = min_data[fw_vorts.I_ind[k][2]]
                range_max = max_data[fw_vorts.I_ind[k][2]]

                if(k == 1)
                    my_label = my_labels[6]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.plot!(p1,[j,j], [range_min,range_max], color=:black,linewidth=2, label = my_label, xticks = [],legend=:topright)

                if(k == 1)
                    my_label = my_labels[1]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[1,par_ind],param_table[1,par_ind]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=4,
                    fontfamily="Computer Modern",
                    left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 40Plots.mm)


                if(k == 1)
                    my_label = my_labels[2]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[2,par_ind],param_table[2,par_ind]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=3)

                StatsPlots.annotate!(p1, j, -5.0, text(guild_names[fw_vorts.I_ind[k][2]], :black, :right, 8,"Computer Modern", rotation = 90))
                
                if(k == 1)
                    my_label = my_labels[3]
                else 
                    my_label = my_labels[4]
                end

                #plot ODE results
                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    StatsPlots.scatter!(p1,[j,j], [param_table_ODE[par_ind_ODE],param_table_ODE[par_ind_ODE]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                        label=my_label,markersize=2)
                end 


                j = j +1

            end
        end
    end

    plot(p1,size=(350,500))

    savefig(figures_to*"/sensitivity_of_half_sat_const.png")
    savefig(figures_to*"/sensitivity_of_half_sat_const.svg")

    #####


    j = 1

    p1 = plot([0,0],[0,0],label="")

    label_ind = 1

    range_max = 0

    StatsPlots.annotate!(p1, -4, -10.0, text("Resource", :black, :left, 10,"Computer Modern",rotation = 0))

    StatsPlots.annotate!(p1, -4, -20.0, text("Consumer", :black, :left, 10,"Computer Modern",rotation = 0))


    for i=5:9

        j = j + 3

        StatsPlots.annotate!(p1, j-0.6, -20.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 0), ylabel = "Half-saturation constant [tonnes/km2]")

        for k=1:length(fw_vorts.I_ind)

            par_ind = B0_ind[k]

            par_ind_ODE = 0

            if(fw_vorts.I_ind[k][1]==i)

                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    index_ODE = findfirst(item -> item == fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    par_ind_ODE = J_ind_ODE[index_ODE]
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind],param_table_ODE[par_ind_ODE],max_data[fw_vorts.I_ind[k][2]]])
                else
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind],max_data[fw_vorts.I_ind[k][2]]])
                end

                StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(-1,50),legend=:topleft)

                range_min = min_data[fw_vorts.I_ind[k][2]]
                range_max = max_data[fw_vorts.I_ind[k][2]]

                if(k == 10)
                    my_label = my_labels[6]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.plot!(p1,[j,j], [range_min,range_max], color=:black,linewidth=2, label = my_label, xticks = [])

                if(k == 10)
                    my_label = my_labels[1]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[1,par_ind],param_table[1,par_ind]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=4,
                    fontfamily="Computer Modern",
                    left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 26Plots.mm, top_margin = 2Plots.mm)


                if(k == 10)
                    my_label = my_labels[2]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[2,par_ind],param_table[2,par_ind]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=3)

                StatsPlots.annotate!(p1, j, -5.0, text(guild_names[fw_vorts.I_ind[k][2]], :black, :right, 8,"Computer Modern", rotation = 90))
                
                if(k == 10)
                    my_label = my_labels[3]
                else 
                    my_label = my_labels[4]
                end

                #plot ODE results
                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)                    
                    StatsPlots.scatter!(p1,[j,j], [param_table_ODE[par_ind_ODE],param_table_ODE[par_ind_ODE]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                        label=my_label,markersize=2)
                end 


                j = j +1

            end
        end
    end

    plot(p1,size=(600,400))

    savefig(figures_to*"/sensitivity_of_half_sat_const_b.png")
    savefig(figures_to*"/sensitivity_of_half_sat_const_b.svg")

    ####

    j = 1

    p1 = plot([0,0],[0,0],label="")

    label_ind = 1

    range_max = 0

    StatsPlots.annotate!(p1, -2, -5.0, text("Resource", :black, :left, 10,"Computer Modern",rotation = 0))

    StatsPlots.annotate!(p1, -1.7, -20.0, text("Consumer", :black, :left, 10,"Computer Modern",rotation = 0))

    for i=10:12

        j = j + 3

        StatsPlots.annotate!(p1, j, -20.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 0), ylabel = "Half-saturation constant [tonnes/km2]")

        for k=1:length(fw_vorts.I_ind)

            par_ind = B0_ind[k]

            par_ind_ODE = 0

            if(fw_vorts.I_ind[k][1]==i)

                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    index_ODE = findfirst(item -> item == fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    par_ind_ODE = J_ind_ODE[index_ODE]
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind],param_table_ODE[par_ind_ODE]])
                else
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind]])
                end

                StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(0,45),legend=:topleft)

                range_min = min_data[fw_vorts.I_ind[k][2]]
                range_max = max_data[fw_vorts.I_ind[k][2]]

                if(k == 30)
                    my_label = my_labels[6]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.plot!(p1,[j,j], [range_min,range_max], color=:black,linewidth=2, label = my_label, xticks = [])

                if(k == 30)
                    my_label = my_labels[1]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[1,par_ind],param_table[1,par_ind]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=4,
                    fontfamily="Computer Modern",
                    left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 40Plots.mm)
                    

                if(k == 30)
                    my_label = my_labels[2]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[2,par_ind],param_table[2,par_ind]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=3)

                StatsPlots.annotate!(p1, j, -1.0, text(guild_names[fw_vorts.I_ind[k][2]], :black, :right, 8,"Computer Modern", rotation = 90))
                
                if(k == 30)
                    my_label = my_labels[3]
                else 
                    my_label = my_labels[4]
                end

                #plot ODE results
                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    StatsPlots.scatter!(p1,[j,j], [param_table_ODE[par_ind_ODE],param_table_ODE[par_ind_ODE]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                        label=my_label,markersize=2)
                end 


                j = j +1

            end
        end
    end

    plot(p1,size=(750,500))

    savefig(figures_to*"/sensitivity_of_half_sat_const_c.png")
    savefig(figures_to*"/sensitivity_of_half_sat_const_c.svg")


    j = 1

    p1 = plot([0,0],[0,0],label="")

    label_ind = 1

    range_max = 0

    StatsPlots.annotate!(p1, -1, -5.0, text("Resource", :black, :left, 10,"Computer Modern",rotation = 0))

    StatsPlots.annotate!(p1, -1, -20.0, text("Consumer", :black, :left, 10,"Computer Modern",rotation = 0))

    for i=13:fw_vorts.nGuilds-1

        j = j + 3

        StatsPlots.annotate!(p1, j+2, -20.0, text(guild_names[i], :black, :left, 8,"Computer Modern",rotation = 0), ylabel = "Half-saturation constant [tonnes/km2]")

        for k=1:length(fw_vorts.I_ind)

            par_ind = B0_ind[k]

            par_ind_ODE = 0

            if(fw_vorts.I_ind[k][1]==i)

                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    index_ODE = findfirst(item -> item == fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    par_ind_ODE = J_ind_ODE[index_ODE]
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind],param_table_ODE[par_ind_ODE]])
                else
                    range_max = maximum([param_table[1,par_ind],param_table[2,par_ind]])
                end

                StatsPlots.plot!(p1,[j,j], [0,range_max], color=my_gray_1,linewidth=7,label =:none, xticks = [],xlim=(2,j+3),ylim=(0,45),legend=:topright)

                range_min = min_data[fw_vorts.I_ind[k][2]]
                range_max = max_data[fw_vorts.I_ind[k][2]]

                if(k == 40)
                    my_label = my_labels[6]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.plot!(p1,[j,j], [range_min,range_max], color=:black,linewidth=2, label = my_label, xticks = [])

                if(k == 40)
                    my_label = my_labels[1]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[1,par_ind],param_table[1,par_ind]], color=:black, markerstrokecolor =:black, markershape=:rect,label=my_label,markersize=4,
                    fontfamily="Computer Modern",
                    left_margin = 12Plots.mm, right_margin = 0Plots.mm, bottom_margin = 40Plots.mm)
                    

                if(k == 40)
                    my_label = my_labels[2]
                else 
                    my_label = my_labels[4]
                end

                StatsPlots.scatter!(p1,[j,j], [param_table[2,par_ind],param_table[2,par_ind]], color=my_fuchsia, markerstrokecolor =my_fuchsia, markershape=:rect,label=my_label,markersize=3)

                StatsPlots.annotate!(p1, j, -1.0, text(guild_names[fw_vorts.I_ind[k][2]], :black, :right, 8,"Computer Modern", rotation = 90))
                
                if(k == 40)
                    my_label = my_labels[3]
                else 
                    my_label = my_labels[4]
                end

                #plot ODE results
                if in(fw_vorts.I_ind[k],fw_vorts_ODE.I_ind)
                    StatsPlots.scatter!(p1,[j,j], [param_table_ODE[par_ind_ODE],param_table_ODE[par_ind_ODE]], color=my_blue, markerstrokecolor = my_blue, markershape=:rect,
                        label=my_label,markersize=2)
                end 


                j = j +1

            end
        end
    end

    plot(p1,size=(750,500))

    savefig(figures_to*"/sensitivity_of_half_sat_const_d.png")
    savefig(figures_to*"/sensitivity_of_half_sat_const_d.svg")



end


end # module


# Example use:


### PREDICTED DYNAMICS

#FoodWebPlots.plot_and_analyze_predictions("results_vorts_environmental_noise/results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_extinction_penalty_const_using_last_estimates_fixed_q_0_3_separate_normal_noise_200_offspring_2000_iterations_MLE_discrete_BIOEN_DETRITAL_LOOP_CATCHES",
#     "post_pred","FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst",0.3)



### SENSITVITY OF PARAMETER ESTIMATES 

#FoodWebPlots.plot_sensitivity_of_parameter_estimates([
#             "results_vorts_environmental_noise/results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_extinction_penalty_const_using_last_estimates_fixed_q_0_3_separate_normal_noise_200_offspring_2000_iterations_MLE_discrete_BIOEN_DETRITAL_LOOP_CATCHES",
#             "results_vorts_environmental_noise/results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_using_last_estimates_fixed_q_0_3_separate_lognormal_noise_200_offspring_1000_iterations_MLE_discrete_BIOEN_DETRITAL_LOOP_CATCHES",
#             "results_vorts_observation_noise/results_vorts_BOUNDED_T_TLG24_1_extinction_penalty_const_fixed_q_0_3_separate_normal_noise_200_offspring_3000_iterations_MLE_ODE_disc_BIOEN_DETRITAL_LOOP_CATCHES"],
#             "results_vorts_environmental_noise")



### LOGNORMAL NOISE MODEL 

#FoodWebPlots.plot_and_analyze_predictions("results_vorts_environmental_noise/results_vorts_link_12_8_and_cannibalism_BOUNDED_T_TLG24_1_using_last_estimates_fixed_q_0_3_separate_lognormal_noise_200_offspring_1000_iterations_MLE_discrete_BIOEN_DETRITAL_LOOP_CATCHES",
#     "post_pred","FixedTypeDetritalLoopSeparateLognormalNoise",0.3)


