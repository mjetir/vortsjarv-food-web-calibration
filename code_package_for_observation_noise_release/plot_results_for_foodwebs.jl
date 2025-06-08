
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

gr(dpi=600)

include("foodweb_utils.jl")
include("foodweb_model_discretized.jl")
include("set_foodweb_parameters.jl")
include("foodweb_fitting_options.jl")

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
function plot_and_analyze_predictions_ODE_disc(results_from,figures_to,model_type,q_init)
        
    Random.seed!(8758743)

    vorts_data, fw_vorts = SetFoodWebParameters.initialize_vorts_foodweb_ODE_disc(model_type,q_init)
    f_opt = FittingOptions.initialize_MLE_options_ODE_disc(fw_vorts,vorts_data,"",1,1)    
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

    fw_pred = SetFoodWebParameters.copy_foodweb_ODE(fw_vorts) 
    
    min_neg_likelihood = 0

    println("")

    println(model_type)

    println("q (init/fixed): ")
    println(q_init)

    ### fixed q
    if(         cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)
    
                transformed_param = FoodWebUtils.rho_transform(param_est)
                            
                fw_pred.r = transformed_param[1:fw_vorts.nGuilds - 1]
    
                fw_pred.J[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds:fw_vorts.nGuilds + fw_vorts.nLinks - 1]
                fw_pred.B0[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + fw_vorts.nLinks:fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]

                fw_pred.std = transformed_param[fw_vorts.nGuilds + 2*fw_vorts.nLinks:2*fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]
    
                fw_pred.K = transformed_param[2*fw_vorts.nGuilds + 2*fw_vorts.nLinks]                        
                
                fw_pred.u0 = transformed_param[2*fw_vorts.nGuilds + 2*fw_vorts.nLinks+1:end]

    elseif(cmp(model_type,"FixedTypeDetritalLoopRelativeNormalNoiseExtinctionPenaltyConst")==0)
    
                transformed_param = FoodWebUtils.rho_transform(param_est)
                            
                fw_pred.r = transformed_param[1:fw_vorts.nGuilds - 1]
    
                fw_pred.J[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds:fw_vorts.nGuilds + fw_vorts.nLinks - 1]
                fw_pred.B0[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + fw_vorts.nLinks:fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]

                fw_pred.cv = transformed_param[fw_vorts.nGuilds + 2*fw_vorts.nLinks:2*fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]
    
                fw_pred.K = transformed_param[2*fw_vorts.nGuilds + 2*fw_vorts.nLinks]                        
                
                fw_pred.u0 = transformed_param[2*fw_vorts.nGuilds + 2*fw_vorts.nLinks+1:end]


    elseif(cmp(model_type,"TypeIIIDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0)

                transformed_param = FoodWebUtils.rho_transform(param_est)
                            
                fw_pred.r = transformed_param[1:fw_vorts.nGuilds - 1]

                fw_pred.J[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds:fw_vorts.nGuilds + fw_vorts.nLinks - 1]
                fw_pred.B0[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + fw_vorts.nLinks:fw_vorts.nGuilds + 2*fw_vorts.nLinks - 1]
                fw_pred.q[fw_vorts.I_ind] = transformed_param[fw_vorts.nGuilds + 2*fw_vorts.nLinks:fw_vorts.nGuilds + 3*fw_vorts.nLinks - 1]

                fw_pred.std = transformed_param[fw_vorts.nGuilds + 3*fw_vorts.nLinks:2*fw_vorts.nGuilds + 3*fw_vorts.nLinks - 1]

                fw_pred.K = transformed_param[end]                        
                
    else 
        println("Model not defined.")
        return 0
    end


    println("")

    prediction_mean =  f_opt.model(fw_pred,f_cont) 

    println("CV of internal fluctuations:")
    println(sum(Statistics.mean(prediction_mean,dims=2)./Statistics.std(prediction_mean,dims=2))/size(prediction_mean,1))

    min_neg_likelihood = minimum(losses)

    if(cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0 ||
        cmp(model_type,"TypeIIIDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0 ||
        cmp(model_type,"FixedTypeDetritalLoopRelativeNormalNoiseExtinctionPenaltyConst")==0 )
        
        min_neg_likelihood -= 100.0*sum(prediction_mean .<= 0.0)
        
    end

    if(cmp(model_type,"TypeIIIDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0 ||
        cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0  )

        posterior_predictives = [Normal(prediction_mean[i,t],fw_pred.std[i]) 
            for i=1:fw_vorts.nGuilds,t=1:length(fw_vorts.time_ind)]

    elseif(cmp(model_type,"FixedTypeDetritalLoopRelativeNormalNoiseExtinctionPenaltyConst")==0)

        posterior_predictives = [Normal(prediction_mean[i,t],fw_pred.cv[i]*prediction_mean[i,t]) 
            for i=1:fw_vorts.nGuilds,t=1:length(fw_vorts.time_ind)]   
            
    else  
        posterior_predictives = 0                
    end 

    lower_quantiles = quantile.(posterior_predictives,0.05)
    upper_quantiles = quantile.(posterior_predictives,0.95)

    println("")
    println("Number of negative mean predictions guild-wise:")
    neg_pred = 0
    neg_excl_det = 0
    for j=1:fw_vorts.nGuilds
        print("Guild"*string(j)*":")
        neg = sum(prediction_mean[j,:] .<= 0.0)
        println(neg)
        neg_pred += neg
        if(neg > 0)
            println(prediction_mean[j,:] .<= 0.0)
            println(prediction_mean[j,:])
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

    data_for_pred = vorts_data
    non_missing = data_for_pred .>= 0.0

    pred = prediction_mean

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

    ylims_upper = [210,16,4,40,12.5,6,3,2.3,22,0.5,1.2,4,7,1.5,40];
    ylims_lower = [-53,-5,-0.3,-18.0,-10,-2,-1.5,-0.8,-15,-0.4,-0.3,-1.5,-3,-0.5,-10];

    guild_names = ["Phytopl.","Protozoopl.","Metazoopl.","Benthos","Ruffe","Roach","Bleak","White bream","Bream",
    "Smelt","Perch","Eel","Pikeperch","Pike","Detritus"]

    i=1 

    plots[i] = StatsPlots.plot(fw_vorts.time, lower_quantiles[i,:], fillrange = upper_quantiles[i,:],
    ylim=(ylims_lower[i],ylims_upper[i]), 
    xlim = (1994.5,2012.5),
    fillalpha = 0.28, linealpha=0, color = my_blue , 
    label = "",  
    ylabel=guild_names[i],
    title = "Observation noise",
    fontfamily=:"Computer Modern",
    left_margin = 20Plots.mm, right_margin = 1Plots.mm)
    StatsPlots.plot!(plots[i],fw_vorts.time,prediction_mean[i,:],color=:black, label = "", linestyle = :dash, linewidth=2.5)
    StatsPlots.scatter!(plots[i],fw_vorts.time,vorts_data[i,:], label = "", color=my_blue)
    StatsPlots.plot!(plots[i],fw_vorts.time,zeros(length(fw_vorts.time_ind)),color=:white, label = "", linewidth=2.0)

    for i=2:fw_vorts.nGuilds

        plots[i] = StatsPlots.plot(fw_vorts.time, lower_quantiles[i,:], fillrange = upper_quantiles[i,:],
        ylim=(ylims_lower[i],ylims_upper[i]),
        xlim = (1994.5,2012.5),
        fillalpha = 0.28, linealpha=0, color = my_blue , 
        label = "",  
        ylabel=guild_names[i],
        fontfamily=:"Computer Modern",
        left_margin = 20Plots.mm, right_margin = 1Plots.mm)
        StatsPlots.plot!(plots[i],fw_vorts.time,prediction_mean[i,:],color=:black, label = "", linestyle = :dash, linewidth=2.5)
        StatsPlots.scatter!(plots[i],fw_vorts.time,vorts_data[i,:], label = "", color=my_blue)
        StatsPlots.plot!(plots[i],fw_vorts.time,zeros(length(fw_vorts.time_ind)),color=:white, label = "", linewidth=2.0)

    end

    imputation_years = findall(i->in(i,[1999,2001,2002,2006,2010,2011]),fw_vorts.time)

    StatsPlots.scatter!(plots[fw_vorts.nGuilds],fw_vorts.time[imputation_years],vorts_data[fw_vorts.nGuilds,imputation_years], color=:white, label = "")

    i = fw_vorts.nGuilds

    plots[i+1] = StatsPlots.plot(fw_vorts.time, lower_quantiles[i,:], fillrange = upper_quantiles[i,:], 
        fillalpha = 0.28, linealpha=0, color = my_blue, 
        label = "Prediction of observed biomass",  
        ylabel="",xlabel="",
        fontfamily=:"Computer Modern",
        left_margin = 20Plots.mm, right_margin = 1Plots.mm)
    StatsPlots.plot!(plots[i+1],fw_vorts.time,prediction_mean[i,:],color=:black, linestyle = :dash, label = "Prediction of true biomass", linewidth=2.5)
    StatsPlots.scatter!(plots[i+1],fw_vorts.time,vorts_data[i,:], label = "Recorded data", color=my_blue)


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


end # module


## Example use:

#FoodWebPlots.plot_and_analyze_predictions_ODE_disc("results_vorts_observation_noise/results_vorts_BOUNDED_T_TLG24_1_extinction_penalty_const_fixed_q_0_3_separate_normal_noise_200_offspring_3000_iterations_MLE_ODE_disc_BIOEN_DETRITAL_LOOP_CATCHES",
#     "post_pred","FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst",0.3)

#FoodWebPlots.plot_and_analyze_predictions_ODE_disc("results_vorts_observation_noise/results_vorts_BOUNDED_T_TLG24_1_extinction_penalty_const_separate_normal_noise_200_offspring_1000_iterations_MLE_ODE_disc_BIOEN_DETRITAL_LOOP_CATCHES",
#     "post_pred","TypeIIIDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst",0.3)


