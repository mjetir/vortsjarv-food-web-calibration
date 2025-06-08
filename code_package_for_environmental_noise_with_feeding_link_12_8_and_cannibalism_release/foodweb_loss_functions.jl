
# Maria Tirronen 2025
# https://orcid.org/0000-0001-6052-3186


module LossFunctions

using Random, Distributions, Statistics

include("foodweb_utils.jl")


# LOSS FUNCTIONS FOR MAXIMUM LIKELIHOOD ESTIMATION

@views function loss_MLE_TypeIII_detrital_loop_carrying_capacity_relative_normal_noise(par_tmp,fw_tmp,f_opt,f_cont)
    
    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    fw_tmp.q[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:fw_tmp.nGuilds + 3*fw_tmp.nLinks - 1]

    fw_tmp.cv = f_cont.transf_par_tmp[fw_tmp.nGuilds + 3*fw_tmp.nLinks:2*fw_tmp.nGuilds + 3*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[end]
    
    f_cont.model_tmp =  f_opt.model(f_opt.training_data,fw_tmp,f_cont)

    return -sum([logpdf(
            Normal(f_cont.model_tmp[i,t],fw_tmp.cv[i]*f_opt.training_data[i,t]), 
            f_opt.training_data[i,t+1]) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-2]) - sum([logpdf(
                Normal(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end-1])],fw_tmp.cv[i]*f_opt.training_data[i,Int(fw_tmp.time_ind[end-1])]), 
                f_opt.training_data[i,Int(fw_tmp.time_ind[end])]) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]])
end


@views function loss_MLE_TypeIII_detrital_loop_carrying_capacity_separate_normal_noise(par_tmp,fw_tmp,f_opt,f_cont)
    
    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    fw_tmp.q[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:fw_tmp.nGuilds + 3*fw_tmp.nLinks - 1]

    fw_tmp.std = f_cont.transf_par_tmp[fw_tmp.nGuilds + 3*fw_tmp.nLinks:2*fw_tmp.nGuilds + 3*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[end]
    
    f_cont.model_tmp =  f_opt.model(f_opt.training_data,fw_tmp,f_cont)

    return -sum([logpdf(
            Normal(f_cont.model_tmp[i,t],fw_tmp.std[i]), 
            f_opt.training_data[i,t+1]) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-2]) - sum([logpdf(
                Normal(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end-1])],fw_tmp.std[i]), 
                f_opt.training_data[i,Int(fw_tmp.time_ind[end])]) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]])
end


@views function loss_MLE_FixedType_detrital_loop_carrying_capacity_relative_normal_noise(par_tmp,fw_tmp,f_opt,f_cont)
    
    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    # do not update q 

    fw_tmp.std = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:2*fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[end]
    
    f_cont.model_tmp =  f_opt.model(f_opt.training_data,fw_tmp,f_cont)

    return -sum([logpdf(
            Normal(f_cont.model_tmp[i,t],fw_tmp.cv[i]*f_opt.training_data[i,t]), 
            f_opt.training_data[i,t+1]) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-2]) - sum([logpdf(
                Normal(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end-1])],fw_tmp.cv[i]*f_opt.training_data[i,Int(fw_tmp.time_ind[end-1])]), 
                f_opt.training_data[i,Int(fw_tmp.time_ind[end])]) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]])
end


@views function loss_MLE_FixedType_detrital_loop_carrying_capacity_separate_normal_noise(par_tmp,fw_tmp,f_opt,f_cont)
    
    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    # do not update q 

    fw_tmp.std = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:2*fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[end]
    
    f_cont.model_tmp =  f_opt.model(f_opt.training_data,fw_tmp,f_cont)

    return -sum([logpdf(
            Normal(f_cont.model_tmp[i,t],fw_tmp.std[i]), 
            f_opt.training_data[i,t+1]) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-2]) - sum([logpdf(
                Normal(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end-1])],fw_tmp.std[i]), 
                f_opt.training_data[i,Int(fw_tmp.time_ind[end])]) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]])
end


@views function loss_MLE_FixedType_detrital_loop_carrying_capacity_separate_normal_noise_extinction_penalty_const(par_tmp,fw_tmp,f_opt,f_cont)
    
    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    # do not update q 

    fw_tmp.std = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:2*fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[end]
    
    f_cont.model_tmp =  f_opt.model(f_opt.training_data,fw_tmp,f_cont)

    return -sum([logpdf(
            Normal(f_cont.model_tmp[i,t],fw_tmp.std[i]), 
            f_opt.training_data[i,t+1]) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-2]) - sum([logpdf(
            Normal(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end-1])],fw_tmp.std[i]), 
            f_opt.training_data[i,Int(fw_tmp.time_ind[end])]) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]]) + 
            100.0*sum(f_cont.model_tmp .<= 0.0)
end


@views function loss_MLE_FixedType_detrital_loop_carrying_capacity_separate_normal_noise_extinction_penalty_const_quant(par_tmp,fw_tmp,f_opt,f_cont)
    
    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    # do not update q 

    fw_tmp.std = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:2*fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[end]
    
    f_cont.model_tmp =  f_opt.model(f_opt.training_data,fw_tmp,f_cont)

    f_cont.pred_distr = [Normal(f_cont.model_tmp[i,t],fw_tmp.std[i]) 
                            for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)]

    f_cont.pred_quantiles = [quantile(f_cont.pred_distr[i,t],f_opt.extinction_penalty_threshold) 
                        for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)] # takes all predictions in model_tmp
    
    return -sum([logpdf(
            Normal(f_cont.model_tmp[i,t],fw_tmp.std[i]), 
            f_opt.training_data[i,t+1]) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-2]) - sum([logpdf(
            Normal(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end-1])],fw_tmp.std[i]), 
            f_opt.training_data[i,Int(fw_tmp.time_ind[end])]) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]]) + 
            100.0*sum(f_cont.pred_quantiles .<= 0.0)
end


@views function loss_MLE_FixedType_detrital_loop_carrying_capacity_relative_lognormal_noise(par_tmp,fw_tmp,f_opt,f_cont)
    
    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    # do not update q 

    fw_tmp.std = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:2*fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[end]
    
    f_cont.model_tmp =  f_opt.model(f_opt.training_data,fw_tmp,f_cont)

    f_cont.model_tmp[12,3] = max(0.00000001,f_cont.model_tmp[12,3]) # truncate eel biomass to be positive
    f_cont.model_tmp[12,13] = max(0.00000001,f_cont.model_tmp[12,13]) # truncate eel biomass to be positive

    if sum(f_cont.model_tmp[:,1:end-1] .<= 0.0)>0 # do not consider the last predictions
        return Inf
    else 
        return -sum([logpdf(
            Normal(log(f_cont.model_tmp[i,t]),fw_tmp.cv[i]*abs(log(f_opt.training_data[i,t]))), 
            log(f_opt.training_data[i,t+1])) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-2]) - sum([logpdf(
            Normal(log(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end-1])]),fw_tmp.cv[i]*abs(log(f_opt.training_data[i,Int(fw_tmp.time_ind[end-1])]))), 
            log(f_opt.training_data[i,Int(fw_tmp.time_ind[end])])) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]])
    end
end


@views function loss_MLE_FixedType_detrital_loop_carrying_capacity_separate_lognormal_noise(par_tmp,fw_tmp,f_opt,f_cont)
    
    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    # do not update q 

    fw_tmp.std = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:2*fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[end]
    
    f_cont.model_tmp =  f_opt.model(f_opt.training_data,fw_tmp,f_cont)

    f_cont.model_tmp[12,3] = max(0.00000001,f_cont.model_tmp[12,3]) # truncate eel biomass to be positive
    f_cont.model_tmp[12,13] = max(0.00000001,f_cont.model_tmp[12,13]) # truncate eel biomass to be positive

    if sum(f_cont.model_tmp[:,1:end-1] .<= 0.0)>0 # do not consider the last predictions
        return Inf
    else 
        return -sum([logpdf(
            Normal(log(f_cont.model_tmp[i,t]),fw_tmp.std[i]), 
            log(f_opt.training_data[i,t+1])) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-2]) - sum([logpdf(
                Normal(log(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end-1])]),fw_tmp.std[i]), 
                log(f_opt.training_data[i,Int(fw_tmp.time_ind[end])])) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]])
    end
end


end # module


