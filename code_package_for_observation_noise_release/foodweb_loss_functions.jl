
# Maria Tirronen 2025
# https://orcid.org/0000-0001-6052-3186



module LossFunctions

using Random, Distributions, Statistics

include("foodweb_utils.jl")



### LOSS FUNCTIONS FOR THE DISCRETIZED ODE MODEL 


@views function loss_MLE_FixedType_detrital_loop_carrying_capacity_separate_normal_noise_ODE_disc_extinction_penalty_const(par_tmp,fw_tmp,f_opt,f_cont)

    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    # do not update q 

    fw_tmp.std = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:2*fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[2*fw_tmp.nGuilds + 2*fw_tmp.nLinks]

    fw_tmp.u0 = f_cont.transf_par_tmp[2*fw_tmp.nGuilds + 2*fw_tmp.nLinks+1:end]
    
    f_cont.model_tmp =  f_opt.model(fw_tmp,f_cont)

    return -sum([logpdf(
            Normal(f_cont.model_tmp[i,t],fw_tmp.std[i]), 
            f_opt.training_data[i,t]) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-1]) - sum([logpdf(
            Normal(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end])],fw_tmp.std[i]), 
            f_opt.training_data[i,Int(fw_tmp.time_ind[end])]) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]]) + 
            100.0*sum(f_cont.model_tmp .<= 0.0) 
end



@views function loss_MLE_FixedType_detrital_loop_carrying_capacity_relative_normal_noise_ODE_disc_extinction_penalty_const(par_tmp,fw_tmp,f_opt,f_cont)

    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    # do not update q 

    fw_tmp.cv = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:2*fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[2*fw_tmp.nGuilds + 2*fw_tmp.nLinks]

    fw_tmp.u0 = f_cont.transf_par_tmp[2*fw_tmp.nGuilds + 2*fw_tmp.nLinks+1:end]
    
    f_cont.model_tmp =  f_opt.model(fw_tmp,f_cont)

    return -sum([logpdf(
                Normal(f_cont.model_tmp[i,t],fw_tmp.cv[i]*f_opt.training_data[i,t]), 
                f_opt.training_data[i,t]) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-1]) - sum([logpdf(
                    Normal(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end])],fw_tmp.cv[i]*f_opt.training_data[i,Int(fw_tmp.time_ind[end])]), 
                    f_opt.training_data[i,Int(fw_tmp.time_ind[end])]) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]]) + 
                    100.0*sum(f_cont.model_tmp .<= 0.0) 
end



@views function loss_MLE_TypeIII_detrital_loop_carrying_capacity_separate_normal_noise_ODE_disc_extinction_penalty_const(par_tmp,fw_tmp,f_opt,f_cont)

    f_cont.transf_par_tmp = FoodWebUtils.rho_transform(par_tmp)

    fw_tmp.r = f_cont.transf_par_tmp[1:fw_tmp.nGuilds - 1]

    fw_tmp.J[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds:fw_tmp.nGuilds + fw_tmp.nLinks - 1]
    fw_tmp.B0[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + fw_tmp.nLinks:fw_tmp.nGuilds + 2*fw_tmp.nLinks - 1]
    fw_tmp.q[fw_tmp.I_ind] = f_cont.transf_par_tmp[fw_tmp.nGuilds + 2*fw_tmp.nLinks:fw_tmp.nGuilds + 3*fw_tmp.nLinks - 1]

    fw_tmp.std = f_cont.transf_par_tmp[fw_tmp.nGuilds + 3*fw_tmp.nLinks:2*fw_tmp.nGuilds + 3*fw_tmp.nLinks - 1]

    fw_tmp.K = f_cont.transf_par_tmp[2*fw_tmp.nGuilds + 3*fw_tmp.nLinks]

    fw_tmp.u0 = f_cont.transf_par_tmp[2*fw_tmp.nGuilds + 3*fw_tmp.nLinks+1:end]
    
    f_cont.model_tmp =  f_opt.model(fw_tmp,f_cont)

    return -sum([logpdf(
            Normal(f_cont.model_tmp[i,t],fw_tmp.std[i]), 
            f_opt.training_data[i,t]) for i=1:fw_tmp.nGuilds,t=1:length(fw_tmp.time_ind)-1]) - sum([logpdf(
            Normal(f_cont.model_tmp[i,Int(fw_tmp.time_ind[end])],fw_tmp.std[i]), 
            f_opt.training_data[i,Int(fw_tmp.time_ind[end])]) for i in [1,2,3,4,5,6,7,8,9,10,11,13,14]]) + 
            100.0*sum(f_cont.model_tmp .<= 0.0) 
end

end # module


