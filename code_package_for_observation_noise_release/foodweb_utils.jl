
# Maria Tirronen 2025
# https://orcid.org/0000-0001-6052-3186



# Some useful functions
module FoodWebUtils

using DelimitedFiles


# TRANSFORMATION TO KEEP THE PARAMETERS STRICTLY POSITIVE
function rho_transform(x)
    return [if (x[i] < 500.0) log(1.0 + exp(x[i]))[1]
                else x[i] end for i=1:length(x)]
end

# INVERSE TRANSFORMATION
function inverse_rho_transform(x)
    return [if (x[i] < 500.0) log(exp(x[i]) - 1.0) # log(exp(500)+1) ~ 500
            else x[i] end for i=1:length(x)] 
end

# WRITE INITIAL PARAMETER VALUES AND OTHER SETTINGS TO FILE
function write_initial_parameters_to_file(f_opt,folder_name)
    println("Writing parameter estimates to "*folder_name)

    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    open(folder_name*"/initial_parameter_values.txt","w") do io
        writedlm(io,f_opt.p_init)
    end    

    open(folder_name*"/param_min.txt","w") do io
        writedlm(io,f_opt.param_min)
    end        
    open(folder_name*"/param_max.txt","w") do io
        writedlm(io,f_opt.param_max)
    end    

    open(folder_name*"/bc_min.txt","w") do io
        writedlm(io,f_opt.bc_min)
    end        
    open(folder_name*"/bc_max.txt","w") do io
        writedlm(io,f_opt.bc_max)
    end    

    open(folder_name*"/settings.txt","w") do f
        println(f,"Model: "*string(f_opt.model))
        println(f,"Model type: "*string(f_opt.model_type))
        println(f,"Loss function: "*string(f_opt.loss_function))
        println(f,"Extinction penalty threshold, if used: "*string(f_opt.extinction_penalty_threshold))
        println(f,"q, if fixed, otherwise initial value: "*string(f_opt.q_init))
        println(f,"Number of restarts: "*string(f_opt.n_iter_restart))
        println(f,"Number of offspring: "*string(f_opt.n_offspring))
        println(f,"Method: "*f_opt.method)
    end
end


# WRITE PARAMETER ESTIMATES TO FILE
function write_parameters_to_file(results,folder_name)
    println("Writing parameter estimates to "*folder_name)

    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    open(folder_name*"/parameter_estimates.txt","w") do io
        writedlm(io,results.param_final)
    end    
    open(folder_name*"/parameters_in_iteration.txt","w") do io
        writedlm(io,results.param)
    end    
end


# WRITE LOSS FUNCTION VALUES TO FILE
function write_losses_to_file(results,folder_name)
    println("Writing loss function values to "*folder_name)

    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    open(folder_name*"/losses.txt","w") do io
        writedlm(io,results.loss)
    end    
end


# WRITE SIMULATED BIOMASSES TO FILE
function write_data_to_file(years, biomasses,folder_name,file_name)
    println("Writing data to "*folder_name)

    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    open(folder_name*"/"*file_name*".txt","w") do f
        print(f,"Year;")
        for i in 1:size(biomasses,1)-1
            print(f,"Guild_"*string(i)*";")
        end
        println(f,"Guild_"*string(size(biomasses,1)))

        for j in 1:size(biomasses,2)
            print(f,string(years[j])*";")
            for i in 1:size(biomasses,1)-1
                print(f,string(biomasses[i,j])*";")
            end
            println(f,string(biomasses[end,j]))
        end
    end
end


end # module
