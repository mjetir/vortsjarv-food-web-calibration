
# Maria Tirronen 2025
# https://orcid.org/0000-0001-6052-3186


module DiscreteFoodWebModel

using LinearAlgebra
using Random, Distributions
using CSV, DataFrames, Statistics

include("set_foodweb_parameters.jl")


# DISCRETE MODEL FOR FOODWEB WITH NONLINEAR FUNCTIONAL RESPONSES AND DETRITAL LOOP, 
# LIMITED CARRYING CAPACITY FOR PRODUCER GROWTH
#
# takes the current values of biomasses
# returns the means of the next biomass values
# delta t = 1
# 
@views function bioenergetic_discretized_detrital_loop(u,p,cont)

    ### the means of the biomass values at the next step
    u_next = copy(u)
    u_next .= 0 

    cont.denom_sum .= 0.0
    cont.F .= 0.0

    # functional responses
    @inbounds for t_ in p.time_ind[1:end-1]
        t = Int(t_)
        @inbounds for j in p.cons_ind 
            @inbounds for i in 1:p.nGuilds
                if in(CartesianIndex(j,i),p.I_ind) 
                    cont.denom_sum[j,t] += (u[i,t]/p.B0[j,i])^(1.0 + p.q[j,i])
                end
            end
            @inbounds for i in 1:p.nGuilds
                if in(CartesianIndex(j,i),p.I_ind) 
                    cont.F[j,i,t] = ((u[i,t]/p.B0[j,i])^(1.0 + p.q[j,i]))/(1.0 + cont.denom_sum[j,t])
                end
            end
        end
    end 

    # producers:
    @inbounds for i in p.prod_ind 
        @inbounds for t_ in p.time_ind[1:end-1]
                t = Int(t_)
                u_next[i,t] = (1.0 + ( p.r[i]*(1.0 - u[i,t]/p.K)*(1.0 - p.s[i]) ) )*u[i,t]
                @inbounds for j in p.cons_ind 
                    if in(CartesianIndex(j,i),p.I_ind) 
                            u_next[i,t] = u_next[i,t] - u[j,t]*p.J[j,i]*cont.F[j,i,t] # the observations are always non-negative
                    end
                end
        end
    end
        
    # consumers: 
    @inbounds for j in p.cons_ind
        @inbounds for t_ in p.time_ind[1:end-1]
            t = Int(t_)
            u_next[j,t] = (1.0 - (p.r_lower[j] + p.r[j]))*u[j,t] - p.catches[j,t]

            # gain from consumption
            @inbounds for i in 1:p.nGuilds
                if in(CartesianIndex(j,i),p.I_ind)                         
                        # to include activity respiration, implement it here
                        u_next[j,t] = u_next[j,t] + u[j,t]*p.e[j,i]*p.J[j,i]*cont.F[j,i,t]
                end
            end
            # loss to consumption
            @inbounds for k in p.cons_ind
                if in(CartesianIndex(k,j),p.I_ind) 
                        u_next[j,t] = u_next[j,t] - u[k,t]*p.J[k,j]*cont.F[k,j,t] 
                end
            end            
        end
    end    

    # detritus:
    @inbounds for t_ in p.time_ind[1:end-1]
        t = Int(t_)
        u_next[p.detr_ind,t] = u[p.detr_ind,t] 
        # gain from exudation by producers
        @inbounds for i in p.prod_ind 
            u_next[p.detr_ind,t] = u_next[p.detr_ind,t] + p.r[i]*(1.0 - u[i,t]/p.K)*p.s[i]*u[i,t]
        end
        # gain from egested resources by consumers
        @inbounds for j in p.cons_ind 
            @inbounds for i in 1:p.nGuilds-1 # does not include detritus
                if in(CartesianIndex(j,i),p.I_ind) 
                    u_next[p.detr_ind,t] = u_next[p.detr_ind,t] + u[j,t]*(1.0 - p.e[j,i])*p.J[j,i]*cont.F[j,i,t]
                end
            end
        end
        # loss to consumption by detritivores
        @inbounds for j in p.cons_ind
            if in(CartesianIndex(j,p.detr_ind),p.I_ind) 
                u_next[p.detr_ind,t] = u_next[p.detr_ind,t] - u[j,t]*p.J[j,p.detr_ind]*cont.F[j,p.detr_ind,t] 
            end
        end
    end

    return u_next
end


end # module
