
# Maria Tirronen 2025
# https://orcid.org/0000-0001-6052-3186



# Set food web parameters.
module SetFoodWebParameters

using StatsPlots
using LinearAlgebra
using Random, Distributions
using CSV, DataFrames, Statistics
using DelimitedFiles

vorts_data_file = "vorts_biomass_time_series_with_interpolation.csv"
vorts_catch_data_file = "vorts_catch_time_series.csv"

# FOODWEB STRUCT 
mutable struct FoodWebDetritalLoop
        data_name::String

        model_type::String

        guild_names

        time 

        time_ind::Vector{Float64}

        ### the initial structure of the foodweb:

        # the number of guilds
        nGuilds::Int64

        # the number of feeding links
        nLinks::Int64

        # the number of producers
        nProd::Int64

        # the number of consumers
        nCons::Int64

        # the indeces of feeding links in the feeding matrix
        I_ind::Vector{CartesianIndex{2}}

        # the indices of producers and consumers
        prod_ind::Vector{Int64}
        cons_ind::Vector{Int64}

        # the indices of exploited populations
        exploited_pop_ind::Vector{Int64}

        # the index of detritus 
        detr_ind::Int64

        ### absolute catches 
        catches::Matrix{Float64}
		
        # the intrinsic growth rates of producers (r_i) and the metabolic rates of consumers (T_i)
        r::Vector{Float64}

        # lower bounds of the intrinsic growth rates of producers (r_i) and the metabolic rates of consumers (T_i)
        r_lower::Vector{Float64}

        # the fraction of exudation by a producer
        s::Float64

        # the carrying capacity for producer growth, system-wide
        K::Float64

        # 
        # J[i,j] is the maximum ingestion rate of prey item j by consumer i 
        J::Matrix{Float64}

        # assimilation efficiency
        # e[i,j] is the assimilation efficiency of consumer guild i feeding on guild j
        e::Matrix{Float64}

        # half-saturation constants 
        B0::Matrix{Float64}

        # functional response q parameters 
        q::Matrix{Float64}

        ### coefficients of variation
        cv::Vector{Float64} 

        ### standard deviations
        std::Vector{Float64}
end


# INTIALIZE A FOODWEB STRUCT WITH PARAMETERIZATION CORRESPONDING TO VORTS
function initialize_vorts_foodweb(model_type,q_init)
        include_catches = true

        data_name = "Vorts"
        
        # read the data
        csv_reader = CSV.File(vorts_data_file)
        vorts_names_raw = names(DataFrame(csv_reader))
        vorts_data_raw = transpose(Array(DataFrame(csv_reader)))[2:end,:]

        nGuilds = size(vorts_data_raw,1)-1

        time = Array(DataFrame(csv_reader))[:,1]     
        timeseries_length = length(time)
        time_ind = 1:timeseries_length

        vorts_data = zeros(nGuilds,timeseries_length)
        vorts_data[1,:] = vorts_data_raw[1,:] .+ vorts_data_raw[2,:]
        vorts_data[2:end,:] = vorts_data_raw[3:end,:] 
        
        vorts_names = copy(vorts_names_raw[2:end])
        vorts_names[1] = vorts_names_raw[1]
        vorts_names[2] = "Phytoplankton"
        vorts_names[3:end] .= vorts_names_raw[4:end]

        # read fish catches
        csv_reader_2 = CSV.File(vorts_catch_data_file)
        catch_data = DataFrame(csv_reader_2)
       
        catches=zeros(nGuilds,timeseries_length)
        catches[9,:]=catch_data[:,"Bream_catch"]
        catches[11,:]=catch_data[:,"Perch_catch"]
        catches[12,:]=catch_data[:,"Eel_catch"]
        catches[13,:]=catch_data[:,"Pikeperch_catch"]
        catches[14,:]=catch_data[:,"Pike_catch"]

        exploited_pop_ind=findall(x->(x.>0.0),catches[:,1])
        
        if(include_catches == false)
                catches .= 0
        end

        detr_ind = nGuilds

        # the interaction matrix, i.e., the feeding links for Võrts, according to Cremona et al. (2018)
        # if there is a 1 in the (i,j)th entry of the matrix, then species i feeds on resource j
        I::Matrix{Int64} = zeros(nGuilds,nGuilds)
        @views begin
        I[2:3,1:2].=1 # zooplanktons
        I[4,1]=1 # benthos
        I[5,2:4].=1 # Ruffe
        I[6,2:3].=1 # Roach
        I[7:9,2:4].=1 # Bleak, white bream, bream
        I[10,2:3].=1 # Smelt
        I[11,2:11].=1 # Perch # cannibalism INCLUDED
        I[12,2:10].=1 # Eel DIFFERS FROM CREMONA ET AL. !!!!
        #I[12,2:7].=1 # Eel
        #I[12,9:10].=1 # Eel
        I[13,2:11].=1 #Pikeperch
        I[14,2:3].=1
        I[14,5:14].=1 # Pike # cannibalism INCLUDED
        I[2:detr_ind-1,nGuilds].=1.0 # all other guilds than phytoplankton and detritus itself 
        #consume detritus
        end
        ### to study the impact of different feeding matrices, implement the feeding links above
    
        I_ind=findall(x->(x.>0.0),I)
            
        # the indices of producers and consumers
        prod_ind_=findall(i->i==0,sum(I,dims=2))
        cons_ind_=findall(i->(i>0),sum(I,dims=2))
        
        prod_ind=[prod_ind_[i][1] for i=1:length(prod_ind_)]
        cons_ind=[cons_ind_[i][1] for i=1:length(cons_ind_)]
    
        prod_ind = prod_ind[prod_ind .!= detr_ind]
        
        nLinks = length(I_ind)
        nProd = length(prod_ind)
        nCons = length(cons_ind)
            
        cv = zeros(nGuilds)
        std = zeros(nGuilds)
        
        if(cmp(model_type,"TypeIIIDetritalLoopRelativeNormalNoise")==0 || 
                cmp(model_type,"TypeIIIDetritalLoopSeparateNormalNoise")==0 || 
                cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoise")==0 ||
                cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConst")==0 ||
                cmp(model_type,"FixedTypeDetritalLoopSeparateNormalNoiseExtinctionPenaltyConstQuant")==0 ||
                cmp(model_type,"FixedTypeDetritalLoopRelativeNormalNoise")==0)

                for i=1:nGuilds
                        if(i==12 || i==15)
                                        data_i = vorts_data[i,1:end-1]
                        else 
                                        data_i = vorts_data[i,:]
                        end
                        cv[i] = 0.5.*Statistics.std(skipmissing(data_i))/Statistics.mean(skipmissing(data_i))
                        std[i] = 0.5.*Statistics.std(skipmissing(data_i))
                end
                        
        elseif( cmp(model_type,"FixedTypeDetritalLoopSeparateLognormalNoise")==0 ||
                cmp(model_type,"FixedTypeDetritalLoopRelativeLognormalNoise")==0 )
                
                for i=1:nGuilds
                        if(i==12 || i==15)
                                        data_i = log.(vorts_data[i,1:end-1])
                        else 
                                        data_i = log.(vorts_data[i,:])
                        end
                        cv[i] = 0.5.*Statistics.std(skipmissing(data_i))/abs(Statistics.mean(skipmissing(data_i))) # avoid negative coefficient of variation
                        std[i] = 0.5.*Statistics.std(skipmissing(data_i))
                end
        
        else                
                return 0
        end

        r = ones(nGuilds-1)

        # initialize
        r_lower = zeros(nGuilds-1)

        r_lower[2:4] .= 1.0
        r_lower[5] = 2.89
        r_lower[6] = 1.99
        r_lower[7] = 4.65
        r_lower[8] = 2.3
        r_lower[9] = 1.47
        r_lower[10] = 3.54
        r_lower[11] = 1.55
        r_lower[12] = 1.43
        r_lower[13] = 1.09
        r_lower[14] = 1.00

        s = 0.2

        K = maximum(vorts_data[prod_ind,:])
        
        J = ones(nGuilds,nGuilds)
        
        # init
        e = 0.85.*ones(nGuilds,nGuilds)   
        # e[i,j] is the assimilation efficiency of 
        # consumer guild i feeding on guild j
        e[:,1] .= 0.45 # phytoplankton as food
        e[:,end] .= 0.45 # detritus as food

        # init
        B0 = zeros(nGuilds,nGuilds)
        for i in 1:nGuilds
                if(i==12 || i==15)
                        data_i = vorts_data[i,1:end-1]
                else 
                        data_i = vorts_data[i,:]
                end
                B0[:,i] .= Statistics.mean(data_i)
        end
        
        q = q_init.*ones(nGuilds,nGuilds)

        return vorts_data, FoodWebDetritalLoop(data_name,model_type,vorts_names,
        time, time_ind, 
        nGuilds,
        nLinks, nProd, nCons,
        I_ind,
        prod_ind, cons_ind, exploited_pop_ind, detr_ind, 
        catches,
        r, r_lower, s, K,
        J, e, B0, q, 
        cv, std)

end


function copy_foodweb(fw_old)
        data_name = fw_old.data_name

        model_type = fw_old.model_type

        guild_names = fw_old.guild_names

        time = copy(fw_old.time)

        time_ind = copy(fw_old.time_ind)

        nGuilds = copy(fw_old.nGuilds)

        nLinks = copy(fw_old.nLinks)

        nProd = copy(fw_old.nProd)

        nCons = copy(fw_old.nCons)

        I_ind = copy(fw_old.I_ind)

        prod_ind = copy(fw_old.prod_ind)
        cons_ind = copy(fw_old.cons_ind)
        exploited_pop_ind = copy(fw_old.exploited_pop_ind)
        detr_ind = copy(fw_old.detr_ind)

        catches = copy(fw_old.catches)

        r = copy(fw_old.r)

        r_lower = copy(fw_old.r_lower)

        cv = copy(fw_old.cv)

        std = copy(fw_old.std)
        
        s = copy(fw_old.s)

        K = copy(fw_old.K)

        J = copy(fw_old.J)

        e = copy(fw_old.e)

        B0 = copy(fw_old.B0)
        
        q = copy(fw_old.q)

        return FoodWebDetritalLoop(data_name,model_type,guild_names,
                time, time_ind, 
                nGuilds,
                nLinks,nProd,nCons,
                I_ind,
                prod_ind, cons_ind, exploited_pop_ind, detr_ind,
                catches,
                r, r_lower, s, K,
                J,e,B0,q,
                cv, std)

end

end #module

