# evaluateLogPdf_joint.jl
# Joint log-posterior with 16 learned σ's (8 for CO₂, 8 for Temperature)

using LinearAlgebra
using Distributions

function evaluateLogPdf_joint(P, f_co2_data, f_temp_data, time_window_size)
    # ---- Slice parameter vector P (layout matches main.jl) ----
    # Airflow (shared, CO₂ forward)
    P_air   = P[1:13]        # 13 airflow/occupancy params (8 n_ppl + 5 AF)
    C_air0  = P[14:21]       # 8 CO₂ initial conditions

    # Thermal block (19 R + 8 C)
    P_therm = P[22:48]       # 27 thermal params (19 resistances + 8 capacities)
    C_temp0 = P[49:56]       # 8 initial temperatures

    # NEW: noise σ vectors (CO₂ 8, Temp 8)
    σ_co2_vec = P[57:64]     # matches idx_sigma_air in main.jl
    σ_t_vec   = P[65:72]     # matches idx_sigma_tl  in main.jl

    if any(x -> !isfinite(x) || x <= 0.0, σ_co2_vec) || any(x -> !isfinite(x) || x <= 0.0, σ_t_vec)
        return -Inf
    end

    # ---- Forward models ----
    tspan = (0.0, time_window_size)

    # CO₂
    sol_air      = solutionForward(P_air, C_air0, tspan)
    U_air_model  = reduce(hcat, sol_air.u)         # 8 x N
    u_model_air  = vec(U_air_model)                # 8N

    # CO₂ data: accept either 8×N matrix (new) or Vector{SVector{8}} (old)
    U_air_data   = f_co2_data isa AbstractMatrix ? f_co2_data : reduce(hcat, f_co2_data)
    u_data_air   = vec(U_air_data)

    # Thermal (note: thermal model takes [R,C] then [airflow/occupancy])
    P_therm_full = vcat(P_therm, P_air)
    sol_temp     = solutionForward_thermal(P_therm_full, C_temp0, tspan)
    U_temp_model = reduce(hcat, sol_temp.u)        # 8 x N
    u_model_temp = vec(U_temp_model)               # 8N

    # Temperature data: same
    U_temp_data  = f_temp_data isa AbstractMatrix ? f_temp_data : reduce(hcat, f_temp_data)
    u_data_temp  = vec(U_temp_data)                # 8N

    # ---- Likelihood with node-specific σ's (constant over time per node) ----
    # Repeat each node's σ across all N time points to match vec() ordering
    N_air_cols   = size(U_air_model, 2)
    N_temp_cols  = size(U_temp_model, 2)

    σ_air_full   = repeat(σ_co2_vec, N_air_cols)   # length 8N
    σ_temp_full  = repeat(σ_t_vec,   N_temp_cols)  # length 8N

    # Independent Normals per element (heteroscedastic)
    loglik_air   = sum(logpdf.(Normal.(u_model_air, σ_air_full),  u_data_air))
    loglik_temp  = sum(logpdf.(Normal.(u_model_temp, σ_temp_full), u_data_temp))

    # ---- Priors ----
    logprior = compute_joint_prior(P, f_co2_data, f_temp_data)

    return (isfinite(logprior) ? logprior : -Inf) + loglik_air + loglik_temp
end

function compute_joint_prior(P, f_co2_data, f_temp_data)
    logp = 0.0

    # Slices for readability
    σ_co2_vec = P[57:64]
    σ_t_vec   = P[65:72]
    
    if any(x -> !isfinite(x) || x <= 0.0, σ_co2_vec) || any(x -> !isfinite(x) || x <= 0.0, σ_t_vec)
        return -Inf
    end
    # ---- Airflow/occupancy priors (13) ----
    priors_air = [
        # occupancies (8) – allow up to ~5 people, more mass to 0–3
        truncated(Normal(1.0, 1.0), 0.0, 5.0),
        truncated(Normal(1.0, 1.0), 0.0, 5.0),
        truncated(Normal(1.0, 1.0), 0.0, 5.0),
        truncated(Normal(1.0, 1.0), 0.0, 5.0),
        truncated(Normal(1.0, 1.0), 0.0, 5.0),
        truncated(Normal(1.0, 1.0), 0.0, 5.0),
        truncated(Normal(0.5, 0.7), 0.0, 3.0),
        truncated(Normal(0.5, 0.7), 0.0, 3.0),
        # AF (5) – one order of magnitude looser
        Normal(0.0, 0.2), Normal(0.0, 0.2), Normal(0.0, 0.2), Normal(0.0, 0.2), Normal(0.0, 0.2)
    ]
    @inbounds for i in 1:13
        logp += logpdf(priors_air[i], P[i])
    end

    # Initial CO₂ states (8): node-wise σ
    @inbounds for i in 1:8
        μ0 = f_co2_data[i, 1]
        σi = σ_co2_vec[i]
        logp += logpdf(Normal(μ0, σi), P[13 + i])  # P[14:21]
    end

    # ---- Thermal priors ----
    # 19 resistances (truncated), 8 capacities (truncated by groups)
    priors_therm = vcat(
        [truncated(Normal(2.0, 2.0), 0, 5.0) for _ in 1:19],
        [truncated(Normal(2000.0, 2000.0), 0, 6000.0) for _ in 1:4],
        [truncated(Normal(2000.0, 2000.0), 0, 6000.0) for _ in 1:2],
        [truncated(Normal(2000.0, 2000.0), 0, 6000.0)  for _ in 1:2]
    )
    @inbounds for i in 1:length(priors_therm)
        logp += logpdf(priors_therm[i], P[21 + i]) # P[22:48]
    end

    # Initial temperatures (8): node-wise σ
    @inbounds for i in 1:8
        μ0 = f_temp_data[i, 1]
        σi = σ_t_vec[i]
        logp += logpdf(Normal(μ0, σi), P[48 + i])  # P[49:56]
    end

    # ---- Priors for σ's (ensure positivity via truncation) ----
    prior_σ_co2 = truncated(Normal(5.0,  2.0),  0.00, Inf)
    prior_σ_t   = truncated(Normal(0.5,  0.3),  0.00, Inf)

    @inbounds for i in 1:8
        logp += logpdf(prior_σ_co2, σ_co2_vec[i])
        logp += logpdf(prior_σ_t,   σ_t_vec[i])
    end

    return isfinite(logp) ? logp : -Inf
end
