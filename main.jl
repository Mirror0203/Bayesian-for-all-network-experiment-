# main.jl – Joint Bayesian Inference for Airflow and Thermal Models
using Random, LinearAlgebra, Statistics, RobustAdaptiveMetropolisSampler, Plots
using DifferentialEquations, Distributions, StatsPlots, CSV
using StaticArrays, Measures
using Statistics, DataFrames, Dates


#########################
filepath = "D:/OneDrive/桌面/OneDrive - TU Eindhoven/Desktop/Experiment/20251023 experiment/CO2.csv"
#filepath = "C:/Users/A/OneDrive - TU Eindhoven/Desktop/Experiment/20251022 test experiment/CO2LastMeasure.csv"
df = CSV.read(filepath, DataFrame)
df_CO2 = unstack(df, :date_time, :sensor_id, :CO2)
df_tair = unstack(df, :date_time, :sensor_id, :t_air)

# Map sensor IDs to room names
name_map = Dict(
    :ID3983 => :H2,
    :ID3982 => :E,
    :ID3978 => :H1,
    :ID3987 => :F,
    :ID3985 => :A,
    :ID3979 => :B,
    :ID3980 => :D,
    :ID3981 => :C,
    :ID3986 => :Atm
)

# Rename columns in both wide tables
rename!(df_CO2,  name_map)
rename!(df_tair, name_map)


# === SIMPLE OFFSETS (your numbers) ===
# Offsets are defined by sensor_id, then mapped to room names via `name_map`
co2_offsets_id = Dict(
    :ID3985 => -18.775510204081513,
    :ID3980 =>  35.402040816326405,
    :ID3979 => -10.732653061224426,
    :ID3983 => -26.47142857142859,
    :ID3978 =>  22.0693877551019,
    :ID3982 =>  11.48163265306124,
    :ID3981 =>  -2.2571428571429237,
    :ID3987 =>  12.420408163265279,
    :ID3986 => -23.136734693877543
)

t_offsets_id = Dict(
    :ID3985 =>  0.23378684807256178,
    :ID3980 => -0.27845804988660916,
    :ID3979 =>  0.21950113378685998,
    :ID3983 => -0.4213151927437586,
    :ID3978 => -0.09070294784581279,
    :ID3982 => -0.19274376417236638,
    :ID3981 => -0.07233560090703506,
    :ID3987 => -0.28049886621314,
    :ID3986 =>  0.8827664399092825
)

# Map offsets from sensor_id -> room symbol (A, B, C, …, Atm)
co2_offsets_room = Dict(name_map[k] => v for (k,v) in co2_offsets_id)
t_offsets_room   = Dict(name_map[k] => v for (k,v) in t_offsets_id)

# Rooms you want to adjust/plot
rooms = [:Atm, :H1, :H2, :A, :B, :C, :D, :E, :F]

# --- Apply offsets to get calibrated wide tables ---
df_CO2_cal  = copy(df_CO2)
df_tair_cal = copy(df_tair)

for r in rooms
    if haskey(co2_offsets_room, r)
        # add the constant offset to the whole column (keeps missing as missing)
        df_CO2_cal[!, r] = df_CO2[!, r] .+ co2_offsets_room[r]
    end
    if haskey(t_offsets_room, r)
        df_tair_cal[!, r] = df_tair[!, r] .+ t_offsets_room[r]
    end
end
#########################

start_t = Time(11,10,0)
end_t   = Time(11,25,0)

mask = (df_CO2_cal.date_time .>= start_t) .& (df_CO2_cal.date_time .< end_t)  # [start, end)
df_CO2_cal = df_CO2_cal[mask, :]
df_tair_cal = df_tair_cal[mask, :]

function plot_data(df1)
    rooms = [:Atm, :H1, :H2, :A, :B, :C, :D, :E, :F]
    plt = plot(
        xlabel = "Time",
        ylabel = "T (C)", #"CO2 (ppm)",
        title  =  "Tmeperature levels — all rooms",#"CO2 levels — all rooms",
        legend = :outerright,
        size = (900, 400),
        bottom_margin = 5mm,
        left_margin = 5mm
    )
    for r in rooms
        plot!(plt, df1.date_time, df1[!, r], label = String(r))
    end

    return(plt)
end

plot_data(df_tair_cal)



Random.seed!(0)
include("models.jl")
include("data_gen.jl")
include("inference.jl")
include("samplingDensityVisualize.jl")
include("samplingVisualize.jl")
include("samplingVisualize_t.jl")

const MODEL_ROOMS = [:A, :B, :C, :D, :E, :F, :H1, :H2]
const ROOT = @__DIR__ 


"""
Return an 8×N Float64 matrix (rows=rooms) from a calibrated wide DF.
Drops :Atm column and enforces room order matching the model state.
Throws a clear error if any of the required columns are missing.
"""
function df_to_matrix_8(df_cal)::Matrix{Float64}
    cols_needed = MODEL_ROOMS
    missing_cols = setdiff(cols_needed, propertynames(df_cal))
    if !isempty(missing_cols)
        error("Missing columns in df: $(missing_cols). Did mapping/renaming run?")
    end
    # Select the columns in the exact order, materialize to Array{Float64}
    M = reduce(hcat, (Float64.(df_cal[!, r]) for r in cols_needed))
    # Now M is N×8; transpose to 8×N to match your code expectations
    return permutedims(M)  # 8×N
end

# Drop :Atm and keep only model rooms in correct order
select!(df_CO2_cal, [:date_time; :A; :B; :C; :D; :E; :F; :H1; :H2])
select!(df_tair_cal, [:date_time; :A; :B; :C; :D; :E; :F; :H1; :H2])

# Convert to 8×N matrices (Float64), aligned with model state order
f_co2_real = df_to_matrix_8(df_CO2_cal)   # (8×N) ppm
f_t_real   = df_to_matrix_8(df_tair_cal)  # (8×N) °C

# Number of time points (each column == one 10 s step)
N_real = size(f_co2_real, 2)
@assert size(f_t_real, 2) == N_real "CO₂ and T time lengths differ"

# For plotting elsewhere might still need ODE solutions; here we only need data matrices.
f_co2_noise = f_co2_real         # 8×N
f_t_noise   = f_t_real           # 8×N

# === Joint inference setup (shared across windows) ===

# ---- Baseline initial guesses (used for the first window only) ----
P_air_init   = [0.01,0.01,0.01,0.01,0.01,2,0.01,0.01]
AF_init      = [-0.001, -0.001, -0.001, -0.001, 0.001]
C_air0_init  = copy(f_co2_real[:, 1])
R_init       = fill(1.0, 19)
Ca_init      = vcat(fill(400.0, 4), fill(800.0, 2), fill(200.0, 2))
T_init       = copy(f_t_real[:,   1])
σ_co2_init_vec = fill(2.0, 8)
σ_t_init_vec   = fill(0.1, 8)

P_air  = vcat(P_air_init, AF_init)
P0 = vcat(P_air, C_air0_init, R_init, Ca_init, T_init, σ_co2_init_vec, σ_t_init_vec)  # <- baseline for first window

# Give the first 13 params (8 ppl + 5 AF) more room
sigma_air_proposal = vcat(fill(0.05, 8), fill(0.05, 5))
sigma_rest         = 0.01 .* vcat(C_air0_init, R_init, Ca_init, T_init)
sigma_init         = vcat(sigma_air_proposal, sigma_rest)
sigma_init         = vcat(sigma_init, fill(0.5, 8), fill(0.01, 8))  # σ_CO2 and σ_T proposals

# Helper to robustly update P0
function safe_update_init!(P0_old, candidate)
    if any(!isfinite, candidate)
        @warn "Posterior-mean contained non-finite values; keeping previous init."
        return P0_old
    end
    println("Updated init for next window: ", candidate)
    return candidate
end

########### LOOOOOOP start
# time span in "step units" (1 step = 10 s)
tspan = (0.0, N_real - 1.0)
# choose a window size in steps (e.g., 30 steps = 300 s = 5 min)
time_window_size = 40

for i in 1:N_real-time_window_size
    i = 30
    window_start = i
    window_end = window_start + time_window_size
    f_co2_noise_window = f_co2_noise[:, window_start:window_end]
    f_t_noise_window = f_t_noise[:, window_start:window_end]
    println("Running joint inference for window $i to $(i+time_window_size) ...")

    # Use updated initial guess from previous window
    P_init = P0  # first window uses the baseline; later windows use previous posterior mean

    # Log-posterior unchanged signature
    function logPosterior_joint(P)
        evaluateLogPdf_joint(P, f_co2_noise_window, f_t_noise_window, time_window_size)
    end

    n_samples = 200_000
    output_joint = RAM_sample(logPosterior_joint, P_init, sigma_init, n_samples)

    whole_chain = output_joint.chain
    sampling_chain = output_joint.chain[end-50_000+1:end, :]
    col_means      = vec(mean(sampling_chain, dims=1))

    # ---- Update init for the next window with posterior-mean ----
    P0 = safe_update_init!(P0, col_means)

    # Seperate parameters for visualization
    idx_air_params   = 1:13
    idx_air_init     = 14:21
    idx_thermal_R    = 22:40
    idx_thermal_C    = 41:48
    idx_thermal_init = 49:56
    idx_sigma_air    = 57:64
    idx_sigma_tl     = 65:72

    air_params_mean   = col_means[idx_air_params]
    air_init_mean     = col_means[idx_air_init]
    thermal_R_mean    = col_means[idx_thermal_R]
    thermal_C_mean    = col_means[idx_thermal_C]
    thermal_init_mean = col_means[idx_thermal_init]
    sigma_air_mean    = col_means[idx_sigma_air]
    sigma_tl_mean     = col_means[idx_sigma_tl]

    folder_label = "20251023"
    folder_path_air = joinpath(ROOT, "Figures", folder_label, "air")
    folder_path_thermal = joinpath(ROOT, "Figures", folder_label, "thermal")
    
    # Logging to CSV
    open(joinpath(folder_path_air, "logfile_air.csv"), "a") do f
        values_air = [i; vec([air_params_mean; air_init_mean; sigma_air_mean])]
        write(f, join(values_air, ",") * "\n")
    end
    open(joinpath(folder_path_thermal, "logfile_thermal.csv"), "a") do f
        values_thermal = [i; vec([thermal_R_mean; thermal_C_mean; thermal_init_mean; sigma_tl_mean])]
        write(f, join(values_thermal, ",") * "\n")
    end
    
    # Curve visualization with ribbon plots
    σ_mean_16 = vcat(sigma_air_mean, sigma_tl_mean)

    include("samplingVisualize.jl")
    plt1 = samplingVisualize(
        sampling_chain, f_co2_noise, window_start, window_end, σ_mean_16; 
        f_true_full = f_co2_noise
    )
    savefig(plt1, joinpath(folder_path_air, "curve_plot", "$(i).png"))

    include("samplingVisualize_t.jl")
    plt2 = samplingVisualize_t(
        sampling_chain, air_params_mean, f_t_noise, window_start, window_end, σ_mean_16; 
        f_true_full = f_t_noise
    )
    savefig(plt2, joinpath(folder_path_thermal, "curve_plot", "$(i).png"))
    
    # Distribution Visualization
    plot_trace_and_density(whole_chain[:, idx_air_params], [
        "num_ppl_A", "num_ppl_B", "num_ppl_C", "num_ppl_D", "num_ppl_E", "num_ppl_F", 
        "num_ppl_H1", "num_ppl_H2", 
        "AF_Atm_A", "AF_Atm_B", "AF_Atm_C", "AF_Atm_D", "AF_Atm_E"])
    savefig(joinpath(folder_path_air, "whole_chain", "$(i)_air_params.png"))

    plot_trace_and_density(whole_chain[:, idx_air_init], [
        "CA_0", "CB_0", "CC_0", "CD_0", "CE_0", "CF_0", "CH1_0", "CH2_0"])
    savefig(joinpath(folder_path_air, "whole_chain", "$(i)_air_initials.png"))


    plot_trace_and_density(whole_chain[:, idx_sigma_air], [
        "σ_CO2_A", "σ_CO2_B", "σ_CO2_C", "σ_CO2_D", "σ_CO2_E", "σ_CO2_F", "σ_CO2_H1", "σ_CO2_H2"])
    savefig(joinpath(folder_path_air, "whole_chain", "$(i)_air_sigma.png"))

    plot_trace_and_density(whole_chain[:, idx_thermal_R], [
        "R_A_Atm", "R_B_Atm", "R_C_Atm", "R_D_Atm", "R_E_Atm", "R_F_Atm", "R_H1_Atm", "R_H2_Atm",
        "R_A_B", "R_B_C", "R_C_D",
        "R_A_H1", "R_B_H1", "R_C_H2", "R_D_H2",
        "R_H1_E", "R_H2_F", "R_E_F",
        "R_H1_H2"
    ])
    savefig(joinpath(folder_path_thermal, "whole_chain", "$(i)_thermal_Rs.png"))

    plot_trace_and_density(whole_chain[:, idx_thermal_C], [
        "Ca_A", "Ca_B", "Ca_C", "Ca_D", "Ca_E", "Ca_F", "Ca_H1", "Ca_H2"])
    savefig(joinpath(folder_path_thermal, "whole_chain", "$(i)_thermal_Cs.png"))

    plot_trace_and_density(whole_chain[:, idx_thermal_init], [
        "T_A0", "T_B0", "T_C0", "T_D0", "T_E0", "T_F0", "T_H1_0", "T_H2_0"])
    savefig(joinpath(folder_path_thermal, "whole_chain", "$(i)_thermal_initials.png"))


    println("Joint inference $i finished!")
end
