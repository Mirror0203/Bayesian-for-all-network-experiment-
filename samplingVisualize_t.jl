# samplingVisualize_t.jl
using Statistics
using Measures
using Plots

"""
Full-series Temperature plot with window highlight.

Args
- sampling_chain : MCMC samples (matrix)
- air_params_mean: 13-length mean vector [8 ppl, 5 AF] (from the same window)
- f_noise_full   : 8×N measured T (°C)
- window_start   : Int (1-based, inclusive)
- window_end     : Int (1-based, inclusive)
- sigma_mean_16  : Vector (len 8 or 16; T σ are 9:16 if 16)
"""
function samplingVisualize_t(
    sampling_chain,
    air_params_mean::AbstractVector,
    f_noise_full::AbstractMatrix,
    window_start::Int,
    window_end::Int,
    sigma_mean_16::AbstractVector;
    f_true_full::Union{Nothing,AbstractMatrix}=nothing
)
    col_means = vec(mean(sampling_chain, dims=1))
    R_mean    = col_means[22:40]   # 19 R
    C_mean    = col_means[41:48]   # 8  C
    T0_mean   = col_means[49:56]   # 8  T0

    N      = size(f_noise_full, 2)
    ts_all = collect(0.0:(N-1))
    tspan_win = (window_start - 1.0, window_end - 1.0)

    P_mean   = vcat(R_mean, C_mean, air_params_mean)
    sol_mean = solutionForward_thermal(P_mean, T0_mean, tspan_win)
    ts_win   = sol_mean.t
    Y_win    = permutedims(reduce(hcat, sol_mean.u))     # T_win × 8

    σ_t = length(sigma_mean_16) >= 16 ? sigma_mean_16[9:16] : sigma_mean_16
    ribbon_win = hcat([fill(2 .* σ_t[j], length(ts_win)) for j in 1:8]...)  # T_win × 8

    f_noise_all_T = permutedims(f_noise_full)
    f_true_all_T  = f_true_full === nothing ? nothing : permutedims(f_true_full)

    ymin = min(15.0, minimum(f_noise_all_T) - 1)
    ymax = max(40.0, maximum(f_noise_all_T) + 1)

    labels = ["Room A" "Room B" "Room C" "Room D" "Room E" "Room F" "Room H1" "Room H2"]

    plt = plot(size=(2000, 600),
               xlabel = "Time (steps @ 10 s)",
               ylabel = "Temperature (°C)",
               legend = :outerright,
               left_margin = 12mm, right_margin = 18mm,
               bottom_margin = 14mm, top_margin = 6mm,
               ylim = (ymin, ymax))

    frame_size = 3
    vline!(plt, [window_start-1, window_end-1]; linestyle=:dash, linewidth=frame_size, color=:black, label="")

    if f_true_all_T !== nothing
        plot!(plt, ts_all, f_true_all_T;
              label   = "Ground truth of " .* labels,
              alpha   = 0.9, linestyle = :solid, linewidth = 2.0,
              color   = [:blue :red :purple :orange :teal :brown :green :cyan])
    end

    scatter!(plt, ts_all, f_noise_all_T; label = "",
             markersize = 2.0, alpha = 0.5,
             color = [:blue :red :purple :orange :teal :brown :green :cyan])

    plot!(plt, ts_win, Y_win;
          ribbon    = ribbon_win,
          linestyle = :dash, linewidth = 2,
          color     = :black,
          fillcolor = :gray80, fillalpha = 0.35,
          label     = "Inference of " .* labels)

    return plt
end
