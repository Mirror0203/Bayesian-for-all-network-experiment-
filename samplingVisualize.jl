# samplingVisualize.jl
using Statistics
using Measures
using Plots

"""
Full-series CO₂ plot with window highlight.

Args
- sampling_chain : MCMC samples (matrix)
- f_noise_full   : 8×N measured CO₂ (ppm) [rows: A,B,C,D,E,F,H1,H2]
- window_start   : Int (1-based, inclusive)
- window_end     : Int (1-based, inclusive)
- sigma_mean_16  : Vector (len 8 or 16; CO₂ σ are first 8 if 16)
"""
function samplingVisualize(
    sampling_chain,
    f_noise_full::AbstractMatrix,
    window_start::Int,
    window_end::Int,
    sigma_mean_16::AbstractVector;
    f_true_full::Union{Nothing,AbstractMatrix}=nothing
)
    # --- posterior means for the window ---
    col_means       = vec(mean(sampling_chain, dims=1))
    P_samples_mean  = col_means[1:13]     # 8 ppl + 5 AF_Atm_*
    C0_samples_mean = col_means[14:21]    # initial CO₂ (8)

    # --- time bases ---
    N     = size(f_noise_full, 2)
    ts_all = collect(0.0:(N-1))                         # full series (steps of 10 s)
    tspan_win = (window_start - 1.0, window_end - 1.0)  # only for inference

    # --- forward model on the window only ---
    sol_mean = solutionForward(P_samples_mean, C0_samples_mean, tspan_win)
    ts_win   = sol_mean.t
    Y_win    = permutedims(reduce(hcat, sol_mean.u))    # T_win × 8

    # --- ribbons (±2σ) on the window only ---
    σ_air = length(sigma_mean_16) >= 16 ? sigma_mean_16[1:8] : sigma_mean_16
    ribbon_win = hcat([fill(2 .* σ_air[j], length(ts_win)) for j in 1:8]...)  # T_win × 8

    # --- full-series data matrices transposed to T × 8 ---
    f_noise_all_T = permutedims(f_noise_full)
    f_true_all_T  = f_true_full === nothing ? nothing : permutedims(f_true_full)

    # --- y limits from the full series (safer) ---
    ymin = min(350, round(Int, minimum(f_noise_all_T)) - 50)
    ymax = max(1300, round(Int, maximum(f_noise_all_T)) + 50)

    labels = ["Room A" "Room B" "Room C" "Room D" "Room E" "Room F" "Room H1" "Room H2"]

    # --- plot ---
    plt = plot(size=(2000, 600),
               xlabel = "Time (steps @ 10 s)",
               ylabel = "CO₂ (ppm)",
               legend = :outerright,
               left_margin = 12mm, right_margin = 18mm,
               bottom_margin = 14mm, top_margin = 6mm,
               ylim = (ymin, ymax))

    # window guides
    frame_size = 3
    vline!(plt, [window_start-1, window_end-1]; linestyle=:dash, linewidth=frame_size, color=:black, label="")

    # full-series ground truth (optional, solid)
    if f_true_all_T !== nothing
        plot!(plt, ts_all, f_true_all_T;
              label   = "Ground truth of " .* labels,
              alpha   = 0.9, linestyle = :solid, linewidth = 2.0,
              color   = [:blue :red :purple :orange :teal :brown :green :cyan])
    end

    # full-series noisy measurements (scatter)
    scatter!(plt, ts_all, f_noise_all_T; label = "",
             markersize = 2.0, alpha = 0.5,
             color = [:blue :red :purple :orange :teal :brown :green :cyan])

    # window-only inference with ribbons (dashed)
    plot!(plt, ts_win, Y_win;
          ribbon    = ribbon_win,
          linestyle = :dash, linewidth = 2,
          color     = :black,
          fillcolor = :gray80, fillalpha = 0.35,
          label     = "Inference of " .* labels)

    return plt
end
