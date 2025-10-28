using StatsPlots, Plots
using Measures

function plot_trace_and_density(
    output_adaptiveMCMC, labels;
    titlefontsize = 12,
    xlabelfontsize = 9,
    ylabelfontsize = 9
)
    #chain = output_adaptiveMCMC.chain    
    chain = output_adaptiveMCMC
    num_parameter = size(chain, 2)
    figure_size = (600, num_parameter * 350)
    plot_size = (figure_size[1] *2, figure_size[2])
    marginmm = 50mm
    
    # Trace plot
    p1 = plot(
        chain,
        layout = (num_parameter, 1),
        size = figure_size,
        title = "Sampling of " .* labels,
        titlefontsize = titlefontsize,
        xlabel = "Iteration",
        xlabelfontsize = xlabelfontsize,
        ylabelfontsize = ylabelfontsize,
        ylabel = "Value",
        legend = false,
        left_margin = marginmm,
    )

    # Density plot
    p2 = density(
        eachcol(chain),
        layout = (num_parameter, 1),
        size = figure_size,
        title = "Distribution of " .* labels,
        titlefontsize = titlefontsize,
        xlabel = "Value",
        ylabel = "Distribution",
        xlabelfontsize = xlabelfontsize,
        ylabelfontsize = ylabelfontsize,
        legend = false,
        left_margin = marginmm,
    )
    
    return plot(p1, p2, layout = (1, 2), size = plot_size,margin = 5mm)
end