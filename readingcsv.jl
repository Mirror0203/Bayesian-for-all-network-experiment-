using CSV, DataFrames, Dates, Plots
using Statistics
#filepath = "D:/OneDrive/桌面/OneDrive - TU Eindhoven/Desktop/Experiment/20251021_13_31 test/CO2.csv" # Read the CSV file 

filepath = "C:/Users/A/OneDrive - TU Eindhoven/Desktop/Experiment/20251022 test experiment/CO2LastMeasure.csv"
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


function plot_CO2_absolute_time(df_CO2_cal)
    rooms = [:Atm, :H1, :H2, :A, :B, :C, :D, :E, :F]

    t0 = df_CO2_cal.date_time[1]
    rel_min = Float64.((df_CO2_cal.date_time .- t0) ./ Minute(1))  # e.g., 0, 0.5, 1.0, ...

    plt = plot(
        xlabel = "Minutes from start",
        size=(700,400),
        ylabel = "CO2",
        legend = :outerright,
        xlim = (0, maximum(rel_min)),
        xticks = 0:ceil(Int, maximum(rel_min))  # show 0,1,2,3,...
    )

    for r in rooms
        plot!(plt, rel_min, df_CO2_cal[!, r], label = String(r))
    end

    return plt
end


plot_CO2_absolute_time(df_CO2_cal)