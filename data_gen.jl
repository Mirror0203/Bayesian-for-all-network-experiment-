function co2_data_generation_piecewise(tspan;
    t_switch::Float64,
    ppl_before::NTuple{8,Float64} = (1,1,1,1,1,0,0,0),
    ppl_after::NTuple{8,Float64}  = (3,1,1,0,2,0,0,0),
    AF::NTuple{5,Float64}         = (0.01,0.01,0.01,-0.01,0.01),
    u0::Vector{Float64}           = fill(400.0, 8),
    debug::Bool = false
)
    t0, t1 = tspan
    @assert t0 < t_switch < t1 "t_switch must lie strictly inside tspan"

    # --- segment 1 ---
    P1 = vcat(collect(ppl_before), collect(AF))
    sol1 = solutionForward(P1, u0, (t0, t_switch))

    # --- segment 2 ---
    u0_2 = sol1.u[end]
    P2 = vcat(collect(ppl_after), collect(AF))
    sol2 = solutionForward(P2, u0_2, (t_switch, t1))

    # --- stitch ---
    t_all = vcat(sol1.t, sol2.t[2:end])
    u_all = vcat(sol1.u, sol2.u[2:end])

    # Build an ODESolution-like object so plotting/indexing is consistent
    sol = DiffEqBase.build_solution(sol1.prob, sol1.alg, t_all, u_all;
                                    retcode=:Success)

    if debug
        @info "stitched solution with $(length(t_all)) time points"
    end

    return sol
end

function thermal_data_generation_piecewise(tspan;
    t_switch::Float64,
    ppl_before::NTuple{8,Float64} = (1,1,1,1,1,0,0,0),
    ppl_after::NTuple{8,Float64}  = (3,1,1,0,2,0,0,0),
    AF::NTuple{5,Float64}         = (0.01,0.01,0.01,-0.01,0.01),
    u0_t::Vector{Float64}         = fill(20.0, 8),
    debug::Bool = false
)
    t0, t1 = tspan
    @assert t0 < t_switch < t1

    # --- fixed parameters as before ---
    Ca_basic = 5000.0
    R_Hall_Atm, R_Side_Room_Atm, R_Middle_Room_Atm, R_Lecture_Room_Atm = 1.5, 1.3, 1.2, 1.1
    R_Small_Room_Hall, R_Lecture_Room_Hall = 1.2, 1.1
    R_Small_Room_Between, R_Lecture_Room_Between = 1.15, 1.15
    R_Hall_Between = 1.2

    R_A_Atm = R_Side_Room_Atm;   R_B_Atm = R_Middle_Room_Atm
    R_C_Atm = R_Middle_Room_Atm; R_D_Atm = R_Side_Room_Atm
    R_E_Atm = R_Lecture_Room_Atm; R_F_Atm = R_Lecture_Room_Atm
    R_H1_Atm = R_Hall_Atm;        R_H2_Atm = R_Hall_Atm
    R_A_B = R_Small_Room_Between; R_B_C = R_Small_Room_Between; R_C_D = R_Small_Room_Between
    R_A_H1 = R_Small_Room_Hall;   R_B_H1 = R_Small_Room_Hall
    R_C_H2 = R_Small_Room_Hall;   R_D_H2 = R_Small_Room_Hall
    R_H1_E = R_Lecture_Room_Hall; R_H2_F = R_Lecture_Room_Hall
    R_E_F  = R_Lecture_Room_Between
    R_H1_H2 = R_Hall_Between

    Ca_A = Ca_basic; Ca_B = Ca_basic; Ca_C = Ca_basic; Ca_D = Ca_basic
    Ca_E = Ca_basic*2; Ca_F = Ca_basic*2; Ca_H1 = Ca_basic/2; Ca_H2 = Ca_basic/2
    AF = AF

    Rblock = [R_A_Atm, R_B_Atm, R_C_Atm, R_D_Atm, R_E_Atm, R_F_Atm, R_H1_Atm, R_H2_Atm,
              R_A_B, R_B_C, R_C_D, R_A_H1, R_B_H1, R_C_H2, R_D_H2, R_H1_E, R_H2_F, R_E_F,R_H1_H2]
    Cblock = [Ca_A, Ca_B, Ca_C, Ca_D, Ca_E, Ca_F, Ca_H1, Ca_H2]

    # segment 1
    P1_t = vcat(Rblock, Cblock, collect(ppl_before), collect(AF))
    sol1 = solutionForward_thermal(P1_t, u0_t, (t0, t_switch))

    # segment 2
    u0_2 = sol1.u[end]
    P2_t = vcat(Rblock, Cblock, collect(ppl_after), collect(AF))
    sol2 = solutionForward_thermal(P2_t, u0_2, (t_switch, t1))

    # stitch
    t_all = vcat(sol1.t, sol2.t[2:end])
    u_all = vcat(sol1.u, sol2.u[2:end])

    sol = DiffEqBase.build_solution(sol1.prob, sol1.alg, t_all, u_all;
                                    retcode=:Success)

    return sol
end

