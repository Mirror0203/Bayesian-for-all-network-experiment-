function CO2concentration_synthetic!(du, u, p, t)
    # Assign CO2 concentration variables
    C_CO2_A, C_CO2_B, C_CO2_C, C_CO2_D, C_CO2_E, C_CO2_F, C_CO2_H1, C_CO2_H2 = u
    num_ppl_A, num_ppl_B, num_ppl_C, num_ppl_D, num_ppl_E, num_ppl_F, num_ppl_H1, num_ppl_H2, AF_Atm_A, AF_Atm_B, AF_Atm_C, AF_Atm_D, AF_Atm_E = p

    # know parameters
    V_A = 0.006096
    V_B = 0.006096
    V_C = 0.006096
    V_D = 0.006096
    V_E = 0.01276096
    V_F = 0.01276096
    V_H1 = 0.00512
    V_H2 = 0.00512
 
    ppl_schedule_A = vcat(ones(161))
    ppl_schedule_B = vcat(ones(161))
    ppl_schedule_C = vcat(ones(161))
    ppl_schedule_D = vcat(ones(161))
    ppl_schedule_E = vcat(ones(161))
    ppl_schedule_F = vcat(zeros(161))
    ppl_schedule_H1 = vcat(zeros(161))
    ppl_schedule_H2 = vcat(zeros(161))

    num_ppl_A = ppl_schedule_A[Int(floor(t)) + 1]
    num_ppl_B = ppl_schedule_B[Int(floor(t)) + 1]
    num_ppl_C = ppl_schedule_C[Int(floor(t)) + 1]
    num_ppl_D = ppl_schedule_D[Int(floor(t)) + 1]
    num_ppl_E = ppl_schedule_E[Int(floor(t)) + 1]
    num_ppl_F = ppl_schedule_F[Int(floor(t)) + 1]
    num_ppl_H1 = ppl_schedule_H1[Int(floor(t)) + 1]
    num_ppl_H2 = ppl_schedule_H2[Int(floor(t)) + 1]


    Time_resolution = 10 #seconds
    C_CO2_Atm = 410
    exhale_rate = 4.17e-6 # in m3/s,  250ml/min
    exhale_concentration = 5000 # ppm

    # Airflow mass balance
    # room A
    AF_A_Atm = - AF_Atm_A
    AF_A_H1 = - AF_A_Atm

    # room B
    AF_B_Atm = - AF_Atm_B
    AF_B_H1 = - AF_B_Atm

    # room C
    AF_C_Atm = - AF_Atm_C
    AF_C_H2 = - AF_C_Atm

    # room D
    AF_D_Atm = - AF_Atm_D
    AF_D_H2 = - AF_D_Atm

    # room E
    AF_E_Atm = - AF_Atm_E
    AF_E_H1 = - AF_E_Atm

    # room H1
    AF_H1_A = - AF_A_H1
    AF_H1_B = - AF_B_H1
    AF_H1_E = - AF_E_H1
    AF_H1_H2 = -(AF_H1_A + AF_H1_B + AF_H1_E)

    # room H2
    AF_H2_C = - AF_C_H2
    AF_H2_D = - AF_D_H2
    AF_H2_H1 = - AF_H1_H2
    AF_H2_F = -(AF_H2_C + AF_H2_D + AF_H2_H1)

    # room F
    AF_F_H2 = - AF_H2_F
    AF_F_Atm = - AF_F_H2
    AF_Atm_F = - AF_F_Atm

    # CO2 balance
    # room A
    du[1] = dC_CO2_A = (
      - (AF_A_Atm >= 0 ? AF_A_Atm * C_CO2_A : AF_A_Atm * C_CO2_Atm)
      - (AF_A_H1 >= 0 ? AF_A_H1 * C_CO2_A : AF_A_H1 * C_CO2_H1)
      + exhale_concentration * exhale_rate * num_ppl_A
    ) * Time_resolution / V_A
    
    # room B  
    du[2] = dC_CO2_B = (
      - (AF_B_Atm >= 0 ? AF_B_Atm * C_CO2_B : AF_B_Atm * C_CO2_Atm)
      - (AF_B_H1 >= 0 ? AF_B_H1 * C_CO2_B : AF_B_H1 * C_CO2_H1)
      + exhale_concentration * exhale_rate * num_ppl_B
    ) * Time_resolution / V_B
    


    # room C
    du[3] = dC_CO2_C = (
      - (AF_C_Atm >= 0 ? AF_C_Atm * C_CO2_C : AF_C_Atm * C_CO2_Atm)
      - (AF_C_H2 >= 0 ? AF_C_H2 * C_CO2_C : AF_C_H2 * C_CO2_H2)
      + exhale_concentration * exhale_rate * num_ppl_C
    ) * Time_resolution / V_C

    # room D
    du[4] = dC_CO2_D = (
      - (AF_D_Atm >= 0 ? AF_D_Atm * C_CO2_D : AF_D_Atm * C_CO2_Atm)
      - (AF_D_H2 >= 0 ? AF_D_H2 * C_CO2_D : AF_D_H2 * C_CO2_H2)
      + exhale_concentration * exhale_rate * num_ppl_D
    ) * Time_resolution / V_D
    


    # room E
    du[5] = dC_CO2_E = (
      - (AF_E_Atm >= 0 ? AF_E_Atm * C_CO2_E : AF_E_Atm * C_CO2_Atm)
      - (AF_E_H1 >= 0 ? AF_E_H1 * C_CO2_E : AF_E_H1 * C_CO2_H1)
      + exhale_concentration * exhale_rate * num_ppl_E
    ) * Time_resolution / V_E
    
    # room F
    du[6] = dC_CO2_F = (
      - (AF_F_Atm >= 0 ? AF_F_Atm * C_CO2_F : AF_F_Atm * C_CO2_Atm)
      - (AF_F_H2 >= 0 ? AF_F_H2 * C_CO2_F : AF_F_H2 * C_CO2_H2)
      + exhale_concentration * exhale_rate * num_ppl_F
    ) * Time_resolution / V_F

    # room H1
    du[7] = dC_CO2_H1 = (
      - (AF_H1_H2 >= 0 ? AF_H1_H2 * C_CO2_H1 : AF_H1_H2 * C_CO2_H2)
      - (AF_H1_A >= 0 ? AF_H1_A * C_CO2_H1 : AF_H1_A * C_CO2_A)
      - (AF_H1_B >= 0 ? AF_H1_B * C_CO2_H1 : AF_H1_B * C_CO2_B)
      - (AF_H1_E >= 0 ? AF_H1_E * C_CO2_H1 : AF_H1_E * C_CO2_E)
      + exhale_concentration * exhale_rate * num_ppl_H1
    ) * Time_resolution / V_H1

    # room H2
    du[8] = dC_CO2_H2 = (
      - (AF_H2_H1 >= 0 ? AF_H2_H1 * C_CO2_H2 : AF_H2_H1 * C_CO2_H1)
      - (AF_H2_C >= 0 ? AF_H2_C * C_CO2_H2 : AF_H2_C * C_CO2_C)
      - (AF_H2_D >= 0 ? AF_H2_D * C_CO2_H2 : AF_H2_D * C_CO2_D)
      - (AF_H2_F >= 0 ? AF_H2_F * C_CO2_H2 : AF_H2_F * C_CO2_F)
      + exhale_concentration * exhale_rate * num_ppl_H2
    ) * Time_resolution / V_H2
end
#=
function solutionForward(P, u0, tspan)
    # Initial CO2 concentrations in ppm
    # Solve ODE
    prob = ODEProblem(CO2concentration!, u0, tspan, P)
    sol = solve(prob, Rosenbrock23(), saveat = 1)

    # Return the H room CO2 concentration (or stack all if needed)
    return sol  # Or: vec(hcat(sol.u...)) to flatten all
end
=#

function CO2concentration!(du, u, p, t)
    # Assign CO2 concentration variables
    C_CO2_A, C_CO2_B, C_CO2_C, C_CO2_D, C_CO2_E, C_CO2_F, C_CO2_H1, C_CO2_H2 = u
    num_ppl_A, num_ppl_B, num_ppl_C, num_ppl_D, num_ppl_E, num_ppl_F, num_ppl_H1, num_ppl_H2, AF_Atm_A, AF_Atm_B, AF_Atm_C, AF_Atm_D, AF_Atm_E = p

    # know parameters
    V_A = 0.006096
    V_B = 0.006096
    V_C = 0.006096
    V_D = 0.006096
    V_E = 0.01276096
    V_F = 0.01276096
    V_H1 = 0.00512
    V_H2 = 0.00512

    Time_resolution = 10 #seconds
    C_CO2_Atm = 410
    exhale_rate = 4.17e-6 # in m3/s,  250ml/min
    exhale_concentration = 5000 # ppm

    # Airflow mass balance
    # room A
    AF_A_Atm = - AF_Atm_A
    AF_A_H1 = - AF_A_Atm

    # room B
    AF_B_Atm = - AF_Atm_B
    AF_B_H1 = - AF_B_Atm

    # room C
    AF_C_Atm = - AF_Atm_C
    AF_C_H2 = - AF_C_Atm

    # room D
    AF_D_Atm = - AF_Atm_D
    AF_D_H2 = - AF_D_Atm

    # room E
    AF_E_Atm = - AF_Atm_E
    AF_E_H1 = - AF_E_Atm

    # room H1
    AF_H1_A = - AF_A_H1
    AF_H1_B = - AF_B_H1
    AF_H1_E = - AF_E_H1
    AF_H1_H2 = -(AF_H1_A + AF_H1_B + AF_H1_E)

    # room H2
    AF_H2_C = - AF_C_H2
    AF_H2_D = - AF_D_H2
    AF_H2_H1 = - AF_H1_H2
    AF_H2_F = -(AF_H2_C + AF_H2_D + AF_H2_H1)

    # room F
    AF_F_H2 = - AF_H2_F
    AF_F_Atm = - AF_F_H2
    AF_Atm_F = - AF_F_Atm

    # CO2 balance
    # room A
    du[1] = dC_CO2_A = (
      - (AF_A_Atm >= 0 ? AF_A_Atm * C_CO2_A : AF_A_Atm * C_CO2_Atm)
      - (AF_A_H1 >= 0 ? AF_A_H1 * C_CO2_A : AF_A_H1 * C_CO2_H1)
      + exhale_concentration * exhale_rate * num_ppl_A
    ) * Time_resolution / V_A
    
    # room B  
    du[2] = dC_CO2_B = (
      - (AF_B_Atm >= 0 ? AF_B_Atm * C_CO2_B : AF_B_Atm * C_CO2_Atm)
      - (AF_B_H1 >= 0 ? AF_B_H1 * C_CO2_B : AF_B_H1 * C_CO2_H1)
      + exhale_concentration * exhale_rate * num_ppl_B
    ) * Time_resolution / V_B
    


    # room C
    du[3] = dC_CO2_C = (
      - (AF_C_Atm >= 0 ? AF_C_Atm * C_CO2_C : AF_C_Atm * C_CO2_Atm)
      - (AF_C_H2 >= 0 ? AF_C_H2 * C_CO2_C : AF_C_H2 * C_CO2_H2)
      + exhale_concentration * exhale_rate * num_ppl_C
    ) * Time_resolution / V_C

    # room D
    du[4] = dC_CO2_D = (
      - (AF_D_Atm >= 0 ? AF_D_Atm * C_CO2_D : AF_D_Atm * C_CO2_Atm)
      - (AF_D_H2 >= 0 ? AF_D_H2 * C_CO2_D : AF_D_H2 * C_CO2_H2)
      + exhale_concentration * exhale_rate * num_ppl_D
    ) * Time_resolution / V_D
    


    # room E
    du[5] = dC_CO2_E = (
      - (AF_E_Atm >= 0 ? AF_E_Atm * C_CO2_E : AF_E_Atm * C_CO2_Atm)
      - (AF_E_H1 >= 0 ? AF_E_H1 * C_CO2_E : AF_E_H1 * C_CO2_H1)
      + exhale_concentration * exhale_rate * num_ppl_E
    ) * Time_resolution / V_E
    
    # room F
    du[6] = dC_CO2_F = (
      - (AF_F_Atm >= 0 ? AF_F_Atm * C_CO2_F : AF_F_Atm * C_CO2_Atm)
      - (AF_F_H2 >= 0 ? AF_F_H2 * C_CO2_F : AF_F_H2 * C_CO2_H2)
      + exhale_concentration * exhale_rate * num_ppl_F
    ) * Time_resolution / V_F

    # room H1
    du[7] = dC_CO2_H1 = (
      - (AF_H1_H2 >= 0 ? AF_H1_H2 * C_CO2_H1 : AF_H1_H2 * C_CO2_H2)
      - (AF_H1_A >= 0 ? AF_H1_A * C_CO2_H1 : AF_H1_A * C_CO2_A)
      - (AF_H1_B >= 0 ? AF_H1_B * C_CO2_H1 : AF_H1_B * C_CO2_B)
      - (AF_H1_E >= 0 ? AF_H1_E * C_CO2_H1 : AF_H1_E * C_CO2_E)
      + exhale_concentration * exhale_rate * num_ppl_H1
    ) * Time_resolution / V_H1

    # room H2
    du[8] = dC_CO2_H2 = (
      - (AF_H2_H1 >= 0 ? AF_H2_H1 * C_CO2_H2 : AF_H2_H1 * C_CO2_H1)
      - (AF_H2_C >= 0 ? AF_H2_C * C_CO2_H2 : AF_H2_C * C_CO2_C)
      - (AF_H2_D >= 0 ? AF_H2_D * C_CO2_H2 : AF_H2_D * C_CO2_D)
      - (AF_H2_F >= 0 ? AF_H2_F * C_CO2_H2 : AF_H2_F * C_CO2_F)
      + exhale_concentration * exhale_rate * num_ppl_H2
    ) * Time_resolution / V_H2
end

function solutionForward_synthetic(P, u0, tspan)
    # Initial CO2 concentrations in ppm
    # Solve ODE
    prob = ODEProblem(CO2concentration_synthetic!, u0, tspan, P)
    sol = solve(prob, Rosenbrock23(), saveat = 1)

    # Return the H room CO2 concentration (or stack all if needed)
    return sol  # Or: vec(hcat(sol.u...)) to flatten all
end

function solutionForward(P, u0, tspan)
    # Initial CO2 concentrations in ppm
    # Solve ODE
    prob = ODEProblem(CO2concentration!, u0, tspan, P)
    sol = solve(prob, Rosenbrock23(), saveat = 1)

    # Return the H room CO2 concentration (or stack all if needed)
    return sol  # Or: vec(hcat(sol.u...)) to flatten all
end

using DifferentialEquations, Plots
function ThermalNetworkWall!(du, u, p, t)
    T_A, T_B, T_C, T_D, T_E, T_F, T_H1, T_H2 = u

    # 8 to ambient:
    R_A_Atm, R_B_Atm, R_C_Atm, R_D_Atm, R_E_Atm, R_F_Atm, R_H1_Atm, R_H2_Atm = p[1:8]
    # 11 between rooms (NEW: R_H1_H2 at the end):
    R_A_B, R_B_C, R_C_D, R_A_H1, R_B_H1, R_C_H2, R_D_H2, R_H1_E, R_H2_F, R_E_F, R_H1_H2 = p[9:19]
    # capacities now start at 20:
    Ca_A, Ca_B, Ca_C, Ca_D, Ca_E, Ca_F, Ca_H1, Ca_H2  = p[20:27]
    # people 8 and AF 5 follow:
    num_ppl_A, num_ppl_B, num_ppl_C, num_ppl_D, num_ppl_E, num_ppl_F, num_ppl_H1, num_ppl_H2 = p[28:35]
    AF_Atm_A, AF_Atm_B, AF_Atm_C, AF_Atm_D, AF_Atm_E = p[36:40]

    T_amb = 23
    Time_resolution = 10 #seconds
    Q_ppl = 10 # heat load per person in W
    Ca_air = 1000 # specific heat capacity of air in J/(kg*K)
    ρ_air = 1.2 # density of air in kg/m^3
    # Airflow mass balance
    # room A
    AF_A_Atm = - AF_Atm_A
    AF_A_H1 = - AF_A_Atm

    # room B
    AF_B_Atm = - AF_Atm_B
    AF_B_H1 = - AF_B_Atm

    # room C
    AF_C_Atm = - AF_Atm_C
    AF_C_H2 = - AF_C_Atm

    # room D
    AF_D_Atm = - AF_Atm_D
    AF_D_H2 = - AF_D_Atm

    # room E
    AF_E_Atm = - AF_Atm_E
    AF_E_H1 = - AF_E_Atm

    # room H1
    AF_H1_A = - AF_A_H1
    AF_H1_B = - AF_B_H1
    AF_H1_E = - AF_E_H1
    AF_H1_H2 = -(AF_H1_A + AF_H1_B + AF_H1_E)

    # room H2
    AF_H2_C = - AF_C_H2
    AF_H2_D = - AF_D_H2
    AF_H2_H1 = - AF_H1_H2
    AF_H2_F = -(AF_H2_C + AF_H2_D + AF_H2_H1)

    # room F
    AF_F_H2 = - AF_H2_F
    AF_F_Atm = - AF_F_H2
    AF_Atm_F = - AF_F_Atm
    
    du[1] = dT_A = (
        + (T_amb - T_A)/R_A_Atm
        + (T_B - T_A)/R_A_B
        + (T_H1 - T_A)/R_A_H1
        + Q_ppl *  num_ppl_A
        - (AF_A_Atm >= 0 ? AF_A_Atm * T_A : AF_A_Atm * T_amb) * Ca_air * ρ_air
        - (AF_A_H1 >= 0 ? AF_A_H1 * T_A : AF_A_H1 * T_H1) * Ca_air * ρ_air
        )/Ca_A * Time_resolution

    # Room B
    du[2] = dT_B = (
        + (T_amb - T_B)/R_B_Atm
        + (T_A - T_B)/R_A_B
        + (T_C - T_B)/R_B_C
        + (T_H1 - T_B)/R_B_H1
        + Q_ppl * num_ppl_B
        - (AF_B_Atm >= 0 ? AF_B_Atm * T_B : AF_B_Atm * T_amb) * Ca_air * ρ_air
        - (AF_B_H1 >= 0 ? AF_B_H1 * T_B : AF_B_H1 * T_H1) * Ca_air * ρ_air
        ) / Ca_B * Time_resolution

    # Room C
    du[3] = dT_C = (
        + (T_amb - T_C)/R_C_Atm
        + (T_B - T_C)/R_B_C
        + (T_D - T_C)/R_C_D
        + (T_H2 - T_C)/R_C_H2
        + Q_ppl * num_ppl_C
        - (AF_C_Atm >= 0 ? AF_C_Atm * T_C : AF_C_Atm * T_amb) * Ca_air * ρ_air
        - (AF_C_H2 >= 0 ? AF_C_H2 * T_C : AF_C_H2 * T_H2) * Ca_air * ρ_air
    ) / Ca_C * Time_resolution

    # Room D
    du[4] = dT_D = (
        + (T_amb - T_D)/R_D_Atm
        + (T_C - T_D)/R_C_D
        + (T_H2 - T_D)/R_D_H2
        + Q_ppl * num_ppl_D
        - (AF_D_Atm >= 0 ? AF_D_Atm * T_D : AF_D_Atm * T_amb) * Ca_air * ρ_air
        - (AF_D_H2 >= 0 ? AF_D_H2 * T_D : AF_D_H2 * T_H2) * Ca_air * ρ_air
    ) / Ca_D * Time_resolution

    # Room E
    du[5] = dT_E = (
        + (T_amb - T_E)/R_E_Atm
        + (T_H1 - T_E)/R_H1_E
        + (T_F - T_E)/R_E_F
        + Q_ppl * num_ppl_E
        - (AF_E_Atm >= 0 ? AF_E_Atm * T_E : AF_E_Atm * T_amb) * Ca_air * ρ_air
        - (AF_E_H1 >= 0 ? AF_E_H1 * T_E : AF_E_H1 * T_H1) * Ca_air * ρ_air
    ) / Ca_E * Time_resolution

    # Room F
    du[6] = dT_F = (
        + (T_amb - T_F)/R_F_Atm
        + (T_E - T_F)/R_E_F
        + (T_H2 - T_F)/R_H2_F
        + Q_ppl * num_ppl_F
        - (AF_F_Atm >= 0 ? AF_F_Atm * T_F : AF_F_Atm * T_amb) * Ca_air * ρ_air
        - (AF_F_H2 >= 0 ? AF_F_H2 * T_F : AF_F_H2 * T_H2) * Ca_air * ρ_air
    ) / Ca_F * Time_resolution

    # Room H1
    du[7] = dT_H1 = (
        + (T_amb - T_H1)/R_H1_Atm
        + (T_A - T_H1)/R_A_H1
        + (T_B - T_H1)/R_B_H1
        + (T_E - T_H1)/R_H1_E
        + (T_H2 - T_H1)/R_H1_H2

        + Q_ppl * num_ppl_H1

        - (AF_H1_A >= 0 ? AF_H1_A * T_H1 : AF_H1_A * T_A) * Ca_air * ρ_air
        - (AF_H1_B >= 0 ? AF_H1_B * T_H1 : AF_H1_B * T_B) * Ca_air * ρ_air
        - (AF_H1_E >= 0 ? AF_H1_E * T_H1 : AF_H1_E * T_E) * Ca_air * ρ_air
        - (AF_H1_H2 >= 0 ? AF_H1_H2 * T_H1 : AF_H1_H2 * T_H2) * Ca_air * ρ_air
    ) / Ca_H1 * Time_resolution

    # Room H2
    du[8]  = dT_H2 = (
        + (T_amb - T_H2)/R_H2_Atm
        + (T_C - T_H2)/R_C_H2
        + (T_D - T_H2)/R_D_H2
        + (T_F - T_H2)/R_H2_F
        + (T_H1 - T_H2)/R_H1_H2

        + Q_ppl * num_ppl_H2
        
        - (AF_H2_C >= 0 ? AF_H2_C * T_H2 : AF_H2_C * T_C) * Ca_air * ρ_air
        - (AF_H2_D >= 0 ? AF_H2_D * T_H2 : AF_H2_D * T_D) * Ca_air * ρ_air
        - (AF_H2_F >= 0 ? AF_H2_F * T_H2 : AF_H2_F * T_F) * Ca_air * ρ_air
        - (AF_H2_H1 >= 0 ? AF_H2_H1 * T_H2 : AF_H2_H1 * T_H1) * Ca_air * ρ_air
    ) / Ca_H2 * Time_resolution
end

function solutionForward_thermal(P, u0, tspan)
    # Initial CO2 concentrations in ppm
    # Solve ODE
    prob = ODEProblem(ThermalNetworkWall!, u0, tspan, P)
    sol = solve(prob, Rosenbrock23(), saveat = 1)

    # Return the H room CO2 concentration (or stack all if needed)
    return sol  # Or: vec(hcat(sol.u...)) to flatten all
end
#Rosenbrock23()
