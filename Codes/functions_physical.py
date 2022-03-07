import numpy as np
import pandas as pd
import pyet as pyet
import math

def ET(temp_file, rh_file):
    tmean = pd.read_csv(temp_file).drop('Id Sensore', axis = 1).set_index('Data-Ora')
    rh = pd.read_csv(rh_file).drop('Id Sensore', axis = 1).set_index('Data-Ora')
    pe_romanenko = pyet.romanenko(tmean, rh, k= 4.5)

    return pe_romanenko

def rutter_sparse(Sc, St, c, epsilon, pd, rain_series, et_series, dt=1):
    # This function implements the rutter sparse model to estimate the evapotranspiration and
    # the effective rain (i.e., the rain that effectively falls into the terrain). It is based on the paper in
    # https://www.sciencedirect.com/science/article/pii/S0022169496030661
    # It takes as input 5 parameters and 2 series. The series must be hourly series.
    # rain_series: contains the cummulated rain per hour [mm]
    # et_series: contains the cumulated potential evapotranspiration per hour [mm]
    # Sc: Canopy capacity per unit of area of cover [mm]
    # St: Trunk storage capacity per unit of area of cover [mm]
    # c: Area covered by vegetation / Total area [-]. It must vary between 0 and 1.
    # epsilon: Trunk evaporation as proportion of the evaporation from the saturated canopy [-]. It must vary between 0 and 1.
    # pd: drainage partitioning coefficient [-]. It must vary between 0 and 1.


    rain_series = rain_series.values[:,2]
    cc_0 = np.zeros(len(rain_series))
    cc_rain = np.zeros(len(rain_series))
    cc_drain = np.zeros(len(rain_series))
    cc_end = np.zeros(len(rain_series))
    ctc_0 = np.zeros(len(rain_series))
    ctc_rain = np.zeros(len(rain_series))
    ctc_drain = np.zeros(len(rain_series))
    ctc_end = np.zeros(len(rain_series))
    ec = np.zeros(len(rain_series))
    etc = np.zeros(len(rain_series))
    can_drain = np.zeros(len(rain_series))
    drip = np.zeros(len(rain_series))
    through = np.zeros(len(rain_series))
    trunk_input = np.zeros(len(rain_series))
    trunk_drain = np.zeros(len(rain_series))
    stemflow = np.zeros(len(rain_series))
    ev = np.zeros(len(rain_series))
    et_series = et_series.values
    
    for i in range(len(rain_series)):
        if i == 0:
            cc_0[i] = 0
            cc_rain[i] = rain_series[i]*c*dt
        else:
            cc_0[i] = cc_end[i-1]
            cc_rain[i] = cc_0[i] + rain_series[i]*c*dt
        can_drain[i] = max(cc_rain[i] - Sc, 0)
        drip[i] = (1 - pd)*can_drain[i]
        trunk_input[i] = pd*can_drain[i]
        cc_drain[i] = cc_rain[i] - can_drain[i]
        if cc_drain[i] < Sc:
            ec[i] = min((1-epsilon)*et_series[i]*cc_drain[i]/Sc, cc_rain[i])
        else:
            ec[i] = min((1-epsilon)*et_series[i], cc_drain[i])
        cc_end[i] = cc_drain[i] - ec[i]


        if i == 0:
            ctc_0[i] = 0
            ctc_rain[i] = trunk_input[i]*dt
        else:
            ctc_0[i] = ctc_end[i-1]
            ctc_rain[i] = ctc_0[i] + trunk_input[i]*dt
        trunk_drain[i] = max(ctc_rain[i] - St, 0)
        ctc_drain[i] = ctc_rain[i] - trunk_drain[i]
        if ctc_drain[i] < St:
            etc[i] = min(epsilon*et_series[i]*ctc_drain[i]/St, ctc_drain[i])
        else:
            etc[i] = min(epsilon*et_series[i], ctc_drain[i])
        ctc_end[i] = ctc_drain[i] - etc[i]

        through[i] = (1-c)*rain_series[i]+drip[i]
        stemflow[i] = trunk_drain[i]
        ev[i] = ec[i]+etc[i]

    # Returns ev in mm

    return ev, stemflow, through, ec, etc, cc_0, cc_rain, cc_drain, cc_end, ctc_0, ctc_rain, ctc_drain, ctc_end, can_drain, trunk_drain

def grnampt_getF2(f1, c1, ks, ts):
    f2 = f1
    f2min = f1 + ks*ts          # [in]
    if c1 == 0: return f2min
    if (ts < 10/3600) and (f1 > 0.01*c1):
        f2 = f1 +ks*(1+c1/f1)*ts
        return max(f2, f2min)
    c2 = c1*math.log(f1+c1) - ks*ts
    for j in range (1,20):
        df2 = (f2-f1-c1*math.log(f2+c1)+c2)/(1-c1/(f2+c1))
        if abs(df2)<0.000001: 
            return max(f2, f2min)
        f2 = f2 - df2
    return f2min

def grn_ampt_getSatInfil(ia, infil, Fumax, dt, Tr):
    if ia == 0: return 0
    infil['T'] = Tr
    c1 = infil['S']*infil['IMD']
    F2 = grnampt_getF2(infil['F'], c1, infil['Ks'], dt)
    dF = F2 - infil['F']
    if dF > ia*dt:
        dF = ia*dt
        infil['Sat'] = False
    infil['F'] = infil['F'] + dF
    infil['Fu'] = min(infil['Fu'] + dF, Fumax)  
    return dF/dt

def grn_ampt_getUnsatInfil(ia, infil, Fumax, dt, Tr):
    if ia == 0:
        if infil["Fu"]<= 0:
            return 0
        kr = (infil['Ks']**0.5/75)
        dF = kr*Fumax*dt
        infil['F'] = infil['F'] - dF
        infil['Fu'] = infil['Fu'] - dF
        if infil['Fu'] <= 0:
            infil['Fu'] = 0
            infil['F'] = 0
            infil['IMD'] = infil['IMDmax']
            return 0
        if infil['T'] <= 0:
            infil['IMD'] = (Fumax-infil['Fu'])/infil['Lu']
            infil['F'] = 0
        return 0
    if ia <= infil['Ks']:
        dF = ia*dt
        infil['F'] = infil['F'] + dF
        infil['Fu'] = min(infil['Fu'] + dF, Fumax)
        if infil['T'] <= 0:
            infil['IMD'] = (Fumax-infil['Fu'])/infil['Lu']
            infil['F'] = 0
        return ia
    infil['T'] = Tr
    Fs = infil['Ks']*infil['S']*infil['IMD']/(ia-infil['Ks'])
    if infil['F'] > Fs:
        infil['Sat'] = True
        return grn_ampt_getSatInfil(ia, infil, Fumax, dt, Tr)
    if infil['F']+ia*dt < Fs:
        dF = ia*dt
        infil['F'] = infil['F'] + dF
        infil['Fu'] = min(infil['Fu'] + dF, Fumax)
        return ia
    ts = dt - (Fs - infil['F'])/ia
    if ts <= 0: ts = 0
    c1 = infil['S']*infil['IMD']
    F2 = grnampt_getF2(Fs, c1, infil['Ks'], ts)
    if F2 > Fs + ia*ts:
        F2 = Fs + ia*ts
    dF = F2 - infil['F']
    infil['F'] = infil['F'] + dF
    infil['Fu'] = min(infil['Fu'] + dF, Fumax)
    infil['Sat'] = True
    return dF/dt