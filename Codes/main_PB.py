import pandas as pd
import numpy as np
from functions_physical import rutter_sparse, grn_ampt_getSatInfil, grn_ampt_getUnsatInfil

station = 'Molteno'
variable_precipitation = '9106_4'


##################  Compute interception  ########################
rain_series = pd.read_csv('./../joined_data/'+station+'/'+variable_precipitation+'.csv')        # In mm
et_series = pd.read_csv('./../joined_data/'+station+'/ET.csv').set_index('Data-Ora')            # Potential evapotranspiration in mm

ev, stemflow, through, ec, etc, cc_0, cc_rain, cc_drain, cc_end, ctc_0, ctc_rain, ctc_drain, ctc_end, can_drain, trunk_drain = rutter_sparse(0.9, 0.1, 0.5, 0.023, 0.5, rain_series, et_series)

rain_series['Potential_et'] = et_series.values      # Potential evapotranspitation in mm
rain_series['Evaporation'] = ev                     # Efective evapotranspiration in mm
rain_series['Stemflow'] = stemflow                  # Stemflow in mm
rain_series['through'] = through                    # through (water that effectively reaches the soil) in mm
rain_series['cc_0'] = cc_0
rain_series['cc_rain'] = cc_rain
rain_series['can_drain'] = can_drain
rain_series['cc_drain'] = cc_drain
rain_series['ec'] = ec
rain_series['cc_end'] = cc_end
rain_series['ctc_0'] = ctc_0
rain_series['ctc_rain'] = ctc_rain
rain_series['trunk drain'] = trunk_drain
rain_series['ctc_drain'] = ctc_drain
rain_series['etc'] = etc
rain_series['ctc_end'] = ctc_end


rain_series.to_csv('./../Results/'+station+'_rain.csv')         # All units in mm

##################  Compute infiltration  ########################



Ks = 2.54                   # Introduce the saturated hydraulic conductivity [cm/h]
Ks = Ks/2.54                # Convert the saturated hydraulic conductivity to in/hr
gam_s = 6.5                 # Introduce the capillary suction head [cm]
gas_s = gam_s/2.54          # Convert the capillary suction head to in
IMDmax = 0.3                # Introduce the maximum initial moisture deficit [-]
dt = 1                      # Time step [hours]


Tr = 4.5/(Ks**0.5)                                  # Time interval required to consider events as independent (Eq. 4-37 SWMM Hydrology) [hours]
ia = rain_series['through'].values/25.4             # There was an error in here because the through column in rain 
                                                    # series is in mm not in cm. (Incorrect: values/2.54; Correct: values/25.4). ia [inches]

infil= {
    'IMDmax': IMDmax,           # [-]
    'IMD': IMDmax,              # [-]
    "Fu": 0,                    # [in]
    "F": 0,                     # [in]
    "Sat": False,               # Saturated or not saturated
    'T': 0,                     
    'Ks': Ks,                   # Saturated HC [in/hr]
    'Lu': (4*Ks**0.5),          # Eq. 4-33 SWMM Hydrology [inches]
    'S': gam_s                  # Capillary suction head [inches]
}


f = np.zeros(len(ia))                                       # Infiltration rate [in/hour] as I use timestep = 1 hour f is also the infiltration [in]
for i in range(len(ia)):            
    Fumax = infil['IMDmax']*infil['Lu']                     # [in]
    infil['T'] = infil['T'] - dt                            # [in]
    if infil['Sat'] == True:
        f[i] = grn_ampt_getSatInfil(ia[i], infil, Fumax, dt, Tr)    
    else:
        f[i] = grn_ampt_getUnsatInfil(ia[i], infil, Fumax, dt, Tr)


rain_series['infiltration'] = f*25.4                        # Save infiltration in milimeters
rain_series.to_csv('./../Results/'+station+'_infiltration.csv')