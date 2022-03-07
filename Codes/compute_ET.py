from functions_physical import ET

station = 'Molteno'
var_t = '9017_1'
var_rh = '11020_1'
et_data = ET('./../joined_data/'+station+'/'+var_t+'.csv', './../joined_data/'+station+'/'+var_rh+'.csv')

et_data.to_csv('./../joined_data/'+station+'/ET.csv')


