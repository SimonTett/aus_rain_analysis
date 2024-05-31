# Description of code used in Australian Radar analysis



## Data processing
1) submit_process_reflectivity.sh -- bash script to run on gadi
    
   Runs process_reflectivity.py in several parallel jobs which processes the reflectivity data.
   Main thing are the max values at different time-averaging periods but other useful fields are output.
   Produces monthly output.  Takes cira 6 hours to run abotu 25 years of data.

2) comps_sea_avg_mask.py -- processes the monthly  output to seasonal and applies masking.
      masks are Beam blockage, land and enough samples.  

3) comp_events.py -- processes the seasonal data to give dataset of events. 
