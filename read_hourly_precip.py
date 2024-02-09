import pathlib
def read_meta_data(file:pathlib.Path) -> dict:
    """
      read meta data from houry precip file
      {'Station ID': ' AU_090180',
 'Country': ' Australia',
 'Original Station Number': ' 090180',
 'Original Station Name': ' AIREYS INLET',
 'Path to original data': ' B:/INTENSE data/Original data/Australia/AWS 1min_VIC.zip/HD01D_Data_090180_999999998747801.txt',
 'Latitude': ' -38.4583',
 'Longitude': ' 144.0883',
 'Start datetime': ' 2007072014',
 'End datetime': ' 2015082009',
 'Elevation': ' 95.0m',
 'Number of records': ' 70868',
 'Percent missing data': ' 7.45',
 'Original Timestep': ' 1min',
 'New Timestep': ' 1hr',
 'Original Units': ' mm',
 'New Units': ' mm',
 'Time Zone': ' CET',
 'Daylight Saving info': ' NA',
 'No data value': ' -999',
 'Resolution': ' 0.20',
 'Other': ''}

    """
    result=dict()
    with open(file, 'r') as fh:
        while True:
            key,value = fh.readline().strip().split(":",maxsplit=1)
            value = value.strip()
            result[key]=value
            if key == 'Other': # end fo header
                break # all done

    # convert same values to loads
    for key in ['Latitude','Longitude','Resolution','Percent missing data','No data value']:
        result[key]=float(result[key])
    # integer vales
    for key in ['Number of records']:
        result[key]=int(result[key])
    # convert ht.
    result['height']=float(result['Elevation'][0:-1])
    return result


def gen_time_range(start,end,no):
    import pandas as pd
    start=start[0:4]+"/"+start[4:6]+"/"+start[6:8]+"T"+start[8:]
    end=end[0:4]+"/"+end[4:6]+"/"+end[6:8]+"T"+end[8:]
    time_range = pd.date_range(start,end,no)
    return time_range
