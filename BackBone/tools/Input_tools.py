from BackBone import config as cfg
from pathlib import Path, PosixPath
import pandas as pd
import os
import itertools
import numpy as np
from datetime import datetime

### World map packs
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster

# map of customers 
def map():

    try:
        data = pd.read_pickle(cfg.DATA_PICKLE_PATH)
    except:
        print("Input data missing in directory: " + str(cfg.FILENAME_FOLDER_PATH))

    data['Customer_Countries']['LatLong'] = data['Customer_Countries']['CountryText'].apply(geolocate)
    data['Customer_Countries'][['Latitude', 'Longitude']] = pd.DataFrame(data['Customer_Countries']['LatLong'].tolist(), index=data['Customer_Countries'].index)

    df = data['Customer_Countries'].dropna(axis = 0, how ='any')
    df = df[df['Affiliate_type'] == 1]
    df = df.merge(data['GSF'][data['GSF']['Fiscal year']==2022].groupby('Country_ID').sum()['Volume_U3'], how='left', left_on='Customer_Id', right_on='Country_ID')

    # #empty map
    map = folium.Map(tiles="cartodbpositron", control_scale=False)

    max = df[['Latitude', 'Longitude']].max().values.tolist()
    min = df[['Latitude', 'Longitude']].min().values.tolist()

    #for each coordinate, create circlemarker of user percent
    for i in range(len(df)):
        lat = df.iloc[i]['Latitude']
        long = df.iloc[i]['Longitude']
        radius=2
        popup_text = """Country : {}<br>
                    Demand : {}<br>"""
        popup_text = popup_text.format(df.iloc[i]['Customer_Id'],
                                   df.iloc[i]['Volume_U3']
                                   )
        folium.CircleMarker(location = [lat, long], height= 6, width= 8, radius=radius, popup= popup_text, fill =False).add_to(map)
    map.fit_bounds([min,max])

    return map

# Find geolocation of a country
def geolocate(country):
    try:
        geolocator = Nominatim(user_agent='map')
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return np.nan


def customers():
    customer_text = pd.read_excel(cfg.FILENAME_PATH, sheet_name= 'Customer_Countries')
    customer_text['Customer_Id'] = customer_text['Customer_Id'].astype(str)
    customer_text = customer_text.groupby(['Customer_Id','CountryText']).sum()['Affiliate_type'].reset_index()

    return customer_text