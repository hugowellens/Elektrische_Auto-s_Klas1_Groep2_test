#!/usr/bin/env python
# coding: utf-8

# # Elektrische voertuigen en laadpalen

# ## Code

# In[4]:


##################################################### Importeren packages #####################################################

import pandas as pd
import requests
import json
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import folium
from statsmodels.formula.api import ols
from fuzzywuzzy import process
import statsmodels.api as sm
import plotly.figure_factory as ff
import streamlit as st
from streamlit_folium import folium_static

###################################################### Inladen Data ###########################################################
###################### functie data inladen in streamlit #########################################################################################
st.set_page_config(layout = 'wide')

csv = 'Elektrische_voertuigen.csv'

@st.cache
def load_data(Boolean):
    data = pd.read_csv('Elektrische_voertuigen.csv', low_memory= Boolean)
    return data

elektrisch  = load_data(False)

# Inladen data
laadpaal = pd.read_csv('laadpaaldata.csv')
elektrisch_origineel = elektrisch
laadpaal_origineel = laadpaal

# API inladen
url = "https://api.openchargemap.io/v3/poi"
params = {"countrycode": "NL", "output": "json", "compact": True, "verbose": False, "maxresults": 1000}
headers = {"Content-Type": "application/json", "X-API-Key": "89b2da87-ca59-4004-b45a-2bfc6ce65076"}
response = requests.request("GET", url, headers=headers, params=params)

# Json file maken
json = response.json()

# Dataframe van json file maken
Open_Charge_Map = pd.json_normalize(json)

# Address dictionary als kolommen toevoegen
Connections1 = pd.json_normalize(Open_Charge_Map.Connections)
Connections = pd.json_normalize(Connections1[0])
Laadpalen_map = pd.concat([Open_Charge_Map, Connections], axis = 1)
Laadpalen_map_origineel = Laadpalen_map


############################################# Data cleanen laadpalen dataset ##################################################

# Kolommen selecteren
Laadpalen_map = Laadpalen_map[['ID', 'UUID',
       'DataProviderID', 'NumberOfPoints', 'DateLastStatusUpdate',
       'DataQualityLevel', 'DateCreated', 'AddressInfo.AddressLine1',
       'AddressInfo.Town', 'AddressInfo.StateOrProvince',
       'AddressInfo.Postcode', 'AddressInfo.Latitude',
       'AddressInfo.Longitude', 'AddressInfo.ContactTelephone1', 'UsageCost','PowerKW', 'Quantity']]

# Datelastverified naar datum
Laadpalen_map['DateCreated'] = pd.to_datetime(Laadpalen_map['DateCreated'], format = "%Y-%m-%dT%H:%M:%SZ")
Laadpalen_map['DateLastStatusUpdate'] = pd.to_datetime(Laadpalen_map['DateLastStatusUpdate'], format = "%Y-%m-%dT%H:%M:%SZ")

# Van DataProviderID & DataQualityLevel datatype 'category' maken
Laadpalen_map['DataProviderID'] = Laadpalen_map['DataProviderID'].astype('category')
Laadpalen_map['DataQualityLevel'] = Laadpalen_map['DataQualityLevel'].astype('category')

# Telefoonnummers gelijk maken en niet korter als 10 digits.
Laadpalen_map['AddressInfo.ContactTelephone1'] = Laadpalen_map['AddressInfo.ContactTelephone1'].str.replace("+", "00")
Laadpalen_map['AddressInfo.ContactTelephone1'] = Laadpalen_map['AddressInfo.ContactTelephone1'].str.replace("-", "")
Laadpalen_map['AddressInfo.ContactTelephone1'] = Laadpalen_map['AddressInfo.ContactTelephone1'].str.replace(" ", "")
Laadpalen_map['AddressInfo.ContactTelephone1'] = Laadpalen_map['AddressInfo.ContactTelephone1'].str.replace("(", "")
Laadpalen_map['AddressInfo.ContactTelephone1'] = Laadpalen_map['AddressInfo.ContactTelephone1'].str.replace(")", "")
digits = Laadpalen_map['AddressInfo.ContactTelephone1'].str.len()
Laadpalen_map.loc[digits < 10, "AddressInfo.ContactTelephone1"] = np.nan

# Postcode
Laadpalen_map['AddressInfo.Postcode'] = Laadpalen_map['AddressInfo.Postcode'].str.replace(" ", "")
digits_postcode = Laadpalen_map['AddressInfo.Postcode'].str.len()
Laadpalen_map.loc[digits_postcode < 6, "AddressInfo.Postcode"] = np.nan

# Provincies
Provincies = ['Noord-Holland', 'Zuid-Holland', 'Zeeland', 'Friesland', 'Utrecht', 'Flevoland', 'Noord-Brabant',
            'Groningen', 'Drenthe', 'Overijssel', 'Gelderland', 'Limburg']

# NaN waardes opvullen zodat er geen foutmelding ontstaat
Laadpalen_map['AddressInfo.StateOrProvince'] = Laadpalen_map['AddressInfo.StateOrProvince'].fillna("")

# Voor elke provincie wordt er een match gezocht en voor elke match wordt er gekeken of hij hoger is als 80 om ze te laten vervangen
for province in Provincies:
    matches = process.extract(province, Laadpalen_map['AddressInfo.StateOrProvince'], limit = Laadpalen_map.shape[0])
    for potential_match in matches:
        if potential_match[1] >= 85:
            Laadpalen_map.loc[Laadpalen_map['AddressInfo.StateOrProvince'] == potential_match[0], 'AddressInfo.StateOrProvince'] = province

# Hier zien wij dat sommige niet goed zijn vervangen
Laadpalen_map['AddressInfo.StateOrProvince'] = Laadpalen_map['AddressInfo.StateOrProvince'].str.replace("South Holland", "Zuid-Holland")
Laadpalen_map['AddressInfo.StateOrProvince'] = Laadpalen_map['AddressInfo.StateOrProvince'].str.replace("ZH", "Zuid-Holland")
Laadpalen_map['AddressInfo.StateOrProvince'] = Laadpalen_map['AddressInfo.StateOrProvince'].str.replace("NH", "Noord-Holland")
Laadpalen_map['AddressInfo.StateOrProvince'] = Laadpalen_map['AddressInfo.StateOrProvince'].str.replace("Holandia Północna", "Noord-Holland")
Laadpalen_map['AddressInfo.StateOrProvince'] = Laadpalen_map['AddressInfo.StateOrProvince'].str.replace("Stellendam", "Zuid-Holland")
Laadpalen_map['AddressInfo.StateOrProvince'] = Laadpalen_map['AddressInfo.StateOrProvince'].str.replace("FRL", "Friesland")
Laadpalen_map['AddressInfo.StateOrProvince'] = Laadpalen_map['AddressInfo.StateOrProvince'].str.replace("Stadsregio Arnhem Nijmegen", "Gelderland")

# Punt die niet in Nederland ligt verwijderen 
Laadpalen_map = Laadpalen_map[Laadpalen_map['AddressInfo.AddressLine1'] != 'A16 highway2952 AD ']

################################################### Color producer ###########################################################

# Color producer definieren
def color_producer(provincie):
    if provincie == 'Noord-Holland':
        return 'red'
    elif provincie == 'Zuid-Holland':
        return 'blue' 
    elif provincie == 'Zeeland':
        return 'yellow'  
    elif provincie == 'Friesland':
        return 'orange'  
    elif provincie == 'Flevoland':
        return 'violet'  
    elif provincie == 'Utrecht':
        return 'cyan'  
    elif provincie == 'Noord-Brabant':
        return 'turquoise'  
    elif provincie == 'Groningen':
        return 'gray'  
    elif provincie == 'Drenthe':
        return 'pink'  
    elif provincie == 'Overijssel':
        return 'black'  
    elif provincie == 'Gelderland':
        return 'brown'  
    elif provincie == 'Limburg':
        return 'olive'  
    
################################################### Legenda Code ##########################################################

# Code van legenda
def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))
    
    legend_categories = ""     
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"
        
    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """
   

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map

################################################## Dataset per provincie ##################################################

# Per provincie nieuwe datasets
Laadpalen_Noord_Holland = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Noord-Holland']
Laadpalen_Zuid_Holland = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Zuid-Holland']
Laadpalen_Flevoland = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Flevoland']
Laadpalen_Zeeland = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Zeeland']
Laadpalen_Friesland = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Friesland']
Laadpalen_Groningen = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Groningen']
Laadpalen_Utrecht = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Utrecht']
Laadpalen_Noord_Brabant = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Noord-Brabant']
Laadpalen_Limburg = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Limburg']
Laadpalen_Overijssel = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Overijssel']
Laadpalen_Drenthe = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Drenthe']
Laadpalen_Gelderland= Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == 'Gelderland']

################################################# Laadpaal dataset cleanen #################################################

# Chargetime kan niet negatief zijn
laadpaal = laadpaal[laadpaal['ChargeTime'] >= 0]

# Started en ended naar datum formaat
laadpaal['Started'] = pd.to_datetime(laadpaal['Started'], errors='coerce')
laadpaal['Ended'] = pd.to_datetime(laadpaal['Ended'], errors='coerce')

# Outliers

# ChargeTime:
# Q1 en Q3 definieren
q1_chart = laadpaal.ChargeTime.quantile(0.25)
q3_chart = laadpaal.ChargeTime.quantile(0.75)

# IQR Chargetime
iqr_chart = q3_chart-q1_chart

# Outliers ChargeTime eruit halen
outlier_chart = (laadpaal.ChargeTime <= q3_chart + 1.5*iqr_chart)
laadpaal = laadpaal.loc[outlier_chart]

# ConnectedTime:
# Q1 en Q3 definieren
q1_cont = laadpaal.ConnectedTime.quantile(0.25)
q3_cont = laadpaal.ConnectedTime.quantile(0.75)

# IQR ConnectedTime
iqr_cont = q3_cont-q1_cont

# Outliers ConnectedTime eruit halen
outlier_cont = (laadpaal.ConnectedTime <= q3_cont + 1.5*iqr_cont)
laadpaal = laadpaal.loc[outlier_cont]

# TotalEnergy:
# Q1 en Q3 definieren
q1_te = laadpaal.TotalEnergy.quantile(0.25)
q3_te = laadpaal.TotalEnergy.quantile(0.75)

# IQR TotalEnergy
iqr_te = q3_te-q1_te

#Outliers TotalEnergy eruit halen
outlier_te = (laadpaal.TotalEnergy <= q3_te + 1.5*iqr_te)
laadpaal = laadpaal.loc[outlier_te]

######################################################## Regressie ###########################################################

# Getransformeerde kolommen
laadpaal['TotalEnergy_qdrt'] = laadpaal['TotalEnergy']**0.25
laadpaal['ChargeTime_qdrt'] = laadpaal['ChargeTime']**.25

# Model maken
mdl_totalenergy_vs_chargetime = ols('ChargeTime_qdrt ~ TotalEnergy_qdrt', data = laadpaal).fit()

# Voorspellende data
explanatory_data = pd.DataFrame({"TotalEnergy_qdrt": np.arange(0, 17250, 250) ** 0.25,
                                 "TotalEnergy": np.arange(0, 17250, 250)})

# Voorspelde data
prediction_data = explanatory_data.assign(ChargeTime_qdrt = mdl_totalenergy_vs_chargetime.predict(explanatory_data))
prediction_data["ChargeTime"] = prediction_data["ChargeTime_qdrt"]**4

# Toekomst voorspellen
little_laadpaal = pd.DataFrame({'TotalEnergy_qdrt':np.arange(11.5, 20, 0.3)})
pred_little_laadpaal = little_laadpaal.assign(ChargeTime_qdrt = mdl_totalenergy_vs_chargetime.predict(little_laadpaal))

print(f'R^2: {round(mdl_totalenergy_vs_chargetime.rsquared, 3)}')

############################################ Nieuwe kolom overbodige connected time ########################################

# Overbodige Connected Time berekenen dat groter is als 0
laadpaal['Overbodige_connectedtime'] = laadpaal.ConnectedTime-laadpaal.ChargeTime
laadpaal = laadpaal[laadpaal['Overbodige_connectedtime'] >= 0] 

# laadpaaldata per seizoen
laadpaal_winter1 = laadpaal[(laadpaal['Started'] > "2018-01-01") & (laadpaal['Started'] < "2018-03-20")]
laadpaal_lente = laadpaal[(laadpaal['Started'] > "2018-03-21") & (laadpaal['Started'] < "2018-06-21")]
laadpaal_zomer = laadpaal[(laadpaal['Started'] > "2018-06-22") & (laadpaal['Started'] < "2018-09-23")]
laadpaal_herfst = laadpaal[(laadpaal['Started'] > "2018-09-24") & (laadpaal['Started'] < "2018-12-21")]
laadpaal_winter2 = laadpaal[(laadpaal['Started'] > "2018-12-22") & (laadpaal['Started'] < "2018-12-31")]
laadpaal_winter = laadpaal_winter1.append(laadpaal_winter2)

########################################### Cleanen elektrische voertuigen data #############################################

# Kolommen selecteren
elektrisch = elektrisch[['Kenteken','Voertuigsoort', 'Merk', 'Handelsbenaming', 'Vervaldatum APK', 'Datum tenaamstelling', 'Inrichting', 'Cilinderinhoud', 'Massa ledig voertuig', 'Datum eerste toelating', 'Catalogusprijs', 'Vermogen massarijklaar']]

# Merken veranderen
elektrisch['Merk'] = elektrisch['Merk'].str.replace("TESLA MOTORS", "TESLA")
elektrisch['Merk'] = elektrisch['Merk'].str.replace("BMW I", "BMW")
elektrisch['Merk'] = elektrisch['Merk'].str.replace("M.A.N.", "MAN")
elektrisch['Merk'] = elektrisch['Merk'].str.replace("JAGUAR CARS", "JAGUAR")
elektrisch['Merk'] = elektrisch['Merk'].str.replace("VOLKSWAGEN/ZIMNY", "VW")

# Inrichtingen veranderen
elektrisch['Inrichting'] = elektrisch['Inrichting'].str.replace("niet nader aangeduid", "Niet geregistreerd")
elektrisch['Inrichting'] = elektrisch['Inrichting'].str.replace("kampeerwagen", "Overig")
elektrisch['Inrichting'] = elektrisch['Inrichting'].str.replace("voor rolstoelen toegankelijk voertuig", "Overig")
elektrisch['Inrichting'] = elektrisch['Inrichting'].str.replace("speciale groep", "Overig")
elektrisch['Inrichting'] = elektrisch['Inrichting'].str.replace("gesloten opbouw", "Overig")
elektrisch['Inrichting'] = elektrisch['Inrichting'].str.replace("lijkwagen", "Overig")

# Datum veranderen naar datetime
elektrisch['Datum eerste toelating'] = pd.to_datetime(elektrisch['Datum eerste toelating'], format = '%Y%m%d')
elektrisch['Datum tenaamstelling'] = pd.to_datetime(elektrisch['Datum tenaamstelling'], format = '%Y%m%d')

# Kijken of het 1e eigenaar is of niet
elektrisch['Eerste eigenaar'] = 0

for lab, row in elektrisch.iterrows():
    if row["Datum eerste toelating"] == row["Datum tenaamstelling"]:
        elektrisch.loc[lab, "Eerste eigenaar"] = 0
    else: elektrisch.loc[lab, "Eerste eigenaar"] = 1

# NA Waardes opvullen
elektrisch['Cilinderinhoud'] = elektrisch['Cilinderinhoud'].fillna('0')

# Hybride kolom maken
elektrisch['Hybride'] = 0

for lab, row in elektrisch.iterrows():
    if row["Cilinderinhoud"] != '0':
        elektrisch.loc[lab, "Hybride"] = 1
    else: elektrisch.loc[lab, "Hybride"] = 0
        




# In[3]:
##################################################### STREAMLIT ###############################################################
st.title("Elektrische Auto's en hun laadstations")

st.write("In deze blog gaan wij kijken naar elektrische auto's en hun laadstations. Hierbij maken wij onderscheidt in provincie, kijken wij naar de laadtijd, kijken wij of ze langer aan het station staan dan dat nodig is en kijken wij door de jaren heen naar het aantal elektrische auto's en de catalogusprijs hiervan.")


################################# Datasets weergeven m.b.v een selectbox #########################################################

Input_dataset = st.sidebar.selectbox('Selecteer dataset', ('Originele elektrisch', 'Gecleande elektrisch', 'Originele laadpaal', 'Gecleande laadpaal', 'Originele Laadpalen_map', 'Gecleande Laadpalen_map'))

if Input_dataset == 'Originele elektrisch':
	st.write(f'{Input_dataset} dataset:')
	st.dataframe(elektrisch_origineel.head())
elif Input_dataset == 'Gecleande elektrisch':
	st.write(f'{Input_dataset} dataset:')
	st.dataframe(elektrisch.head())
elif Input_dataset == 'Originele laadpaal':
	st.write(f'{Input_dataset} dataset:')
	st.dataframe(laadpaal_origineel.head())
elif Input_dataset == 'Gecleande laadpaal':
	st.write(f'{Input_dataset} dataset:')
	st.dataframe(laadpaal.head())
elif Input_dataset == 'Originele Laadpalen_map':
	st.write(f'{Input_dataset} dataset:')
	st.dataframe(Laadpalen_map_origineel.head())
elif Input_dataset == 'Gecleande Laadpalen_map':
	st.write(f'{Input_dataset} dataset:')
	st.dataframe(Laadpalen_map.head())



##################################################### KAART ###############################################################

st.header('Laadstations per provincie')


m = folium.Map(location = [52.3702157, 4.8951679], zoom_start = 7)
	
# Er wordt per provincie folium featuregroups aangemaakt
Noord_Holland = folium.FeatureGroup(name="Noord-Holland", show=False).add_to(m)
Zuid_Holland = folium.FeatureGroup(name="Zuid-Holland").add_to(m)
Flevoland = folium.FeatureGroup(name="Flevoland", show=False).add_to(m)
Zeeland = folium.FeatureGroup(name="Zeeland", show=False).add_to(m)
Friesland = folium.FeatureGroup(name="Friesland", show=False).add_to(m)
Groningen = folium.FeatureGroup(name="Groningen", show=False).add_to(m)
Utrecht = folium.FeatureGroup(name="Utrecht", show=False).add_to(m)
Noord_Brabant = folium.FeatureGroup(name="Noord-Brabant", show=False).add_to(m)
Limburg = folium.FeatureGroup(name="Limburg", show=False).add_to(m)
Overijssel = folium.FeatureGroup(name="Overijssel", show=False).add_to(m)
Drenthe = folium.FeatureGroup(name="Drenthe", show=False).add_to(m)
Gelderland = folium.FeatureGroup(name="Gelderland", show=False).add_to(m)

# Voor elk laadpunt per provincie een marker aanmaken in de kleur van de provincie
for index, row in Laadpalen_Noord_Holland.iterrows():
    Noord_Holland.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                         popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
    		                                color= color_producer(row['AddressInfo.StateOrProvince']),
        		                        	        fill=True).add_to(m))
	
for index, row in Laadpalen_Zuid_Holland.iterrows():
	Zuid_Holland.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                           popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
                                            color= color_producer(row['AddressInfo.StateOrProvince']),
                                            fill=True).add_to(m))

for index, row in Laadpalen_Zeeland.iterrows():
	Zeeland.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                           popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
        	                                   color= color_producer(row['AddressInfo.StateOrProvince']),
        	                                   fill=True).add_to(m))

for index, row in Laadpalen_Friesland.iterrows():
	Friesland.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                            popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
        	                                    color= color_producer(row['AddressInfo.StateOrProvince']),
                                            fill=True).add_to(m))

for index, row in Laadpalen_Flevoland.iterrows():
	Flevoland.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
       	                                    popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
                	                           color= color_producer(row['AddressInfo.StateOrProvince']),
                  		                   fill=True).add_to(m))
    
for index, row in Laadpalen_Utrecht.iterrows():
    	Utrecht.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
      		                                    popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
                 	                           color= color_producer(row['AddressInfo.StateOrProvince']),
                         		                   fill=True).add_to(m))
	
for index, row in Laadpalen_Noord_Brabant.iterrows():
	Noord_Brabant.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
							popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
							color= color_producer(row['AddressInfo.StateOrProvince']),
 							fill=True).add_to(m))
    
for index, row in Laadpalen_Groningen.iterrows():
	Groningen.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
							popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
							color= color_producer(row['AddressInfo.StateOrProvince']),
							fill=True).add_to(m))
	
for index, row in Laadpalen_Drenthe.iterrows():
	Drenthe.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                            popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
                                            color= color_producer(row['AddressInfo.StateOrProvince']),
                                            fill=True).add_to(m))

for index, row in Laadpalen_Overijssel.iterrows():
	Overijssel.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                            popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
                                            color= color_producer(row['AddressInfo.StateOrProvince']),
                                            fill=True).add_to(m))
    
for index, row in Laadpalen_Gelderland.iterrows():
	Gelderland.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                            popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
                                            color= color_producer(row['AddressInfo.StateOrProvince']),
                                            fill=True).add_to(m))
    
for index, row in Laadpalen_Limburg.iterrows():
	Limburg.add_child(folium.CircleMarker(location= [row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                                            popup= '<strong>' + row['AddressInfo.AddressLine1'] + '<strong>',
                                            color= color_producer(row['AddressInfo.StateOrProvince']),
                                            fill=True).add_to(m))
    
# Layercontrol toevoegen
folium.LayerControl(position='bottomleft', collapsed=False).add_to(m)

# Legenda toevoegen
m = add_categorical_legend(m, 'Provincie',
                            colors = ['red', 'blue', 'yellow', 'orange', 'violet', 'cyan', 'turquoise', 'gray', 'pink', 'black', 'brown', 'olive'],
                         labels = ['Noord-Holland', 'Zuid-Holland', 'Zeeland', 'Friesland', 'Flevoland', 'Utrecht', 'Noord-Brabant', 'Groningen', 'Drenthe', 'Overijssel', 'Gelderland', 'Limburg'])                                                        
                                                        
folium_static(m)

st.write("In deze kaart zijn de verschillende laadstations van de elektrische auto's te zien. Deze zijn onderverdeeld op kleur per provincie. Wat op te merken is, is dat er veel laadstations liggen in het westen van het land. Hier wonen ook veel meer mensen, dus is dat goed te begrijpen. Voor de precieze locatie van het laadstation, kunt u op de punten klikken!")

################################################ Provincies tegen PowerKW ####################################################
a1, a2 = st.columns(2, gap="medium")

with a1:
	# Figuur maken
	fig = go.Figure()
	
	# Voor elke provincie een boxplot aanmaken
	for provincie in ['Noord-Holland', 'Zuid-Holland', 'Zeeland', 'Friesland', 'Flevoland', 'Utrecht', 'Noord-Brabant', 'Groningen', 'Drenthe', 'Overijssel', 'Gelderland', 'Limburg']:
	    df = Laadpalen_map[Laadpalen_map['AddressInfo.StateOrProvince'] == provincie]
	    fig.add_trace(go.Box(x = df['AddressInfo.StateOrProvince'], y = df['PowerKW'], name = provincie))
	
	# Updaten layout
	fig.update_layout(title="Power KW per provincie", legend_title="Provincie",)
	fig.update_xaxes(title_text = 'Provincie')
	fig.update_yaxes(title_text = 'Power (KW)')
	
	st.plotly_chart(fig, use_container_width = True)
	st.write("Hier is per provincie de maximale power te zien in kilowatt. In het plaatje is te zien dat de spreidingsbreedte van Drenthe heel groot is. Dit komt doordat er niet veel waarnemingen zijn in Drenthe, waardoor het niet wordt gezien als uitschieter. Verder is er niet een heel groot verschil te zien met vermogen dat de stations leveren. Wat een goed iets is, aangezien er dan privileges ontstaan in verschillende provincies.")


############################################## Chargetime van auto's ##########################################################



with a2:
	# Histogram van de ChargeTime tegen het aantal keer dat hij voorkomt
	hist_data = [laadpaal.ChargeTime]
	groep_labels = ['Laadtijd']
	
	fig1 = ff.create_distplot(hist_data, groep_labels, bin_size = 0.1, curve_type = 'normal', show_rug = False)
	
	# Lijnen toevoegen bij het gemiddelde en de mediaan
	fig1.add_trace(go.Scatter(x = [laadpaal.ChargeTime.mean(), laadpaal.ChargeTime.mean()], y = [0, 0.75], 
	                        mode = 'lines', line = {'color':'red'}, name = 'Gemiddelde'))
	fig1.add_trace(go.Scatter(x = [laadpaal.ChargeTime.median(), laadpaal.ChargeTime.median()], y = [0, 0.75], 
	                        mode = 'lines', line = {'color':'green'}, name = 'Mediaan'))
	
	# Annotatie bij het gemiddelde en mediaan toevoegen
	annotation_gem = [{'x': laadpaal.ChargeTime.mean(), 'y':0.65, 
	                  'showarrow': True, 'arrowhead': 4, 'arrowcolor':'black', 
	                    'font': {'color': 'black', 'size':15}, 'text': 'Gemiddelde'}]
	annotation_median = [{'x': laadpaal.ChargeTime.median(), 'y':0.65, 'showarrow': True, 'arrowhead': 4,
	                    'font': {'color': 'black', 'size':15}, 'text': 'Mediaan'}]
	
	# Button voor het kiezen van de mediaan of het gemiddelde
	buttons = [ 
	{'label': "Gemiddelde", 'method': "update", 'args': [{"visible": [True, True, True, False]}, {"annotations": annotation_gem}]},
	{'label': "Mediaan", 'method': "update", 'args': [{"visible": [True, True, False, True]}, {"annotations": annotation_median}]}
	]
	
	# Titel en labels toevoegen. 
	fig1.update_layout({
	    'updatemenus':[{
	            'type': "buttons",
	            'direction': 'down',
	            'x': 1.15,'y': 0.7, 'buttons': buttons
	            }]})
	fig1.update_layout(title_text = 'Histogram over de laadtijd', xaxis_title = 'Laadtijd (uur)', yaxis_title = 'Aantal')
	
	st.plotly_chart(fig1, use_container_width = True)
	st.write("In dit histogram is de laadtijd te zien. Hierbij kijken wij naar hoe vaak een laadtijd precies voorkomt. Er is een button toegevoegd om hiernaar preciezer te kijken. Zo is er een button waarbij u 			de gemiddelde laadtijd kunt zien, en een button waarbij u de mediaan kunt zien. Verder is er ook een normale verdeling curve te zien. Als u hier naar kijkt ziet u dat het histogram, op een paar uitzonderingen na, erg op een normale verdeling begint te lijken.")

###################################################### Regressie ###############################################################
b1, b2 = st.columns(2, gap="medium")

with b1:
	# Plot the transformed variables
	fig2 = go.Figure()
	
	# Dropdown buttons
	dropdown_buttons = [
	    {'label': 'Zonder transformatie', 'method':'update',
	    'args':[{'visible':[True, False, False, False]}, {'title':'Zonder transformatie', 'xaxis': {'title':'Totale energie (W/uur)'}, 'yaxis': {'title':'Totale energie (W/uur)'}}]},
	    {'label': 'Met transformatie', 'method':'update',
	    'args':[{'visible':[False, True, False, False]}, {'title':'Met transformatie', 'xaxis': {'title':'Totale energie**0.25 (W/uur)'}, 'yaxis': {'title':'Laadtijd**0.25 (uren)'}}]},
	    {'label': 'Voorspelling nu', 'method':'update',
	    'args':[{'visible':[False, True, True, False]}, {'title':'Voorspelling nu', 'xaxis': {'title':'Totale energie**0.25 (W/uur)'}, 'yaxis': {'title':'Laadtijd**0.25 (uren)'}}]},
	    {'label': 'Voorspelling toekomst', 'method':'update',
	    'args':[{'visible':[False, True, True, True]}, {'title':'Voorspelling toekomst', 'xaxis': {'title':'Totale energie**0.25 (W/uur)'}, 'yaxis': {'title':'Laadtijd**0.25 (uren)'}}]}    
	]
	
	# Traces toevoegen
	fig2.add_trace(go.Scatter(x = laadpaal.TotalEnergy, y = laadpaal.ChargeTime, opacity = 0.8, mode = 'markers', name = 'Zonder transformatie', visible = True))
	fig2.add_trace(go.Scatter(x=laadpaal["TotalEnergy_qdrt"], y=laadpaal["ChargeTime_qdrt"], opacity= 0.8, mode = 'markers', name = 'Getransformeerd', visible = False))
	fig2.add_trace(go.Scatter(x=prediction_data["TotalEnergy_qdrt"], y=prediction_data["ChargeTime_qdrt"], mode = 'markers', name = 'Voorspelling nu', visible = False))
	fig2.add_trace(go.Scatter(x=pred_little_laadpaal["TotalEnergy_qdrt"], y=pred_little_laadpaal["ChargeTime_qdrt"], mode = 'markers', name = 'Voorspelling Toekomst', visible = False))
	
	# Updaten layout
	fig2.update_layout({'updatemenus':[{'type': 'dropdown',
	                                 'x':1.3, 'y':0.8,
	                                 'showactive':True,
	                                 'active': 0,
	                                 'buttons':dropdown_buttons}]},
	                  title_text = 'Totale energie tegen de laadtijd'
	                 )
	
	st.plotly_chart(fig2, use_container_width = True)
	st.write("Vervolgens is er een regressie gedaan over de totale energie tegen de laadtijd. In het eerste figuur, van de dropdown, kunt u de relatie zien tussen deze 2 variabelen zonder enige transformatie. In het tweede figuur, is deze getransformeerd door de variabelen tot de macht 0.25 te zetten. Vervolgens hebben wij aan de hand van deze transformatie een model gemaakt en hebben wij punten voorspeld. In het 3e figuur kunt u de voorspelde waarden van de al gegeven datasets zien. In het 4e figuur is ook de voorspelde waarden van de toekomst te zien.")

############################################# Overbodige ConnectedTime per seizoen ###########################################


with b2:
	# Figuur maken
	fig3 = go.Figure()
	
	# Traces toevoegen
	fig3.add_trace(go.Histogram(x = laadpaal_zomer.Overbodige_connectedtime, name = 'Zomer'))
	fig3.add_trace(go.Histogram(x = laadpaal_winter.Overbodige_connectedtime, name = 'winter'))
	fig3.add_trace(go.Histogram(x = laadpaal_herfst.Overbodige_connectedtime, name = 'Herfst'))
	fig3.add_trace(go.Histogram(x = laadpaal_lente.Overbodige_connectedtime, name = 'Lente'))
	
	# Figuur updaten
	fig3.update_layout(title_text = 'Histogram over de tijd dat een auto langer dan de benodigde laadtijd aan een laadpaal zit.') 
	fig3.update_xaxes(title_text = 'De overbodige tijd dat een auto aan de laadpaal zit.')
	fig3.update_yaxes(title_text = 'Het aantal keer overbodige tijd') 
	
	st.plotly_chart(fig3, use_container_width = True)
	st.write("In dit histogram is het aantal overbodige laadtijd te zien per seizoen. Dat is de tijd dat de auto nog in het laadstation zit waarbij hij niet meer oplaad. Door dit te visualiseren is er per auto de overbodige laadtijd berekend en is dit onderverdeeld per seizoen. Uit het histogram is af te lezen dat er in de herfst de auto's er over het algemeen het langst overbodig in zitten.")
########################################### Het aantal voertuigen per maand ###############################################
c1, c2 = st.columns(2, gap="medium")

with c1:
	# Per Maand
	by_month = pd.to_datetime(elektrisch['Datum eerste toelating']).dt.to_period('M').value_counts().sort_index()
	by_month.index = pd.PeriodIndex(by_month.index)
	df_month = by_month.rename_axis('month').reset_index(name='counts')
	
	# Figuur toevoegen 
	fig4 = go.Figure()
	fig4.add_traces(data=go.Scatter(x=df_month['month'].astype(dtype=str), 
	                        y=df_month['counts'].cumsum(),
	                        marker_color='indianred', text="counts"))
	
	fig4.layout.xaxis = { 
	  "range": ["2015-01-01",'2022-11-01']
	 }
	
	fig4.update_yaxes(title_text = "Aantal elektrische auto's")
	fig4.update_xaxes(title_text = 'Maanden')
	fig4.update_layout(title_text = "Totaal aantal elektrische auto's per maand")
	
	st.plotly_chart(fig4, use_container_width = True)
	st.write("In dit figuur is de cumulatieve aantal elektrische auto's weergeven per maand. De lijn begint erg op een kwadratische formule te lijken. Dat houdt in dat het aantal elektrische auto's in de laatste paar jaren steeds vaker wordt gekocht en dat ook zou blijven doen.")
	st.write('Het aantal elektrische auto\'s waarvan de eigenaar de eerste eigenaar is: ' + str(elektrisch.loc[elektrisch['Eerste eigenaar'] == 0, 'Eerste eigenaar'].count()))
	st.write('Het aantal tweede-hands elektrische auto\'s: ' + str(elektrisch.loc[elektrisch['Eerste eigenaar'] == 1, 'Eerste eigenaar'].count()))
	st.write('Het aantal hybride auto\'s: ' + str(elektrisch.loc[elektrisch['Hybride'] != 0, 'Hybride'].count()))

########################################## Catalogusprijs door de jaren heen ###############################################

with c2:
	# Plotten van catalogusprijs door de jaren heen
	fig5 = px.line(
	    elektrisch.groupby(elektrisch['Datum eerste toelating'].dt.year).mean().reset_index(),
	    x="Datum eerste toelating",
	    y="Catalogusprijs",
	    title = 'Gemiddelde catalogusprijs door de jaren heen'
	)
	
	# Updaten layout
	fig5.layout.xaxis = {"range": ["2015",'2022']}
	fig5.update_xaxes(title_text = 'Jaar')
	
	st.plotly_chart(fig5, use_container_width = True)
	st.write("In dit figuur is de catalogusprijs van de elektrische auto's door de jaren heen te zien. Hierin is af te lezen dat er een piek van boven de 70k euro in 2018 zat, echter dat hij daarna sterk afnam naar onder de 50k. Echter daarna nam het weer langzaam toe.")

st.header('Conclusie')
st.write("In deze blog heeft u kunnen lezen over de elektrische auto's en hun laadstations. Hopelijk heeft u hier wat van opgestoken!")

# In[ ]:





