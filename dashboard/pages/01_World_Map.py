import pandas as pd 
import streamlit as st
import plotly.express as px
from utils import ImportRiskCSVScores

st.set_page_config(
    layout='wide'
)

#should probably cache this since i might need this in 02 country deepdive
if 'RiskDataframe' not in st.session_state:
    st.session_state['RiskDataframe'] = ImportRiskCSVScores()

#RiskDataframe now exists in the session state
st.session_state['RiskDataframe'] = st.session_state['RiskDataframe'].sort_values(by='date',
                                                                                    axis=0,
                                                                                    ascending=False)

DateCounts = st.session_state['RiskDataframe']['date'].value_counts() 
MinDate = min(DateCounts.index)
MaxDate = max(DateCounts.index)

Year = st.slider(label='Select date to view risk score for',
                min_value=MinDate,
                max_value=MaxDate,
                value = 2020)

if 'RiskDataframeByDate' not in st.session_state:
    st.session_state['RiskDataframeByDate'] = [group for _,group in st.session_state['RiskDataframe'].groupby('date')]

if 'LocationOfYears' not in st.session_state:
    IndexLocationOfYears = []
    for i in st.session_state['RiskDataframeByDate']:
        IndexLocationOfYears.append(i['date'].iloc[0])

    st.session_state['LocationOfYears'] = IndexLocationOfYears

IndexOfYear = st.session_state['LocationOfYears'].index(Year)

ChoroplethMapDataframe = st.session_state['RiskDataframeByDate'][IndexOfYear]

fig = px.choropleth(
    ChoroplethMapDataframe,
    locations=ChoroplethMapDataframe['countryiso3code'],
    locationmode='ISO-3',
    scope='world',
    title=f'Risk Score Around the World during {Year}',
    hover_name=ChoroplethMapDataframe['country'],
    hover_data={'RiskScore':':.2f','countryiso3code':False},
    color='RiskScore',
    color_continuous_scale='RdYlGn_r',
    range_color=[0,1]   
)

fig.update_geos(visible=True,
    projection_scale=1,  
    center=dict(lat=0, lon=0),  
    lataxis_range=[-60, 90],
    lonaxis_range=[-180, 180],
    showcountries = True,
    countrycolor='black',
    countrywidth=1

)

fig.update_layout(
    title_font_family = 'Sans-Serif',
    coloraxis_colorbar_title = 'Risk Score',
    coloraxis_colorbar_title_font_family = 'Sans-Serif',
    dragmode=False,
    margin={'r':0,'t':40,'l':0,'b':0},
)

fig.update_layout(
    title = {
        'x' : 0.5,
        'y' : 0.95,
        'xanchor' : 'center',
        'yanchor' : 'top'
    },
)


st.info(body='Countries covered in the color :green[green] are at low risk of facing a spike in economic inequality in the future, countries covered in the color :red[red] are at a high risk of facing a spike in the future')
st.info(body='Countries that are covered in the color white have insufficient data and thus no prediction can be made on them')
st.info(body='A spike means Gini rising by more than 1.8 points within three years.')
st.info(body='Scores for countries without a Gini Score are estimates built from other indicators and should be viewed with caution')
st.plotly_chart(fig)
#choropleth map is interactive and changes dynamically based on the date, double check if risk scores are accurate
#dont forget to add borders and make it so that zooming is possible without breaking the map
#need to add an explanation on why there are so many risk scores for earlier years (i.e 1990 having 100+ entries)

st.info(body='Please note that scores for countries without a Gini Score are estimates built with other indicators and should be viewed with caution')
First15Scores = ChoroplethMapDataframe.sort_values(by='RiskScore',ascending=False) 
First15Scores = First15Scores.drop(columns = (list(First15Scores.columns))[0]) 
First15Scores = First15Scores.iloc[:15]

First15Scores = First15Scores.reset_index(drop = True) #to prevent the reset indexes from being added as a column
 
First15Scores.index = First15Scores.index + 1 #to make the indexes start from 1

st.dataframe(First15Scores)