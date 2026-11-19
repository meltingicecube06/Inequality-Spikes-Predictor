import pandas as pd 
import streamlit as st
from dashboard import utils
import shap
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from copy import copy

st.title('Country-wise Deepdive')
st.divider()

if 'PredictShapValues' not in st.session_state:
    TemporaryVariable = utils.ImportPredictShapValues()
    st.session_state['PredictShapValues'] = pd.DataFrame(TemporaryVariable)
#imported the predict shap values and put them inside of session state

if 'EmptyGiniSpikeDataframe' not in st.session_state:
    st.session_state['EmptyGiniSpikeDataframe'] = utils.ImportEmptyGiniSpikeDataframe()
#imported the empty Gini spike dataframe

if 'RiskCSVScores' not in st.session_state:
    st.session_state['RiskCSVScores'] = utils.ImportRiskCSVScores()

if 'CountriesList' not in st.session_state:
    CountriesAndNumberOfEntries = st.session_state['EmptyGiniSpikeDataframe']['country'].unique()
    CountriesAndNumberOfEntries = np.sort(CountriesAndNumberOfEntries)
    st.session_state['CountriesList'] = list(CountriesAndNumberOfEntries)

if 'PredictShapData' not in st.session_state:
    st.session_state['PredictShapData'] = utils.ImportPredictShapData()

if 'PredictShapBaseValues' not in st.session_state:
    st.session_state['PredictShapBaseValues'] = utils.ImportPredictShapBaseValues()

if 'PredictShapFeatureNames' not in st.session_state:
    st.session_state['PredictShapFeatureNames'] = utils.ImportPredictShapValuesFeatureNames()

if 'IndicatorTitles' not in st.session_state:
    st.session_state['IndicatorTitles'] = utils.ImportIndicatorTitles()


UserInputCountry = st.selectbox(label='Pick a country to get more information about:',
                        options=st.session_state['CountriesList'])

CountryDataframe = st.session_state['RiskCSVScores'][st.session_state['RiskCSVScores']['country'] == UserInputCountry]
MostRecentDateOfRiskEntryIndex = CountryDataframe['date'].idxmax()
MostRecentDateOfRiskEntry = CountryDataframe['date'].max()

#i need to get the most recent entry of the year and display that using st.metric

MostRecentRiskScore = st.session_state['RiskCSVScores']['RiskScore'].iloc[MostRecentDateOfRiskEntryIndex]
CountryInfo = st.session_state['EmptyGiniSpikeDataframe'].iloc[MostRecentDateOfRiskEntryIndex]
CountryShapValues = st.session_state['PredictShapValues'].iloc[MostRecentDateOfRiskEntryIndex]
CountryShapData = st.session_state['PredictShapData'].iloc[MostRecentDateOfRiskEntryIndex]
CountryShapBaseValues = float((st.session_state['PredictShapBaseValues'].iloc[MostRecentDateOfRiskEntryIndex]).iloc[0])

MostRecentRiskScoreRounded = round(MostRecentRiskScore * 100)

if MostRecentRiskScoreRounded >= 40:
    MetricValue = f":yellow[**{MostRecentRiskScoreRounded}%**]"
elif MostRecentRiskScoreRounded >= 60:
    MetricValue = f":red[**{MostRecentRiskScoreRounded}%**]"
elif MostRecentRiskScoreRounded <= 40: 
    MetricValue = f":green[**{MostRecentRiskScoreRounded}%**]"


st.metric(label=f'Most recent risk score as of {int(MostRecentDateOfRiskEntry)}',
                value=f'{MetricValue}',
                )

st.info(body='The risk score is XGBoosts predicted probability and not calibrated probability')
#** makes it so that the values are in bold
    
st.divider()

CountryShapValue = shap.Explanation(base_values=CountryShapBaseValues,
                                    data=CountryShapData,
                                    values=CountryShapValues,
                                    feature_names=st.session_state['PredictShapFeatureNames'])

st.subheader("SHAP Waterfall Plot")
st.info("The chart below shows the economic indicators that drive this country's risk score, pink bars increase the risk while blue bars decrease it.",
        icon='ℹ️')

fig, ax = plt.subplots(tight_layout=True)
shap.plots.waterfall(CountryShapValue, max_display=15, show=False)
st.pyplot(fig, width='stretch')
plt.close(fig)

TopCountryShapFeatureNames = utils.ExtractTopFeatureNames(CountryShapValue,15)
TitlesInShapFigure = {}

for i in TopCountryShapFeatureNames:
    if i in st.session_state['IndicatorTitles'].keys():
        TitlesInShapFigure[i] = st.session_state['IndicatorTitles'][i]

st.table(data = TitlesInShapFigure,
         border = 'horizontal',
         width = 'content')  

###From this point all of this is for the Country Py plot
if 'MasterDataframe' not in st.session_state:
    st.session_state['MasterDataframe'] = utils.ImportMasterDataframe()

CountryIso3code = CountryInfo['countryiso3code'] #got the country iso3codeinfo
#need to locate the country inside of masterdataframe and get its gini values
MasterDataframeCountry = st.session_state['MasterDataframe'][st.session_state['MasterDataframe']['countryiso3code']==CountryIso3code]
#located the country 
#extract gini values while looking at the IsOriginalFlag

MasterDataframeCountry = MasterDataframeCountry.dropna(subset = ['SI.POV.GINI'])

CountryLineChart = px.line(
    data_frame=MasterDataframeCountry,
    x='date',
    y='SI.POV.GINI',
    title='Historical Wealth Inequality data from 1990-2020',
    markers=True,
)

marker_colors = ['blue' if val == True else 'red' for val in MasterDataframeCountry['IsOriginalValue']]

CountryLineChart.update_traces(
    mode='lines+markers',
    marker = dict(
        color = marker_colors
    ),
    line = dict(
        color = 'darkblue',
        width=6
    )
)

CountryLineChart.update_xaxes(
    title='Year',
    range=[1990,2020]
)

CountryLineChart.update_yaxes(
    title='Gini Index',
    range = [0.0,100.0],
    dtick = 10
)



st.plotly_chart(CountryLineChart)
st.markdown(':blue[Blue] points are real, :red[Red] points are interpolated')