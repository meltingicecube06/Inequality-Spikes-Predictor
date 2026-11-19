import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).resolve().parent
assets_dir = current_dir / 'assets'

if assets_dir.is_dir() == False:
    raise FileNotFoundError('Assets folder not found')


@st.cache_data
def ImportMasterDataframe():
    MasterDataframe_path = assets_dir / 'CleanedMasterDataFrame.csv'
    try:
        MasterDataframe = pd.read_csv(MasterDataframe_path) #do not forget to move this to assets 
        MasterDataframe = MasterDataframe.drop(columns = (list(MasterDataframe.columns))[0])
        return MasterDataframe
    except FileNotFoundError:
        st.error('File not found!')
#imported the RiskCSV and Cleaned MasterDataFrame

@st.cache_data
def ImportPredictShapValues():
    PredictShapValues_path = assets_dir / 'PredictShapValues.npy'
    try:
        with open(PredictShapValues_path,'rb') as f:
            PredictShapValues = np.load(f)
        return PredictShapValues
    except FileNotFoundError:
        st.error('File not found!')

@st.cache_data
def ImportEmptyGiniSpikeDataframe():
    EmptyGiniSpikeDataframe_path = assets_dir / 'EmptyGiniSpikeValuesDataframe.csv'
    try:
        EmptyGiniSpikeDataframe = pd.read_csv(EmptyGiniSpikeDataframe_path)
        EmptyGiniSpikeDataframe = EmptyGiniSpikeDataframe.drop(columns= (list(EmptyGiniSpikeDataframe.columns))[0])
        return EmptyGiniSpikeDataframe
    except FileNotFoundError:
        st.error('File not found!')

@st.cache_data
def ImportRiskCSVScores():
    RiskCSVScores_path =  assets_dir / 'RiskCSVScores.csv'
    try: 
        RiskCSVScores = pd.read_csv(RiskCSVScores_path)
        return RiskCSVScores
    except FileNotFoundError:
        st.error('File not found!')

@st.cache_data
def ImportPredictShapData():
    PredictShapData_path = assets_dir / 'PredictShapValuesData.csv'
    try:
        PredictShapData = pd.read_csv(PredictShapData_path)
        PredictShapData = PredictShapData.drop(columns= (list(PredictShapData.columns))[0])
        return PredictShapData
    except FileNotFoundError:
        st.error('File not found!')
    
@st.cache_data
def ImportPredictShapBaseValues():
    PredictShapBaseValues_path = assets_dir / 'PredictShapValuesBaseValues.csv'
    try:
        PredictShapBaseValues = pd.read_csv(PredictShapBaseValues_path)
        PredictShapBaseValues = PredictShapBaseValues.drop(columns= (list(PredictShapBaseValues.columns))[0])
        return PredictShapBaseValues
    except FileNotFoundError:
        st.error('File not found!')

@st.cache_data
def ImportPredictShapValuesFeatureNames():
    PredictShapValuesFeatureNames_path = assets_dir / 'PredictShapValuesFeatureNames.csv'
    try:
        ShapValuesFeatureNames = pd.read_csv(PredictShapValuesFeatureNames_path)
        ShapValuesFeatureNames = ShapValuesFeatureNames.drop(columns= (list(ShapValuesFeatureNames.columns))[0])
        FeatureNames = ShapValuesFeatureNames[(list(ShapValuesFeatureNames.columns)[0])]
        FeatureNames = FeatureNames.to_list()
        return FeatureNames
    except FileNotFoundError:
        st.error('File not found!')

@st.cache_data
def ImportIndicatorTitles():
    try:
        from assets import IndicatorTitles
        IndicatorTitles_dict = IndicatorTitles.variable_titles
        return IndicatorTitles_dict
        
    except FileNotFoundError:
        st.error('File not found!')

def ExtractTopFeatureNames(ShapExplanationObject,max_display):
# Assuming `ShapExplanationObject` is your 1D shap.ShapExplanationObject object (e.g., shap_values[0])
    num_features = len(ShapExplanationObject.values)

    # Get indices sorted by absolute SHAP value (descending)
    order = np.argsort(-np.abs(ShapExplanationObject.values))

    # If features exceed max_display, the last slot is reserved for "other features"
    if num_features > max_display:
        top_indices = order[: max_display - 1]
    else:
        top_indices = order

    # Extract the exact feature names shown on the plot
    top_feature_names = [ShapExplanationObject.feature_names[i] for i in top_indices]
    return(top_feature_names)
