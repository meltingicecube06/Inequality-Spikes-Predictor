import pandas as pd
import streamlit as st
import numpy as np
import utils
from pathlib import Path

assets_dir = Path(__file__).resolve().parent / 'assets'

IntroductoryParagraph1 = '''The Gini index is a World Bank measure of economic inequality within a country, on a scale from 0 (perfect equality) to 100 (one person holds everything). It is calculated from the area between the line of perfect equality and the Lorenz curve, divided by the total area under the equality line (see below for reference). This project combines the Gini index with 14 other World Bank indicators to estimate the risk that a country’s Gini will spike over the next three years. '''

IntroductoryParagraph2 = '''Predicting spikes in inequality is an important task as it allows governments and
economic bodies to remedy the issue before it causes irreperable damage. Spikes in inequality can lead to issues such
as social unrest, economic downfall and other related issues.'''

IntroductoryParagraph3 = '''Rising inequality is linked to social and economic strain, so early warning signs could help analysts decide where to look more closely. This is an exploratory project, not a tool for policy decisions.
'''

MethodologyParagraph = '''Data come from the World Bank (via the wbdata library): 15 indicators for 158 countries from 1990 to 2020. Gini is recorded for only about 62% of country-years, and about a third of those values are interpolated across gaps of up to three years. Features include current values plus one- to three-year lags and one-year changes for five indicators. The model was trained on roughly 1990 to 2011 and tested on 2012 to 2020 so it is never judged on years it learned from. XGBoost was compared with a logistic regression baseline, and SHAP is used to explain individual predictions.'''

KeyFindingsParagraph = '''A rise in Gini over the previous year is associated with higher predicted risk, while small to moderate declines are associated with lower risk. Very large declines show a weaker effect. \n
On average, Latin America & Caribbean (0.55) and Sub-Saharan Africa (0.52) received the highest scores in 2012 to 2020. Observed spike rates support this only for Latin America & Caribbean (32%), while Sub-Saharan Africa (24%) was near average.\n
The one-year change in Gini and a country’s region were the most influential features (average absolute SHAP value 0.26 each), followed by household consumption per capita (0.17), GDP per capita (0.16), inflation three years earlier (0.13) and trade as a share of GDP (0.12). The other 27 features together contribute about 0.58, so predictions draw on many indicators and no single one dominates  '''

st.set_page_config(
    layout='wide',
    page_title='Inequality Predictor',
    initial_sidebar_state="locked" 
    )

st.title('Inequality Spikes Predictor Using Indicators from the World Bank')

st.header('Introduction',
          divider='grey')

st.write(IntroductoryParagraph1)

st.image(str(assets_dir / 'GiniReferenceImage.png'),
         caption='Gini Coefficient Calculation by researchgate (via WikiMedia Commons) / CC BY-SA 4.0',
         link='https://www.researchgate.net/figure/Lorenz-curve-The-Gini-index-is-calculated-as-the-ratio-of-the-area-between-the-perfect_fig1_341916065')

st.write(IntroductoryParagraph2)

st.header('Methodology',
          divider='grey')

st.write(MethodologyParagraph)

st.header('Key findings',
          divider='grey')

st.write(KeyFindingsParagraph)

with st.expander("Limitations",expanded=False):
    st.write("- Gini is missing for 38% of country-years, and a third of the rest is interpolated.\n" \
    "- Scores for years with no recorded Gini are estimates based on other indicators.\n" \
    "- Scores are not calibrated probabilities.\n" \
    "- SHAP shows how the model uses features, not what causes inequality.")