import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
def main():
    df = pd.read_csv('ct_info/ct_sat_ngss.csv')
    needed = ['Math','ELA','NGSS','Year','DRG_num']
    df = df.dropna(subset=needed)
    years = list(set(df['Year']))
    years.sort()
    
    drgs = list(set(df['DRG']))
    temp_drgs = []
    temp_drgs_long = []
    for i in range(len(drgs)):
        drgs[i] = str(drgs[i])
        if len(drgs[i]) < 2:
            temp_drgs.append(drgs[i])
        else:
            temp_drgs_long.append(drgs[i])
    temp_drgs.sort()
    for i in range(len(temp_drgs_long)):
        temp_drgs.append(temp_drgs_long[i])
    drgs = temp_drgs

    col1,col2 = st.columns(2)
    with col1:
        year_choice = st.selectbox("View a Year",years)
    with col2:
        drg_choice = st.selectbox("Choose a DRG", drgs)

    df_choice = df[df['Year'] == year_choice]
    df_choice2 = df_choice[df_choice['DRG'] == drg_choice].sort_values(by='NGSS',ascending = True)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10,4), layout = 'constrained')
    ax = plt.axes()
    plt.xticks(rotation=45, ha = 'right')
    ax.bar(df_choice2['District'].values, df_choice2['NGSS'].values - min(df['NGSS'].values), bottom = min(df['NGSS'].values))

    st.pyplot(fig)


if __name__ == "__main__":
   main()