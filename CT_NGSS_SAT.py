import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
def main():
    df = pd.read_csv('ct_info/ct_sat_ngss.csv')
    needed = ['Math','ELA','NGSS','Year','DRG_num']
    df = df.dropna(subset=needed)

    df_19 = df[df['Year'] == 2019]
    df_22 = df[df['Year'] == 2022]
    df_23 = df[df['Year'] == 2023]
    df_24 = df[df['Year'] == 2024]
    set_19, set_22, set_23, set_24 = set(df_19['District'].values),set(df_22['District'].values),set(df_23['District'].values),set(df_24['District'].values)
    
    common_set = set_19&set_22& set_23& set_24
    set_list = [set_19,set_22,set_23,set_24]
    districts = list(common_set)
    districts.sort()
    chosen_district = st.selectbox("Highlight a District",districts,index = None)
    colors_list, sizes_list, alpha_list = [],[],[]
    for sets in set_list:
        colors = []
        sizes = []
        alpha = []
        for district in sets:
            if district == chosen_district:
                colors.append((0.0,0.9,0.0,1.0))
                sizes.append(64)
                alpha.append(1.0)
            else:
                colors.append((0.0, 0.2, 0.8, 0.5))
                sizes.append(16)
                alpha.append(0.5)
        colors_list.append(colors)
        sizes_list.append(sizes)
        alpha_list.append(alpha)
    
    X_vals = df['Math'].values
    Y_vals = df['ELA'].values

    
    

    math_df = pd.read_csv('ct_info/ct_math_spline.csv')
    math_effect = math_df['Math_Effect'].values
    math_confi_lower = math_df['Math Confidence Lower'].values
    math_confi_upper = math_df['Math Confidence Upper'].values

    read_df = pd.read_csv('ct_info/ct_reading_spline.csv')
    reading_effect = read_df['Reading_Effect'].values
    read_confi_lower = read_df['Reading Confidence Lower'].values
    read_confi_upper = read_df['Reading Confidence Upper'].values

    year_df = pd.read_csv('ct_info/ct_year_spline.csv')
    year_effect = year_df['Year_Effect'].values
    year_confi_lower = year_df['Year Confidence Lower'].values
    year_confi_upper = year_df['Year Confidence Upper'].values

    drg_df = pd.read_csv('ct_info/ct_drg_spline.csv')
    drg_effect = drg_df['DRG_Effect'].values
    drg_confi_lower = drg_df['DRG Confidence Lower'].values
    drg_confi_upper = drg_df['DRG Confidence Upper'].values


    spline_stat_df = pd.read_csv('ct_info/ct_spline_stats.csv')
    coeff = spline_stat_df['coeff'][1]

    
    fig = plt.figure(figsize=(8,8), layout = 'constrained')
    ax = plt.axes(projection='3d')
    plt.style.use("dark_background")
    ax.set_xlabel('SAT Math Score')
    ax.set_ylabel('SAT Reading Score')
    ax.set_zlabel('NGSS Passing Percent')
    

    gam_plot_X = math_df['SAT_Math'].values
    gam_plot_Y = read_df['SAT_Reading'].values
    ax.plot(gam_plot_X,gam_plot_Y,math_effect + reading_effect + year_effect + drg_effect + coeff,linewidth=3,c=(0.0,0.9,0.0,0.8))

    # Lower Confidence reading
    ax.plot(gam_plot_X,gam_plot_Y, math_confi_lower + read_confi_lower + year_confi_lower + drg_confi_lower + coeff, 
            c = (1.0,0.1,0.3,1.0), ls = '--',linewidth=2)

    # upper confidence reading
    ax.plot(gam_plot_X,gam_plot_Y, math_confi_upper + read_confi_upper + year_confi_upper + drg_confi_upper + coeff, 
            c = (1.0,0.1,0.3,1.0), ls = '--',linewidth=2)


    col1, col2 = st.columns(2)
    with col1:
        elevation = st.slider("Elevation", min_value=0, max_value=90, value = 0)
    with col2:
        azimuth = st.slider("Rotation", min_value=-90, max_value=15, value = 0)

    col1, col2 = st.columns([1,8])
    with col1:
        year_19 = st.checkbox("Show 2019 Data", value=False)
        if year_19:
            ax.scatter(df_19['Math'].values, 
                       df_19['ELA'].values,
                       df_19['NGSS'].values,
                       c = colors_list[0], s = sizes_list[0], alpha = alpha_list[0] )
            
        year_22 = st.checkbox("Show 2022 Data", value=False)
        if year_22:
            ax.scatter(df_22['Math'].values, 
                       df_22['ELA'].values,
                       df_22['NGSS'].values,
                       c = colors_list[1],s = sizes_list[1], alpha = alpha_list[1])
        
        year_23 = st.checkbox("Show 2023 Data", value=False)
        if year_23:
            ax.scatter(df_23['Math'].values, 
                       df_23['ELA'].values,
                       df_23['NGSS'].values ,
                       c = colors_list[2], s = sizes_list[2], alpha = alpha_list[2] )

        year_24 = st.checkbox("Show 2024 Data", value=False)
        if year_24:
            ax.scatter(df_24['Math'].values, 
                       df_24['ELA'].values,
                       df_24['NGSS'].values ,
                       c = colors_list[3], s = sizes_list[3], alpha = alpha_list[3])
    ax.view_init(elev = elevation, azim = azimuth, roll = 0)
    with col2:
        st.pyplot(fig)

    
    
    #plt.tight_layout()
    


if __name__ == "__main__":
   main()