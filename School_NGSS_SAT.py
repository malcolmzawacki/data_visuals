import streamlit as st

def test_main_2():
    st.title("School NGSS Scores vs SAT Math and Reading Scores")
    import pandas as pd
    import numpy as np

    df = pd.read_csv('ngss_data.csv')
    train = df[df['Year']> 2021]
    df_22 = df[df['Year']== 2022]
    df_23 = df[df['Year']== 2023]
    df_24 = df[df['Year']== 2024]
    df_25 = df[df['Year']== 2025]


    X_train = train[['SAT_Math','SAT_Reading']].values
    y_train = train['NGSS'].values

    math_df = pd.read_csv('math_spline.csv')
    math_effect = math_df['Math_Effect'].values
    math_confi_lower = math_df['Math Confidence Lower'].values
    math_confi_upper = math_df['Math Confidence Upper'].values

    read_df = pd.read_csv('reading_spline.csv')
    reading_effect = read_df['Reading_Effect'].values
    read_confi_lower = read_df['Reading Confidence Lower'].values
    read_confi_upper = read_df['Reading Confidence Upper'].values

    year_df = pd.read_csv('year_spline.csv')
    year_effect = year_df['Year_Effect'].values
    year_confi_lower = year_df['Year Confidence Lower'].values
    year_confi_upper = year_df['Year Confidence Upper'].values

    tensor_df = pd.read_csv('tensor_effect.csv')
    tensor_effect = tensor_df['Tensor Effect'].values

    spline_stat_df = pd.read_csv('spline_stats.csv')
    coeff = spline_stat_df['coeff'][1]

    binom_df = pd.read_csv('ngss_binom_plot.csv')
    binom_pts = binom_df['ngss_pred'].values
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,8), layout = 'constrained')
    ax = plt.axes(projection='3d')

    z = y_train
    x = X_train[:,0]
    y = X_train[:,1]
    c = x+y

    level_1 = 1072
    level_2 = 1098
    level_3 = 1140
    X_vals = math_df['SAT_Math'].values
    Y_vals = read_df['SAT_Reading'].values
    zero_ish = np.arange(0,0.1,len(X_vals))
    



    tensor_effect = 0
    # Linear GAM trendline
    #ax.plot(df['SAT_Math'].values,df['SAT_Reading'].values, binom_pts,linewidth=4,c=(0.0,0.9,0.0,0.8))
    ax.plot(X_vals,Y_vals,math_effect + reading_effect + year_effect + tensor_effect + coeff,linewidth=4,c=(0.0,0.9,0.0,0.8))

    # Lower Confidence reading
    ax.plot(X_vals,Y_vals, math_confi_lower + read_confi_lower + year_confi_lower + coeff, c = (1.0,0.1,0.3,1.0), ls = '--',linewidth=3)

    # upper confidence reading
    ax.plot(X_vals,Y_vals, math_confi_upper + read_confi_upper + year_confi_upper + coeff, c = (1.0,0.1,0.3,1.0), ls = '--',linewidth=3)

    # data
    #ax.scatter(x,y,z,c=df['Year']//2,s=(df['Year']-2020)**(5/2))
    ax.set_xlabel('SAT Math Score')
    ax.set_ylabel('SAT Reading Score')
    ax.set_zlabel('NGSS Score')
    plt.style.use("dark_background")

    col1, col2, col3 = st.columns([5,5,2])
    with col1:
        elevation = st.slider("Elevation", min_value=0, max_value=90, value = 0)
    with col2:
        azimuth = st.slider("Rotation", min_value=-90, max_value=15, value = 0)
    with col3:
        surface = st.checkbox("Show Levels", value=False)
        if surface:
            x_surf = np.arange(200,900,100)
            y_surf = np.arange(200,900,100)
            X_surf,Y_surf = np.meshgrid(x_surf, y_surf)
            
            Z_surf1 = (X_surf*level_1/X_surf + Y_surf*level_1/Y_surf) / 2
            ax.plot_surface(X_surf,Y_surf, Z_surf1,color=(1.0, 0.5, 0.0, 0.4))
            
            Z_surf2 = (X_surf*level_2/X_surf + Y_surf*level_2/Y_surf) / 2
            ax.plot_surface(X_surf,Y_surf, Z_surf2,color=(0.0, 1.0, 0.2, 0.3))

            Z_surf3 = (X_surf*level_3/X_surf + Y_surf*level_3/Y_surf) / 2
            ax.plot_surface(X_surf,Y_surf, Z_surf3,color=(0.0, 0.2, 0.8, 0.2))
    side_roll = 0
    col1, col2 = st.columns([1,8])
    with col1:
        year_22 = st.checkbox("Show 2022 Data", value=False)
        if year_22:
            ax.scatter(df_22['SAT_Math'].values, 
                       df_22['SAT_Reading'].values,
                       df_22['NGSS'].values,
                       color=(0.5,0.0,0.5,1.0),s=16)
 
        year_23 = st.checkbox("Show 2023 Data", value=False)
        if year_23:
            ax.scatter(df_23['SAT_Math'].values, 
                       df_23['SAT_Reading'].values,
                       df_23['NGSS'].values,
                       color=(0.0,1.0,1.0,1.0),s=16)
        
        year_24 = st.checkbox("Show 2024 Data", value=False)
        if year_24:
            ax.scatter(df_24['SAT_Math'].values, 
                       df_24['SAT_Reading'].values,
                       df_24['NGSS'].values,
                       color=(1.0,1.0,0.0,1.0),s=16)

        year_25 = st.checkbox("Show 2025 Data", value=False)
        if year_25:
            ax.scatter(df_25['SAT_Math'].values, 
                       df_25['SAT_Reading'].values,
                       df_25['NGSS'].values,
                       color=(0.8,1.0,0.8,1.0),s=16)

    ax.view_init(elev = elevation, azim = azimuth, roll = side_roll)
    with col2:
    #plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    test_main_2()