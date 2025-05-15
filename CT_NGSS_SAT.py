import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
def main():
    df = pd.read_csv('ct_sat_ngss.csv')
    df_19 = df[df['Year'] == 2019]
    df_22 = df[df['Year'] == 2022]
    df_23 = df[df['Year'] == 2023]
    df_24 = df[df['Year'] == 2024]

    districts = list(df_19['District'].values)
    colors = []
    sizes = []
    for district in districts:
       if district == 'East Hampton School District':
          colors.append((0.0,0.9,0.0,1.0))
          sizes.append(32)
       else:
          colors.append((0.0, 0.2, 0.8, 1.0))
          sizes.append(16)
    
    X_vals = df['Math'].values
    Y_vals = df['ELA'].values
    train = False
    if train:
        from pygam import GAM, s, te, f
        train = df[df['Year']> 2018]
        test = df[df['Year'] < 2026]

        X_train = train[['Math','ELA','Year']].values
        y_train = train['NGSS'].values

        X_test = test[['Math','ELA','Year']].values
        y_test = test['NGSS'].values

        dist = 'normal'
        link = 'identity'
        terms = (
            s(0, constraints='monotonic_inc') 
            + s(1, constraints='monotonic_inc') 
            + s(2)
            )
        gam = GAM(
            terms = terms ,
            distribution=dist,
            link = link
        ).fit(X_train, y_train)
        r2 = gam.statistics_['pseudo_r2']['explained_deviance']
        print(r2)

        save = True
        if save:
            math_grid = gam.generate_X_grid(term=0,n=len(X_train[:,0]))
            math_effect, math_confi = gam.partial_dependence(term=0,X=math_grid, width = 0.99)

            reading_grid = gam.generate_X_grid(term=1,n=len(X_train[:,1]))
            reading_effect, reading_confi = gam.partial_dependence(term=1,X=reading_grid, width = 0.99)

            year_grid = gam.generate_X_grid(term=2,n=len(X_train[:,2]))
            year_effect, year_confi = gam.partial_dependence(term=2,X=year_grid, width = 0.99)

            



            pd.DataFrame({
                'SAT_Math': math_grid[:,0],
                'Math_Effect': math_effect,
                'Math Confidence Lower': math_confi[:,0],
                'Math Confidence Upper': math_confi[:,1]
            }).to_csv('ct_math_spline.csv',index=False)

            pd.DataFrame({
                'SAT_Reading': reading_grid[:,1],
                'Reading_Effect': reading_effect,
                'Reading Confidence Lower': reading_confi[:,0],
                'Reading Confidence Upper': reading_confi[:,1]
            }).to_csv('ct_reading_spline.csv',index=False)

            pd.DataFrame({
                'Year': year_grid[:,2],
                'Year_Effect': year_effect,
                'Year Confidence Lower': year_confi[:,0],
                'Year Confidence Upper': year_confi[:,1]
            }).to_csv('ct_year_spline.csv',index=False)

    
    

    math_df = pd.read_csv('ct_math_spline.csv')
    math_effect = math_df['Math_Effect'].values
    math_confi_lower = math_df['Math Confidence Lower'].values
    math_confi_upper = math_df['Math Confidence Upper'].values

    read_df = pd.read_csv('ct_reading_spline.csv')
    reading_effect = read_df['Reading_Effect'].values
    read_confi_lower = read_df['Reading Confidence Lower'].values
    read_confi_upper = read_df['Reading Confidence Upper'].values

    year_df = pd.read_csv('ct_year_spline.csv')
    year_effect = year_df['Year_Effect'].values
    year_confi_lower = year_df['Year Confidence Lower'].values
    year_confi_upper = year_df['Year Confidence Upper'].values

    #coeff = gam.coef_[-1]
    coeff = 44
    
    fig = plt.figure(figsize=(8,8), layout = 'constrained')
    ax = plt.axes(projection='3d')
    ax.set_xlabel('SAT Math Score')
    ax.set_ylabel('SAT Reading Score')
    ax.set_zlabel('NGSS Score')
    plt.style.use("dark_background")

    gam_plot_X = math_df['SAT_Math'].values
    gam_plot_Y = read_df['SAT_Reading'].values
    ax.plot(gam_plot_X,gam_plot_Y,math_effect + reading_effect + year_effect + coeff,linewidth=4,c=(0.0,0.9,0.0,0.8))

    # Lower Confidence reading
    ax.plot(gam_plot_X,gam_plot_Y, math_confi_lower + read_confi_lower + year_confi_lower + coeff, c = (1.0,0.1,0.3,1.0), ls = '--',linewidth=3)

    # upper confidence reading
    ax.plot(gam_plot_X,gam_plot_Y, math_confi_upper + read_confi_upper + year_confi_upper + coeff, c = (1.0,0.1,0.3,1.0), ls = '--',linewidth=3)


    col1, col2 = st.columns(2)
    with col1:
        elevation = st.slider("Elevation", min_value=0, max_value=90, value = 0)
    with col2:
        azimuth = st.slider("Rotation", min_value=-90, max_value=15, value = 0)
    side_roll = 0
    col1, col2 = st.columns([1,8])
    with col1:
        year_19 = st.checkbox("Show 2019 Data", value=False)
        if year_19:
            ax.scatter(df_19['Math'].values, 
                       df_19['ELA'].values,
                       df_19['NGSS'].values,
                       c=colors,s= sizes )
            
        year_22 = st.checkbox("Show 2022 Data", value=False)
        if year_22:
            ax.scatter(df_22['Math'].values, 
                       df_22['ELA'].values,
                       df_22['NGSS'].values,
                       c=colors,s= sizes)
        
        year_23 = st.checkbox("Show 2023 Data", value=False)
        if year_23:
            ax.scatter(df_23['Math'].values, 
                       df_23['ELA'].values,
                       df_23['NGSS'].values ,
                       c=colors,s= sizes)

        year_24 = st.checkbox("Show 2024 Data", value=False)
        if year_24:
            ax.scatter(df_24['Math'].values, 
                       df_24['ELA'].values,
                       df_24['NGSS'].values ,
                       c=colors,s= sizes)
    ax.view_init(elev = elevation, azim = azimuth, roll = side_roll)
    with col2:
        st.pyplot(fig)

    
    
    #plt.tight_layout()
    


if __name__ == "__main__":
   main()