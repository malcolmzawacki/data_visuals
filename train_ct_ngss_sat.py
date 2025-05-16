import pandas as pd
from pygam import GAM, s, te, f



df = pd.read_csv('ct_info/ct_sat_ngss.csv')
needed = ['Math','ELA','NGSS','Year','DRG_num']
df = df.dropna(subset=needed)

train = df[df['Year']> 2018]
test = df[df['Year'] < 2026]

X_train = train[['Math','ELA','Year','DRG_num']].values
y_train = train['NGSS'].values

X_test = test[['Math','ELA','Year']].values
y_test = test['NGSS'].values

dist = 'normal'
link = 'identity'
default_spline = [20,20,20,20]
train = False
if train:
    pass
else:
    spline_score = default_spline
    terms = (
        s(0, constraints='monotonic_inc') 
        + s(1, constraints='monotonic_inc') 
        + s(2)
        +s(3)
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


    drg_grid = gam.generate_X_grid(term=3,n=len(X_train[:,3]))
    drg_effect, drg_confi = gam.partial_dependence(term=3,X=year_grid, width = 0.99)

    pd.DataFrame({
        'SAT_Math': math_grid[:,0],
        'Math_Effect': math_effect,
        'Math Confidence Lower': math_confi[:,0],
        'Math Confidence Upper': math_confi[:,1]
    }).to_csv('ct_info/ct_math_spline.csv',index=False)

    pd.DataFrame({
        'SAT_Reading': reading_grid[:,1],
        'Reading_Effect': reading_effect,
        'Reading Confidence Lower': reading_confi[:,0],
        'Reading Confidence Upper': reading_confi[:,1]
    }).to_csv('ct_info/ct_reading_spline.csv',index=False)

    pd.DataFrame({
        'Year': year_grid[:,2],
        'Year_Effect': year_effect,
        'Year Confidence Lower': year_confi[:,0],
        'Year Confidence Upper': year_confi[:,1]
    }).to_csv('ct_info/ct_year_spline.csv',index=False)

    pd.DataFrame({
        'DRG': drg_grid[:,3],
        'DRG_Effect': drg_effect,
        'DRG Confidence Lower': drg_confi[:,0],
        'DRG Confidence Upper': drg_confi[:,1]
    }).to_csv('ct_info/ct_drg_spline.csv',index=False)

    pd.DataFrame({
    'Spline': spline_score,
    'Stat Values': r2,
    'coeff': gam.coef_[-1]
    }).to_csv('ct_info/ct_spline_stats.csv',index=False)