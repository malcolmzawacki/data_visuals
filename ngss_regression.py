
import pandas as pd, numpy as np
from pygam import GAM, s, te, f
from sklearn.metrics import r2_score, mean_squared_error
import datetime
from pygam.distributions import BinomialDist

df = pd.read_csv('ngss_data.csv')
train = df[df['Year']> 2021]
test = df[df['Year'] < 2026]

X_train = train[['SAT_Math','SAT_Reading','Year']].values
y_train = train['NGSS'].values

X_test = test[['SAT_Math','SAT_Reading','Year']].values
y_test = test['NGSS'].values

dist = 'normal'
link = 'identity'
normalize = False
if normalize:
    
    for col in ['SAT_Reading','SAT_Math']:
        df[f'{col}_z'] = (df[col]-df[col].mean())/df[col].std()

    df['y_bin'] = ((df['NGSS'] - 1000)/199).astype(int)

    X = df[['SAT_Reading_z','SAT_Math_z','Year']].values
    y = df['y_bin'].values

    pd.DataFrame({
        'Year': df['Year'],
        'SAT_Math_z': df['SAT_Math_z'],
        'SAT_Read_z': df['SAT_Reading_z'],
        'NGSS': df['NGSS']
    }).to_csv('SAT_z_vals.csv',index=False)
    
    pd.DataFrame({
        'SAT_Math_mean and std': [df['SAT_Math'].mean(),df['SAT_Math'].std()],
        'SAT_Read_mean and std': [df['SAT_Reading'].mean(),df['SAT_Reading'].std()]
    }).to_csv('SAT_stats.csv',index=False)
    X_train = X
    y_train = y
    dist = BinomialDist(levels=199)
    link = 'logit'

spline_stat_df = pd.read_csv('spline_stats.csv')
spline_num = spline_stat_df['Spline'].values
spline_step = 2
current_spline = list(spline_num)
spline_test = []
spline_score = None
r2 = None
lam = 1.0
tensor = False

train = False
if train:
    start = datetime.datetime.now()
    for i in range(current_spline[0], current_spline[0] + spline_step):
        print(i-current_spline[0])
        for j in range(current_spline[1], current_spline[1]+  spline_step):
            for k in range(current_spline[2], current_spline[2] + spline_step):
                for l in range(current_spline[3], current_spline[3] + spline_step):

                    terms = (
                        s(0, n_splines=i, constraints='monotonic_inc') 
                        + s(1, n_splines=j, constraints='monotonic_inc') 
                        + s(2,  n_splines=k) 
                        + te(0,1, n_splines=l,lam=lam)
                        )
                    gam = GAM(
                        terms = terms ,
                        distribution=dist,
                        link = link
                ).fit(X_train, y_train)

                candidate_r2 = gam.statistics_['pseudo_r2']['explained_deviance']
                candidate_spline_score = [i,j,k,l]
                if spline_score == None:
                    spline_score = candidate_spline_score
                if r2 == None:
                    r2 = candidate_r2
                if candidate_r2 > r2:
                    r2 = candidate_r2
                    spline_score = candidate_spline_score
                    print(r2)
    print("Time: "+ str(datetime.datetime.now() - start))
else:
    spline_score = current_spline
    i, j, k, l = tuple(current_spline)
    terms = (
        s(0, n_splines=i, constraints='monotonic_inc') 
        + s(1, n_splines=j, constraints='monotonic_inc') 
        + s(2,  n_splines=k)
        + te(0,1, n_splines=l,lam=lam)) if tensor == True else (s(0, n_splines=i, constraints='monotonic_inc') 
        + s(1, n_splines=j, constraints='monotonic_inc') 
        + s(2,  n_splines=k)
        )
    gam = GAM(
        terms = terms ,
        distribution=dist,
        link = link
    ).fit(X_train, y_train)
    r2 = gam.statistics_['pseudo_r2']['explained_deviance']


print(spline_score, r2)
#gam.summary()

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
    }).to_csv('math_spline.csv',index=False)

    pd.DataFrame({
        'SAT_Reading': reading_grid[:,1],
        'Reading_Effect': reading_effect,
        'Reading Confidence Lower': reading_confi[:,0],
        'Reading Confidence Upper': reading_confi[:,1]
    }).to_csv('reading_spline.csv',index=False)

    pd.DataFrame({
        'Year': year_grid[:,2],
        'Year_Effect': year_effect,
        'Year Confidence Lower': year_confi[:,0],
        'Year Confidence Upper': year_confi[:,1]
    }).to_csv('year_spline.csv',index=False)
    if tensor:
        tensor_x, tensor_y = gam.generate_X_grid(term = 3, meshgrid=True, n=60)
        tensor_effect = gam.partial_dependence(term=3, X=(tensor_x, tensor_y), meshgrid=True)
        tensor_df = pd.DataFrame(
            tensor_effect, 
            index=np.round(tensor_x[:,0],2), # rows = math
            columns = np.round(tensor_y[0],2) # cols = read
        ).to_csv('tensor_surface.csv')


pd.DataFrame({
    'Spline': spline_score,
    'Stat Values': r2,
    'coeff': gam.coef_[-1]
 }).to_csv('spline_stats.csv',index=False)

if tensor:
    tensor_df = pd.read_csv('tensor_surface.csv', index_col = 0)
    tensor_df.columns = tensor_df.columns.astype(float)
    tensor_df.index = tensor_df.index.astype(float)
    tensor_effects = []

    for i in range(len(X_train)):
        x0, y0 = X_train[i,0], X_train[i,1]
        closest_x = min(np.round(tensor_x[:,0],2), key = lambda x:abs(x - x0))
        closest_y = min(np.round(tensor_y[0],2), key = lambda y: abs(y - y0))
        best_match = tensor_df.loc[closest_x, closest_y]
        tensor_effects.append(best_match)

    pd.DataFrame({
        'Tensor Effect': tensor_effects
    }).to_csv('tensor_effect.csv',index=False)


