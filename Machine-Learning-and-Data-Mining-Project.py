import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib.colors import ListedColormap
import seaborn
import itertools
import scipy as sp
import sklearn
import statsmodels.api as sm
import statsmodels
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import random
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import datetime
import copy


#Data loading
df = pd.read_csv("/Users/danboguslavsky/Downloads/weatherAUS.csv")


#Create location map and distance from country center
cities_loc = pd.read_csv("/Users/danboguslavsky/Downloads/simplemaps_worldcities_basicv1/worldcities.csv")
cities_loc = cities_loc.loc[cities_loc['country']  == 'Australia']
cities_loc = cities_loc[['city','lat','lng']]

df = pd.read_csv("/Users/danboguslavsky/Downloads/weatherAUS.csv")
cities = df['Location'].unique()

for cit in cities:
    if cit not in cities_loc['city'].to_list():
        print(cit)

loc_dic = {'BadgerysCreek' : (150.7634,-33.8751), 'Cobar' : (145.8389,-31.4958),
           'CoffsHarbour' : (153.1094,-30.2986),'NorahHead' : (151.5667,-33.2833),
           'NorfolkIsland' : (167.9547,-29.0408),'SydneyAirport' : (151.1753,-33.9399),
           'WaggaWagga' : (147.3598,-35.1082),'Williamtown' : (151.8443,-32.8115),
           'Tuggeranong' : (149.0888,-35.4244),'MountGinini' : (148.7723,-35.5294),
           'MelbourneAirport' : (144.8410,-37.6690),'Nhil' : (141.6503,-36.3328),
           'Watsonia' : (145.0830,-37.7080),'Dartmoor' : (141.2730,-37.9144),
           'GoldCoast' : (153.4000,-28.0167),'MountGambier' : (140.7807,-37.8284),
           'Witchcliffe' : (115.1003,-34.0261),'PearceRAAF' : (116.0292,-31.6676),
           'PerthAirport' : (115.9672,-31.9385),'SalmonGums' : (121.6438,-32.9815),
           'Walpole' : (116.7338,-34.9777),'AliceSprings' : (133.8807,-23.6980),
           'Uluru' : (131.0369 ,-25.3444)}

for cit in cities:
    if cit in cities_loc['city'].to_list():
        loc_dic.update({cit: ((cities_loc.loc[cities_loc['city'] == cit]['lng']).to_list()[0],(cities_loc.loc[cities_loc['city'] == cit]['lat']).to_list()[0] )})
    

center_loc = (137.220050,-26.889864)

dist_from_out_loc = {}
for loc in loc_dic:
    dist_from_out_loc.update({loc : ((loc_dic[loc][0] - center_loc[0]) ** 2 + (loc_dic[loc][1] - center_loc[1]) ** 2) ** 0.5})



m = (folium.Map(zoom_start=100, tiles='OpenStreetMap'))

for city in loc_dic:
    folium.Marker((loc_dic[city][1],loc_dic[city][0])).add_to(m)

#Saving map image
m.save("/Users/danboguslavsky/Desktop/School/Machine learning and data mining/AustraliaMap.html")


#Data exploration
df_copy_numeric = df[[col for col in df.columns if is_numeric_dtype(df[col])]]
pd.set_option("display.max_columns", None)
summary_of_numeric_vars = df.describe().round(2)
summary_of_numeric_vars.to_csv('/Users/danboguslavsky/Desktop/School/Machine learning and data mining/numerc_summary.csv')

categorical_vars = ['Location','WindGustDir','WindDir9am',
                  'WindDir3pm','Cloud9am', 'Cloud3pm']

print(df[categorical_vars].describe())

temp = ['MinTemp', 'MaxTemp','Temp9am', 'Temp3pm']
winds = ['WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']
humidity = ['Humidity9am','Humidity3pm']
pressure = ['Pressure9am', 'Pressure3pm']


df.boxplot(column = temp ,grid = False)
df.boxplot(column = winds ,grid = False)
df.boxplot(column = humidity ,grid = False)
df.boxplot(column = pressure ,grid = False)

df.boxplot(column = 'Rainfall' ,grid = False)
df.boxplot(column = 'Evaporation' ,grid = False)
df.boxplot(column = 'Sunshine' ,grid = False)


df['WindGustDir'].value_counts(dropna = False).plot.bar(title = 'WindGustDir')
df['WindDir9am'].value_counts(dropna = False).plot.bar(title = 'WindDir9am')
df['WindDir3pm'].value_counts(dropna = False).plot.bar(title = 'WindDir3pm')
df['Cloud9am'].value_counts(dropna = False).plot.bar(title = 'Cloud9am')
df['Cloud3pm'].value_counts(dropna = False).plot.bar(title = 'Cloud3pm')
plt.figure(dpi = 300)
plt.rcParams.update({'font.size': 6})
df['Location'].value_counts(dropna = False).plot.bar(title = 'Location')
plt.figure(dpi = None)

plt.rcParams.update({'font.size': 9})
df['RainToday'].value_counts(dropna = False).plot.bar(title = 'RainToday')
plt.xticks(rotation = 'horizontal')

df['RainTomorrow'].value_counts(dropna = False).plot.bar(title = 'RainTomorrow')
plt.xticks(rotation = 'horizontal')

pd.DataFrame(df.isnull().sum(
    axis = 0)/len(df)).sort_values(
        ascending = False,by = 0).plot.bar(
            ylim = [0,1],title = '% of NA values by variable',
            legend = False)

df['Date'] = pd.DatetimeIndex(df['Date'])
df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Day'] = pd.DatetimeIndex(df['Date']).day

year_vs_rain = pd.DataFrame(df.groupby(['Year','RainTomorrow']).size()).unstack().transpose()
year_vs_rain.div(year_vs_rain.sum()).transpose().plot.bar(stacked = True,legend = False,title = 'RainTommorow distribution by year')
plt.legend(['No','Yes'],loc = 'lower right')

month_vs_rain = pd.DataFrame(df.groupby(['Month','RainTomorrow']).size()).unstack().transpose()
month_vs_rain.div(month_vs_rain.sum()).transpose().plot.bar(stacked = True,legend = False,title = 'RainTommorow distribution by month')
plt.legend(['No','Yes'],loc = 'lower right')


#correlation plot
corr_plot = df.drop(['Day','Month','Year'],axis = 1).corr()
cmap = seaborn.diverging_palette(250, 10, as_cmap=True)
fig = seaborn.heatmap(corr_plot,cmap=cmap,annot_kws={"fontsize":6},annot=True,fmt=".2f")
plt.tight_layout()
fig.figure.savefig('/Users/danboguslavsky/Desktop/School/Machine learning and data mining/project/corr_plot.png',dpi = 250)


#Check NA vs time
pd.crosstab(df.Sunshine.isna(),df.RainTomorrow == 'Yes')
pd.crosstab(df.Evaporation.isna(),df.RainTomorrow == 'Yes')
pd.crosstab(df.Cloud9am.isna(),df.RainTomorrow == 'Yes')
pd.crosstab(df.Cloud3pm.isna(),df.RainTomorrow == 'Yes')
pd.crosstab(df.Rainfall.isna(),df.RainTomorrow == 'Yes')

plt.plot(df.groupby(['Date']).size())
temp_df = df[['Date','Sunshine']]
plt.plot(temp_df.loc[temp_df.Sunshine.isna()].groupby(['Date']).size())

temp_df = df[['Date','Evaporation']]
plt.plot(temp_df.loc[temp_df.Evaporation.isna()].groupby(['Date']).size())


temp_df = df[['Date','Cloud3pm']]
plt.plot(temp_df.loc[temp_df.Cloud3pm.isna()].groupby(['Date']).size())

rain_vs_winddir = pd.DataFrame(df.groupby(['WindDir9am','RainTomorrow']).size()).unstack().transpose()
rain_vs_winddir.div(rain_vs_winddir.sum()).transpose().plot.bar(stacked = True,legend = False,title = 'RainTommorow distribution by wind direction (9am)')


df[['WindDir9am','WindDir3pm','WindGustDir']].corr(method = 'spearman')

i = 0
for index, row in df.iterrows():
    if row.WindDir9am == row.WindDir3pm:
        i += 1


to_fill_na_vars_freq = ['WindDir9am','WindDir3pm','WindGustDir']
to_fill_na_vars = ['MinTemp','MaxTemp','Rainfall','WindGustSpeed',
                   'WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am']

df_no_na = df

mean_values = {}
for month_loc in itertools.product(df.Month.unique(),df.Location.unique()):
    temp = df[(df.Location == month_loc[1])  & (df.Month == month_loc[0])]
    for var in to_fill_na_vars:
        mean_value = temp[var].mean()
        if pd.isna(mean_value):
            mean_values.update({(month_loc[0],month_loc[1],var):df[var].mean()})
        else:
            mean_values.update({(month_loc[0],month_loc[1],var):mean_value})
        
wind_frequent =  {}
for month_loc in itertools.product(df.Month.unique(),df.Location.unique()):
    temp = df[(df.Location == month_loc[1])  & (df.Month == month_loc[0])]
    for var in to_fill_na_vars_freq:
        most_freq = temp[var].mode()
        if len(most_freq) > 0:
            wind_frequent.update({(month_loc[0],month_loc[1],var):most_freq[0]})
        else:
            wind_frequent.update({(month_loc[0],month_loc[1],var):df[var].mode()[0]})
            

na_vars = to_fill_na_vars_freq + to_fill_na_vars

mean_values.update(wind_frequent)
            
for row in range(len(df_no_na)):
    for col in na_vars:
        if pd.isna(df_no_na.loc[row,col]):
            month = df_no_na.loc[row,'Month']
            location = df_no_na.loc[row,'Location']
            df_no_na.loc[row,col] = mean_values[(month,location,col)]
    print(row)

    
#Plot to check affect on NAs removal
pd.DataFrame(df_no_na.isnull().sum(
    axis = 0)/len(df)).sort_values(
        ascending = False,by = 0).plot.bar(
            ylim = [0,1],title = '% of NA values by variable',
            legend = False)

df_no_na['dist_from_center'] = None
for i in range(len(df_no_na)):
    df_no_na.loc[i,'dist_from_center'] = dist_from_out_loc[df_no_na.loc[i,'Location']]
#Generate is_winter var
df_no_na['is_winter'] = np.where((5<df_no_na['Month']) & (df_no_na['Month']<9),1,0)  


X = df_no_na[df['Sunshine'].notna()][to_fill_na_vars + ['is_winter','dist_from_center']]


X_linear_reg = statsmodels.tools.tools.add_constant(X)
sunshine = df['Sunshine'].dropna()
sunshine_transformed  = [x+0.000001 for x in sunshine]
#Boxcox:
box_cox = sp.stats.boxcox(sunshine_transformed)
sunshine_transformed = box_cox[0]
lambda_sunshine = box_cox[1]


train, test  = train_test_split(list(range(len(sunshine))), test_size = 0.2,shuffle=True, random_state = 0)
X_train, Y_train, X_test, Y_test = X.iloc[train], sunshine.iloc[train], X.iloc[test], sunshine.iloc[test]
Y_train_reg, Y_test_reg = sunshine_transformed[train], sunshine_transformed[test]
X_train_reg = statsmodels.tools.tools.add_constant(X_train)

#Prediction of other missing variables

#sunshine - 


#OLS:
model_sunshine = sm.OLS(Y_train_reg.astype(float), X_train_reg.astype(float)).fit()
model_sunshine.summary()
X_test_reg = statsmodels.tools.tools.add_constant(X_test)
Y_hat = model_sunshine.predict(X_test_reg)
mean_squared_error(Y_test_reg, Y_hat) #2.34

#KNN:
for size in [10,100,1000]:
    KNN =   KNeighborsRegressor(n_neighbors = size)
    KNN.fit(X_train,Y_train)
    Y_hat = KNN.predict(X_test)
    print("KNN for size", size,":",mean_squared_error(Y_test, Y_hat)) # about 6-7

#Random forest
RFreg = RandomForestRegressor(criterion='mse')
RFreg.fit(X_train,Y_train)
y_hat = RFreg.predict(X_test)
mean_squared_error(Y_test, y_hat)#7.19

model_sunshine = sm.OLS(sunshine_transformed.astype(float), X_linear_reg.astype(float)).fit()

#Insert predicted values to sunshine 
for row in range(len(df_no_na)):
    if pd.isna(df_no_na.loc[row,'Sunshine']):
        X_row = np.array(df_no_na.loc[row,X.columns]).reshape(1,-1)
        X_row = np.insert(X_row,0,1)#add a constant
        Y_hat = model_sunshine.predict(X_row)[0]
        Y_hat = sp.special.inv_boxcox(Y_hat,lambda_sunshine)
        if Y_hat > 14.5: #max_value
            Y_hat = 14.5
        elif pd.isna(Y_hat):
            Y_hat = 0
        df_no_na.loc[row,'Sunshine'] = Y_hat
    print(row)
    

plt.hist(df['Sunshine'])
plt.hist(df_no_na['Sunshine'])
df_no_na['Sunshine'].isna().any()

#evaporation has one major outlier: 
#We decide to leave it as is.
X = df_no_na[df['Evaporation'].notna()][to_fill_na_vars + ['is_winter','dist_from_center','Sunshine']]


evaporation = df['Evaporation'].dropna()
evaporation_transformed  = [x+0.000001 for x in evaporation]
#Boxcox:
box_cox = sp.stats.boxcox(evaporation_transformed)
evaporation_transformed = box_cox[0]
lambda_evaporation = box_cox[1]

train, test  = train_test_split(list(range(len(evaporation))), test_size = 0.2,shuffle=True, random_state = 0)
X_train, Y_train, X_test, Y_test = X.iloc[train], evaporation.iloc[train], X.iloc[test], evaporation.iloc[test]
Y_train_reg, Y_test_reg = evaporation_transformed[train], evaporation_transformed[test]
X_train_reg = statsmodels.tools.tools.add_constant(X_train)

model_evaporation = sm.OLS(Y_train_reg.astype(float), X_train_reg.astype(float)).fit()
model_evaporation.summary()

#Rainfall was unsegnificant -> we will drop it
X_train_reg = X_train_reg.drop(['Rainfall'],axis=1)

model_evaporation = sm.OLS(Y_train_reg.astype(float), X_train_reg.astype(float)).fit()
model_evaporation.summary()

X_test_reg = statsmodels.tools.tools.add_constant(X_test)
X_test_reg = X_test_reg.drop(['Rainfall'],axis=1)

model_evaporation = sm.OLS(Y_test_reg.astype(float), X_test_reg.astype(float)).fit()
Y_hat = model_evaporation.predict(X_test_reg)
mean_squared_error(Y_test_reg, Y_hat) #0.7

for size in [10,100,1000]:
    KNN =   KNeighborsRegressor(n_neighbors = size)
    KNN.fit(X_train,Y_train)
    Y_hat = KNN.predict(X_test)
    print("KNN for size", size,":",mean_squared_error(Y_test, Y_hat)) # about 8-9

RFreg = RandomForestRegressor(criterion='mse')
RFreg.fit(X,evaporation)
y_hat = RFreg.predict(X_test)
mean_squared_error(Y_test, y_hat)#1.19

X_full_reg = statsmodels.tools.tools.add_constant(X.drop(['Rainfall'],axis=1))
model_evaporation = sm.OLS(evaporation_transformed.astype(float), X_full_reg.astype(float)).fit()

#Insert predicted values to evaporation
for row in range(len(df_no_na)):
    if pd.isna(df_no_na.loc[row,'Evaporation']):
        X_row = np.array(df_no_na.loc[row,X.drop(['Rainfall'],axis=1).columns]).reshape(1,-1)
        X_row = np.insert(X_row,0,1)#add a constant
        Y_hat = model_evaporation.predict(X_row)[0]
        Y_hat = sp.special.inv_boxcox(Y_hat,lambda_evaporation)
        if Y_hat > 145: #max_value
            Y_hat = 145
        elif pd.isna(Y_hat): #only if negative before inverse boxcox
            Y_hat = 0
        df_no_na.loc[row,'Evaporation'] = Y_hat
    print(row)


plt.hist(df['Evaporation'])
plt.hist(df_no_na['Evaporation'])
df_no_na['Evaporation'].isna().any()

#Correct clouds (there are some '9' value):
df_no_na.loc[np.where(df_no_na['Cloud9am'] == 9)[0],'Cloud9am'] = 8
df_no_na.loc[np.where(df_no_na['Cloud3pm'] == 9)[0],'Cloud3pm'] = 8

#Predict clouds

X = df_no_na[df['Cloud9am'].notna()][to_fill_na_vars + ['is_winter','dist_from_center','Sunshine','Evaporation']]
Cloud9am = df['Cloud9am'].dropna()

#Use Desicion tree regression and round values
clouds_model = DecisionTreeRegressor(random_state=(0))
clouds_model.fit(X, Cloud9am)

#Insert predicted values to Cloud9am
for row in range(len(df_no_na)):
    if pd.isna(df_no_na.loc[row,'Cloud9am']):
        X_row = np.array(df_no_na.loc[row,X.columns]).reshape(1,-1)
        Y_hat = clouds_model.predict(X_row)[0]
        if Y_hat > 8: #max_value
            Y_hat = 8
        elif Y_hat < 0: #min value
            Y_hat = 0
        df_no_na.loc[row,'Cloud9am'] = Y_hat
    print(row)


X = df_no_na[df['Cloud3pm'].notna()][to_fill_na_vars + ['is_winter','dist_from_center','Sunshine','Evaporation']]
Cloud3pm = df['Cloud3pm'].dropna()

#Use Desicion tree regression and round values
clouds_model = DecisionTreeRegressor(random_state=(0))
clouds_model.fit(X, Cloud3pm)

#Insert predicted values to Cloud9am
for row in range(len(df_no_na)):
    if pd.isna(df_no_na.loc[row,'Cloud3pm']):
        X_row = np.array(df_no_na.loc[row,X.columns]).reshape(1,-1)
        Y_hat = clouds_model.predict(X_row)[0]
        if Y_hat > 8: #max_value
            Y_hat = 8
        elif Y_hat < 0: #min value
            Y_hat = 0
        df_no_na.loc[row,'Cloud3pm'] = Y_hat
    print(row)
    
#Transform wind direcion to binary vars:
def wind_dir_binary(direction):
    '''
    N = loc[0]
    E = loc[1]
    S = loc[2]
    W = loc[3]
    '''
    bin_vec = np.array(['N' in direction,'E' in direction,'S' in direction,'W' in direction])
    bin_vec = bin_vec.astype(int)
    return bin_vec

dir3pm = []
for index,row in df_no_na.iterrows():
    dir3pm.append(wind_dir_binary(row.WindDir3pm))
    
dir9am = []
for index,row in df_no_na.iterrows():
    dir9am.append(wind_dir_binary(row.WindDir9am))
    
gustdir = []
for index,row in df_no_na.iterrows():
    gustdir.append(wind_dir_binary(row.WindGustDir))
    
    
df_no_na =  df_no_na.assign(dir3pm_N = np.array(dir3pm).transpose()[0],
                            dir3pm_E = np.array(dir3pm).transpose()[1],
                            dir3pm_S = np.array(dir3pm).transpose()[2],
                            dir3pm_W = np.array(dir3pm).transpose()[3])

df_no_na =  df_no_na.assign(dir9am_N = np.array(dir9am).transpose()[0],
                            dir9am_E = np.array(dir9am).transpose()[1],
                            dir9am_S = np.array(dir9am).transpose()[2],
                            dir9am_W = np.array(dir9am).transpose()[3]) 

df_no_na =  df_no_na.assign(gustdir_N = np.array(gustdir).transpose()[0],
                            gustdir_E = np.array(gustdir).transpose()[1],
                            gustdir_S = np.array(gustdir).transpose()[2],
                            gustdir_W = np.array(gustdir).transpose()[3]) 

#check correlation with wind variables
directions = ['N','W','S','E']
_vars = ['dir3pm_','dir9am_','gustdir_']
for d in directions:
    for var1 in _vars:
        for var2 in _vars:
            _var1 = var1+d
            _var2 = var2+d
            print("Correlation in direction", d, "for", var1, "and", var2,":", accuracy_score(df_no_na[_var1],df_no_na[_var2]))


clean_data = df_no_na.drop(['Temp3pm','Temp9am','Pressure3pm','gustdir_N','gustdir_E','gustdir_S','gustdir_W',
                            'WindGustDir','WindDir9am','WindDir3pm','RainToday'],axis = 1)
clean_data = clean_data.dropna(subset = ['RainTomorrow'])

#Process data from LSTM form
def location_split(data):
    assert str(type(data)) == "<class 'pandas.core.frame.DataFrame'>"
    assert ('Location' in data.columns and 'Year' in data.columns and
            'Month' in data.columns and 'Day' in data.columns)
    locations = data['Location'].unique()
    areas = []
    for c,loc in enumerate(locations):
        print("Splitting locations", str(round((c/len(locations))*100,2)),"% Done.")
        temp_df = data[data['Location'] == loc]
        temp_df['Date'] = None
        for i in range(len(temp_df)):
            temp_df['Date'].iloc[i] = datetime.date(
                temp_df['Year'].iloc[i], temp_df['Month'].iloc[i], temp_df['Day'].iloc[i])
            temp_df = temp_df.sort_values('Date')
            #temp_df = temp_df.drop(['Year','Month','Day','Location'],axis = 1)
        areas.append(temp_df)
    return areas
        

def data_blocks(data, block_size, stride = 1):
    empty_block_count = 0
    not_sequence_count = 0
    #data = location_split(data)
    sequences = []
    for c,loc in enumerate(data):
        print("Creating blocks", str(round((c/len(data))*100,2)),"% Done.")
        block = []
        for i in range(len(loc)):
            for j in range(block_size):
                if j == 0:
                    block.append(loc.iloc[i])
                    continue
                if i + block_size < len(loc):
                    if (loc['Date'].iloc[i] + datetime.timedelta(j)) == loc['Date'].iloc[i+j]:
                        block.append(loc.iloc[i+j])
                    else:
                        block = []
                        not_sequence_count += 1
                        break
                else:
                    block = []
                    break

            if block == []:
                empty_block_count += 1
                continue
            else:
                sequences.append(block)
                block = []
    return sequences, empty_block_count, not_sequence_count


    
def make_sequences(data,seq_size):
    splitted_data = location_split(data)
    sequences, empty_block_count, not_sequence_count = data_blocks(splitted_data,seq_size)
    _sequences = copy.deepcopy(sequences)
    new_appended_data = []
    new_y_s = []
    for i,seq in enumerate(sequences):
        for j in range(len(sequences[i])):
            sequences[i][j].RainTomorrow = 1 if sequences[i][j].RainTomorrow == "Yes" else 0
            new_y_s.append(sequences[i][j].RainTomorrow)
            sequences[i][j].dist_from_center = float(sequences[i][j].dist_from_center)
            sequences[i][j] = sequences[i][j].drop(['Date','Year','Month','Day','Location','RainTomorrow'])
            t = np.array(sequences[i][j])
            new_appended_data.append(t)
        if not len(new_appended_data)%seq_size == 0:
            break
        print(len(new_appended_data)%seq_size == 0,len(new_y_s)%seq_size == 0, str(i/len(sequences)))
    return new_appended_data, new_y_s, _sequences

new_appended_data, new_y_s, _sequences = make_sequences(clean_data,4)

### LSTM fitting:
scaler = MinMaxScaler()
scaler.fit(new_appended_data)
clean_data_scaled = scaler.transform(new_appended_data)


X = np.array(clean_data_scaled)
Y = np.array(new_y_s)

X = X.reshape(137615,4,23)
Y = Y.reshape(137615,4,1)


kf = KFold(n_splits = 5)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    model = keras.Sequential(
    [
    layers.LSTM(256,input_shape = (4,23),return_sequences=True),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='relu'),
    ]
    )
    
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x = X_train, y = y_train, batch_size=8,epochs = 3)
    y_pred = model.predict_classes(X_train)
    y_pred = y_pred.transpose()[0][-1]#the last day - what we want to predict
    y_true = y_train.transpose()[0][-1]
    print("train accuracy score:", str(accuracy_score(y_true,y_pred)))
    print("")
    print(classification_report(y_pred, y_true))
    
    y_pred = model.predict_classes(X_test)
    y_pred = y_pred.transpose()[0][-1]#the last day - what we want to predict
    y_true = y_test.transpose()[0][-1]
    print("test accuracy score:", str(accuracy_score(y_true,y_pred)))
    print("")
    print(classification_report(y_pred, y_true))
    print("-- -- -- -- -- -- -- -- -- -- --")

### Random Forest fitting:
clean_data['RainTomorrow'] = [1 if x == 'Yes' else 0 for x in clean_data['RainTomorrow']]
clean_data['dist_from_center'] = clean_data['dist_from_center'].astype(float)

clean_data = clean_data.drop(['Date','Year','Month','Day','Location'],axis = 1)

data_for_rf = clean_data
X = data_for_rf.drop(['RainTomorrow'],axis = 1)
Y = data_for_rf['RainTomorrow']
Y = np.array(Y)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

kf = KFold(n_splits = 5)
for min_sample_split in [2, 3, 4, 5, 6, 8, 10]:
    print("min sample split:",  str(min_sample_split))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
        RF = RandomForestClassifier(n_estimators = 400,
                                    min_samples_split = min_sample_split,
                                    max_features = 'sqrt')
        RF.fit(X = X_train, y = y_train)
        y_pred = RF.predict(X_train)
        print("train accuracy score:", str(accuracy_score(y_train,y_pred)))
        print("")
        print(classification_report(y_pred, y_train))
        
        y_pred = RF.predict(X_test)
        print("test accuracy score:", str(accuracy_score(y_test,y_pred)))
        print("")
        print(classification_report(y_pred, y_test))
        print("-- -- -- -- -- -- -- -- -- -- --")

### XGB fitting:

kf = KFold(n_splits = 5)
for trees in [400,500]:
    print("Trees:",  str(trees))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
        GBC = GradientBoostingClassifier(n_estimators = trees,
                                    min_samples_split = 10,
                                    max_features = 'sqrt')
        GBC.fit(X = X_train, y = y_train)
        y_pred = GBC.predict(X_train)
        print("train accuracy score:", str(accuracy_score(y_train,y_pred)))
        print("")
        print(classification_report(y_pred, y_train))
        
        y_pred = GBC.predict(X_test)
        print("test accuracy score:", str(accuracy_score(y_test,y_pred)))
        print("")
        print(classification_report(y_pred, y_test))
        print("-- -- -- -- -- -- -- -- -- -- --")
            

#Best model with under/over sampling evaluation:
ones = clean_data.iloc[np.where(clean_data['RainTomorrow'] == 1)]
zeros = clean_data.iloc[np.where(clean_data['RainTomorrow'] == 0)]

#Under sampling:
new_zeros = zeros.iloc[np.random.choice(zeros.shape[0], 31877,replace = False)]
balances_data = ones.append(new_zeros)
X = balances_data.drop(['RainTomorrow'],axis = 1)
Y = balances_data['RainTomorrow']
Y = np.array(Y)
X = np.array(X)
this_list = list(range(len(X)))
random.shuffle(this_list)
X = X[this_list]
Y = Y[this_list]


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

RF = RandomForestClassifier(n_estimators = 400,
                            min_samples_split = 10,
                            max_features = 'sqrt')
RF.fit(X = X_train, y = y_train)
y_pred = RF.predict(X_train)
print("train accuracy score:", str(accuracy_score(y_train,y_pred)))
print("")
print(classification_report(y_pred, y_train))

y_pred = RF.predict(X_test)
print("test accuracy score:", str(accuracy_score(y_test,y_pred)))
print("")
print(classification_report(y_pred, y_test))
print("-- -- -- -- -- -- -- -- -- -- --")

confusion_matrix(y_test,y_pred)


#Over sampling:


kf = KFold(n_splits = 5)
for train_index, test_index in kf.split(clean_data):
    X = clean_data.drop(['RainTomorrow'],axis = 1)
    Y = clean_data['RainTomorrow']
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    X_ones = X_train.iloc[np.where(y_train == 1)]
    X_zeros = X_train.iloc[np.where(y_train == 0)]
    Y_ones = y_train.iloc[np.where(y_train == 1)]
    Y_zeros = y_train.iloc[np.where(y_train == 0)]
    
    X_new_ones = X_ones.append(X_ones)
    Y_new_ones = Y_ones.append(Y_ones)
    for _ in range(round(len(X_zeros)/len(X_ones))-2):
        X_new_ones = X_new_ones.append(X_ones)
        Y_new_ones = Y_new_ones.append(Y_ones) 
        samples = np.random.choice(X_ones.shape[0], (len(X_zeros)%len(X_ones)),replace = False)
    X_new_ones = X_new_ones.append(X_ones.iloc[samples])
    Y_new_ones = Y_new_ones.append(Y_ones.iloc[samples])
    
    X_balanced = X_new_ones.append(X_zeros)
    Y_balanced = Y_new_ones.append(Y_zeros)
    this_list = list(range(len(X_balanced)))
    random.shuffle(this_list)
    X = X_balanced.iloc[this_list]
    Y = Y_balanced.iloc[this_list]

   
    RF = RandomForestClassifier(n_estimators = 400,
                               min_samples_split = 10,
                               max_features = 'sqrt')
    RF.fit(X = X, y = Y)
    y_pred = RF.predict(X)
    print("train accuracy score:", str(accuracy_score(Y,y_pred)))
    print("")
    print(classification_report(y_pred, Y))
   
    y_pred = RF.predict(X_test)
    print("test accuracy score:", str(accuracy_score(y_test,y_pred)))
    print("")
    print(classification_report(y_pred, y_test))
    print("-- -- -- -- -- -- -- -- -- -- --")

confusion_matrix(y_test,y_pred)



#Cross validation hyperparameter tuning charts code: (To be set with values as needed)

ax = plt.subplot(111)
plt.ylim(bottom = 0.7)
plt.ylim(top = 1)

x = list(range(3))
ax.bar(x=[_x-0.2 for _x in x],height = [0.84,0.7998,0.84796],width = 0.4, align='center',label = "Accuracy")
ax.bar(x=[_x+0.2 for _x in x],height = [0.7685,0.8,0.762009162],width = 0.4, align='center',label = "F1")
ax.set_xticks(x)
ax.set_xticklabels(['Oversampling','Undersampling','Unbalanced'])
plt.legend(loc = "upper right")
plt.title("Balanced vs Unbalanced data with best random forest model")


plt.ylim(bottom = 0.3)
plt.plot([50,100,200,300,400,500,600],[0.759296978,0.762109948,0.762109948,0.761558612,0.763112418,0.762561151,0.762109948] , label = "F1")
plt.plot([50,100,200,300,400,500,600],[0.8464,0.8476,0.8476,0.8478,0.8482,0.8484,0.8479], label = "Accuracy")
plt.legend(loc = "upper right")
plt.title("Random Forest - Number of trees tuning")
