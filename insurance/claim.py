
import pandas as pd
import pickle
import numpy as np

df=pd.read_csv("carclaims.csv")
df1=pd.get_dummies(df,columns=['AccidentArea','Sex','Fault','PoliceReportFiled','WitnessPresent','AgentType'],drop_first=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['Make']=le.fit_transform(df1['Make'])
df1['MaritalStatus']=le.fit_transform(df1['MaritalStatus'])
df1['VehicleCategory']=le.fit_transform(df1['VehicleCategory'])
df1['BasePolicy']=le.fit_transform(df1['BasePolicy'])


vp_dict={'more than 69,000':6, '20,000 to 29,000':2, '30,000 to 39,000':3,
       'less than 20,000':1, '40,000 to 59,000':4, '60,000 to 69,000':5}
dpa_dict={'more than 30':5, '15 to 30':4, 'none':1, '1 to 7':2, '8 to 15':3}
dpc_dict={'more than 30':4, '15 to 30':3, 'none':1, '8 to 15':2}
pnoc_dict={'none':1, '1':2, '2 to 4':3, 'more than 4':4}
aov={'3 years':3, '6 years':6, '7 years':7, 'more than 7':8, '5 years':5, 'new':1,
       '4 years':4, '2 years':2}
aoph={'26 to 30':4, '31 to 35':5, '41 to 50':7, '51 to 65':8, '21 to 25':3,
       '36 to 40':6, '16 to 17':1, 'over 65':9, '18 to 20':2}
nos={'none':1, 'more than 5':4, '3 to 5':3, '1 to 2':2}
acc={'1 year':2, 'no change':1, '4 to 8 years':4, '2 to 3 years':3,
       'under 6 months':5}
noc={'3 to 4':3, '1 vehicle':1, '2 vehicles':2, '5 to 8':4, 'more than 8':5}
f={'No':0, 'Yes':1}
pt={'Sport - Liability':0, 'Sport - Collision':1, 'Sedan - Liability':4,
       'Utility - All Perils':9, 'Sedan - All Perils':6, 'Sedan - Collision':5,
       'Utility - Collision':8, 'Utility - Liability':7, 'Sport - All Perils':3}

df1['VehiclePrice']=df['VehiclePrice'].map(vp_dict)
df1['Days:Policy-Accident']=df['Days:Policy-Accident'].map(dpa_dict)
df1['Days:Policy-Claim']=df['Days:Policy-Claim'].map(dpc_dict)
df1['PastNumberOfClaims']=df['PastNumberOfClaims'].map(pnoc_dict)
df1['AgeOfVehicle']=df['AgeOfVehicle'].map(aov)
df1['AgeOfPolicyHolder']=df['AgeOfPolicyHolder'].map(aoph)
df1['NumberOfSuppliments']=df['NumberOfSuppliments'].map(nos)
df1['AddressChange-Claim']=df['AddressChange-Claim'].map(acc)
df1['NumberOfCars']=df['NumberOfCars'].map(noc)
df1['FraudFound']=df['FraudFound'].map(f)
df1['PolicyType']=df['PolicyType'].map(pt)



df1.drop(['VehicleCategory','BasePolicy','Age','Year'],axis=1,inplace=True)
df1.drop(df1[df1['PolicyNumber']==1517].index,inplace=True)
df1=df1.reset_index()
df1.drop('index',axis=1,inplace=True)


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
x = df1.drop(['FraudFound'], axis=1)
y = df1['FraudFound']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

model_lr =lr.fit(x_train,y_train)
pickle.dump(model_lr, open('clm.pkl', 'wb'))
