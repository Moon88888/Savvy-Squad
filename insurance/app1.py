from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open('clm.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    data13 = request.form['m']
    data14 = request.form['n']
    data15 = request.form['o']
    data16 = request.form['p']
    data17 = request.form['q']
    data18 = request.form['r']
    data19 = request.form['s']
    data20 = request.form['t']
    data21 = request.form['u']
    data22 = request.form['v']
    data23 = request.form['w']
    data24 = request.form['x']
    data25 = request.form['y']
    data26 = request.form['z']
    data27 = request.form['a1']
    data28 = request.form['a2']
    

    #Mapping
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

    data1=vp_dict.get(data1)
    data2=dpa_dict.get(data2)
    data3=dpc_dict.get(data3)
    data4=pnoc_dict.get(data4)
    data5=aov.get(data5)
    data6=aoph.get(data6)
    data7=nos.get(data7)
    data8=acc.get(data8)
    data9=noc.get(data9)
    data10=f.get(data10)
    data11=pt.get(data11)


    

    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data23,data24,data25,data26,data27,data28]])

    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)