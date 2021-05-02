from django.shortcuts import render
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LogisticRegression
import datetime
import pickle


def arrgen(request):
    if request.method == 'POST':
        age = request.POST['age']
        if 'fever' in request.POST:
            fever = request.POST['fever']
        else:
            fever = False
        tired = request.POST['tired']
        pains = request.POST.get('pains')
        pains = 1 if pains else 0
        cough = request.POST.get('cough')
        cough = 1 if cough else 0
        nasal_congestion = request.POST.get('nasal_congestion')
        nasal_congestion = 1 if nasal_congestion else 0
        breath = request.POST.get('breathe')
        breath = 1 if breath else 0
        runny_nose = request.POST.get('runny_nose')
        runny_nose = 1 if runny_nose else 0
        sore_throat = request.POST.get('sore_throat')
        sore_throat = 1 if sore_throat else 0
        none = request.POST.get('none')
        none = 1 if none else 0
        nopain= request.POST.get('no_pain')
        nopain= 1 if nopain else 0
    
        
        usrin = np.array(
            (
             fever,
             tired,
             cough,
             breath,
             sore_throat,
             none,
             pains,
             nasal_congestion,
             runny_nose,
             nopain,
             age
             )
        ).reshape(1, 11)

    return render(request, "page.html",{"usrin":usrin})



        
       

df = pd.read_csv("C:/Users/Lenovo/Desktop/hackathon/medivid/predictor/src/dataset.csv")

l1=["Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat", "None_Sympton", "Pains", "Nasal-Congestion", "Runny-Nose", "None_Experiencing", "Age", "Infection_Prob"]

df = df[l1]
df = df.dropna()

def predictcovid(xin):

    def data_split(data, ratio):
        np.random.seed(42)
        train, test = model_selection.train_test_split(data, test_size=ratio)
        return train, test

   
    def data_split(data, ratio):
  
        train, test = model_selection.train_test_split(data, test_size=ratio)
        return train, test

    train_data, test_data = data_split(df, 0.2)

    #Train Test Data
    x_train = train_data[["Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat", "None_Sympton", "Pains", "Nasal-Congestion", "Runny-Nose", "None_Experiencing", "Age"]].to_numpy()
    x_test = test_data[["Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat", "None_Sympton", "Pains", "Nasal-Congestion", "Runny-Nose", "None_Experiencing", "Age"]].to_numpy()

    #print(x_train)
    #Train Test Data

    y_train = train_data[["Infection_Prob"]]
    y_test = test_data[["Infection_Prob"]]

    #print(y_train)
    # Replacing infinite with nan
    df.replace(-1, 1, inplace=True)

    # Dropping all the rows with nan values
    df.dropna(inplace=True)

    df.head()

    np.ravel(y_train)
    np.ravel(y_test)

    
    model = LogisticRegression()
    model.fit(x_train, y_train)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    yout=model.predict(xin)
    if(yout==0):
        return "Infected"
    else:
        return "not Infected"

def probab(xin):
    cnt=0
    if(xin[0][0]>=99):
        cnt=1
    else:
        cnt=0
    for i in xin:
        for j in i:
            if(j==1):
                cnt=cnt+1
    return(cnt/7)

def home(request):
    return render(request,'home.html',{})

def self(request):
    return render(request,"self.html")

def covid(request):
    return render(request,"covid.html")

def video(request):
    return render(request,"video.html")

#arr=[[[[94.33,0,1,0,1,0,1,1,1,0,21]]]]
def model(request):
    usrin=arrgen()
    usrin=[usrin1]
    data = [predictcovid(usrin ) , probab(usrin)]
    return render(request, 'page.html', {'data':data})

# def model1(request):
#     xnew1=[[94.33,0,1,0,1,0,1,1,1,0,21]]
#     data1 = probab(xnew1)
#     return render(request, 'page.html', {'data1':data1})

