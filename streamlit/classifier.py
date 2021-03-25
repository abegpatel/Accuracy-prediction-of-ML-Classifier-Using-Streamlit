# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:03:01 2021

@author: Abeg
""" 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
st.title("streamlit example")
st.write("""
         # Explore different Classifier
         which one is best
         """)
dataset_name=st.sidebar.selectbox("Select Dataset",["Iris","Breast Cancer","Wine Dataset"])    
st.sidebar.write(dataset_name)     
classifier_name=st.sidebar.selectbox("Select classifier",["KNN","SVM","Random Forest"])    
st.sidebar.write(classifier_name)

def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    X=data.data
    y=data.target
    return X,y
X,y=get_dataset(dataset_name)
st.write("shape of the dataset",X.shape,end="")
st.write("no of classes",len(np.unique(y)),end="")

def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=="KNN":
        k=st.sidebar.slider("k",1,15)
        params["k"]=k
    elif clf_name=="SVM":
        C=st.sidebar.slider("C",0.1,10.0)
        params["C"]=C
        #S=st.sidebar.selectbox("SELSECT KERNEL",["a","b","c"])
        #params["S"]=S
    else:
        clf_name=="Random Forest"
        max_depth=st.sidebar.slider("max_depth",2,15)
        params["max_depth"]=max_depth
        n_estimators=st.sidebar.slider("n_estimators",1,100)
        params["n_estimators"]=n_estimators
    return params
params=add_parameter_ui(classifier_name)    

def get_classifier(clf_name,params):
    if clf_name=="KNN":
         clf=KNeighborsClassifier(n_neighbors=params['k'])
    elif clf_name=="SVM":
        clf=SVC(C=params["C"])
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return clf

clf=get_classifier(classifier_name,params)  
#clasification
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
acc=accuracy_score(y_test,y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")
#plot
PCA=PCA(2)
X_projected=PCA.fit_transform(X)
X1=X_projected[:,0]
X2=X_projected[:,1]
fig=plt.figure()
plt.scatter(X1,X2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel("principle component 1")
plt.ylabel("principle component 2")
plt.colorbar()
st.pyplot(fig)


   