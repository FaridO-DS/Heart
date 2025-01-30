import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Prédiction maladie cardiaque")

st.sidebar.title("Sommaire")
pages=["1.Exploration données","2.Data visualisation", "3.Réduction de dimension", "4.Machine Learning", "5.Prédiction"]
page=st.sidebar.radio("Aller vers", pages)

df = pd.read_csv("heart.csv")

if page == pages[0] : 
    st.write("1.Exploration des données")
    if st.checkbox("Afficher les 5 premières lignes du dataframe",False):
        st.dataframe(df.head())
    if st.checkbox("Afficher les dimensions du dataframe",False):
        st.write(df.shape)
    st.write("Distribution des valeurs de la target : \n", df['target'].value_counts())
    st.write("Le jeu de données est bien équilibré.")
    if st.checkbox("Afficher une description du dataframe",False):
        st.dataframe(df.describe())
    if st.checkbox("Afficher les N/A du dataframe",False):
        st.dataframe(pd.DataFrame(df.isnull().any()))

df['class_ages']=pd.qcut(df['age'],[0,0.25,0.5,0.75,1],labels=["29-48","48-56","56-61","61-77"])
X=df.drop('target',axis=1)
y=df['target']
X1= X.drop('class_ages',axis=1)
import numpy as np

if page == pages[1] :
    st.write("2.Data visualisation")
    fig0=plt.figure(figsize=(6,6))
    sns.displot(data=df['age'],kde=True)
    plt.title("Distribution de l'âge");
    plt.xlabel("Âge");
    plt.ylabel("Fréquence")
    st.pyplot(fig0)
    fig1=plt.figure(figsize=(6,6))
    sns.lineplot(data=df,x='class_ages',y='chol',hue='sex')
    st.pyplot(fig1)
    st.write("On note que le taux de cholestérol pour les femmes est significativement plus élevé à partir de 48 ans et se stabilise.")
    st.write("Quant aux hommes,le taux de cholestérol est plus élevé à partir de 56 ans et augmente avec l'âge.")
    fig2=plt.figure(figsize=(6,6))
    sns.lineplot(data=df,x='class_ages',y='thalach',hue='sex')
    st.pyplot(fig2)
    fig3=plt.figure(figsize=(6,6))
    sns.lineplot(data=df,x='class_ages',y='trestbps',hue='sex')
    st.pyplot(fig3)
    fig4=plt.figure()
    sns.heatmap(data=X1.corr(),annot=True,cmap='coolwarm',fmt='.1f')
    st.pyplot(fig4)    

from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression, RFE, RFECV,f_classif
sel1 = SelectKBest(f_classif,k=7)
sel2 = SelectKBest(chi2,k=7)
sel3 = SelectKBest(f_regression,k=7)
X1_new = sel1.fit_transform(X1,y)
X2_new = sel2.fit_transform(X1,y)
X3_new = sel3.fit_transform(X1,y)

if page == pages[2] :
    st.write("3.Réduction de dimension, feature selection")
    st.write("X shape :", X1.shape)
    st.write("X_new shape :", X1_new.shape)
    st.write("On teste 3 méthodes de feature selection : \n - f_classif \n - chi2 \n - f_regression")
    st.write("Les variables sélectionnées par méthode sont : \n")
    st.dataframe(pd.DataFrame({"f_classif":sel1.get_support()*sel1.feature_names_in_,
                                "chi2": sel2.get_support()*sel2.feature_names_in_,
                                "f_regression":sel3.get_support()*sel3.feature_names_in_})                        
                )
    st.write("On choisit la méthode f_classif avec 7 variables pertinentes sur 13.")
    st.write(sel1.get_feature_names_out())
    
    scores = -np.log10(sel1.pvalues_)
    scores /= scores.max()
    
    fig=plt.figure()
    plt.clf()
    plt.bar(np.array(X1.columns), scores, width=0.2)
    plt.title("Feature univariate score")
    plt.xlabel("Feature name")
    plt.xticks(rotation=45)
    plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
    st.pyplot(fig)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X1)
X_new_scaled = scaler.fit_transform(X1_new)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
rf = RandomForestClassifier(n_jobs=-1)
knn = KNeighborsClassifier(n_jobs=-1)
dt = DecisionTreeClassifier(max_depth=4)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)
X_ntrain,X_ntest,y_ntrain,y_ntest = train_test_split(X_new_scaled,y,test_size=0.2,random_state=42)

rf.fit(X_ntrain,y_ntrain)
knn.fit(X_ntrain,y_ntrain)
dt.fit(X_ntrain,y_ntrain)

from sklearn.metrics import confusion_matrix, classification_report
y_pred = rf.predict(X_ntest)
prob = rf.predict_proba(X_ntest)

# Courbe ROC-AUC
from sklearn.metrics import roc_curve, auc
fpr,tpr,seuils = roc_curve(y_test,prob[:,1],pos_label =1)
roc_auc = auc(fpr,tpr)
data = pd.DataFrame([rf.score(X_ntest,y_ntest),
                             knn.score(X_ntest,y_ntest),  
                            dt.score(X_ntest,y_ntest)],index=["RF","KNN","DT"],columns=["Score"])


if page == pages[3] :
    st.write("4.Machine Learning")  
    st.dataframe(pd.DataFrame(data.round(3)))
    st.write("On garde le modèle RF.")
    st.write('Matrice de confusion :\n')
    st.dataframe(pd.crosstab(y_ntest,y_pred))
    print('-'*100)
    st.write('Classification report :\n') 
    st.dataframe(pd.DataFrame(classification_report(y_ntest,y_pred,output_dict=True)).round(2))
    
    st.write("ROC-AUC :",roc_auc.round(2))
    fig5=plt.figure()
    plt.plot(fpr,tpr,c="orange",label="Modèle clf(auc = 0.72)");
    plt.xlim(0,1);
    plt.xlabel("Taux de faux positifs");
    plt.plot(fpr,fpr,c="blue", linestyle ='--', label="Aléatoire (auc = 0.99)");
    plt.ylim(0,1.05);
    plt.ylabel('Taux de vrais positifs');
    plt.title("Courbe ROC");
    plt.legend(loc="lower right");
    st.pyplot(fig5)

if page == pages[4] :
    st.write("5.Prédiction")
    st.write("Enter the values requested below :")
    val1=st.selectbox("Chest pain (cp) value",[0,1,2,3])
    ''' Value 0: typical angina,
        Value 1: atypical angina,
        Value 2: non-anginal pain,
        Value 3: asymptomatic'''
    val2=st.number_input("Maximum heart achieved (thalach) value")
    val3=st.selectbox("Exercise-induced angina (exang) value",[0,1])
    ''' Value 1 = yes,
        Value 0 = no'''
    val4=st.number_input("Stress test depression induced by exercise relative to rest (oldpeak) value")
    val5=st.selectbox("The slope of the peak exercise ST segment (slope) value",[0,1,2])
    ''' Value 0: upsloping,
        Value 1: flat
        Value 2: downsloping'''
    val6=st.selectbox("Number of major vessels (ca) value",[0,1,2,3])
    '''Number of major vessels (0–3) colored by fluoroscopy'''
    val7=st.selectbox("Thallium heart rate (thal) value",[0,1,2])
    '''Value 0 = normal, Value 1 = fixed defect,
        Value 2 = reversible defect'''
    
    X_submitted = np.array([val1,val2,val3,val4,val5,val6,val7])
    pred = rf.predict(X_submitted.reshape(1,-1))
    st.write("Heart disease prediction :",pred)
