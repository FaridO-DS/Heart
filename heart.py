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
    if st.checkbox("Afficher les explications des variables",False):
        '''Order   Feature	       Description	                   Feature Value Range'''
        
        '''1: Age, Age in years, 29 to 77'''

        '''2: Sex, Gender, Value 1 = male, Value 0 = female'''
               
        '''3: Cp, Chest pain type, 0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic'''
                                                                
        '''4: Trestbps, Resting blood pressure (mm/Hg), 94 to 200'''

        '''5: Chol, Serum cholesterol in mg/dL, 126 to 564'''

        '''6: Fbs, Fasting blood sugar > 120 mg/dL, 1 = true, 0 = false'''
                                                                        
        '''7: Restecg, Resting electrocardiographic results, 0: Normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy'''
                       	                                                
        '''8: Thalach, Maximum heart rate achieved, 71 to 202'''

        '''9: Exang, Exercise-induced angina, 1 = yes, 0 = no'''
                                                                    
        '''10: Oldpeak, Stress test depression induced by exercise relative to rest, 0 to 6.2'''
                                    	        
        '''11: Slope, Slope of the peak exercise ST segment, 0: upsloping, 1: flat, 2: downsloping'''
                                       	                            
        '''12: Ca, Number of major vessels, 0–3 colored by fluoroscopy'''

        '''13: Thal, Thallium heart rate, 0 = normal; 1: fixed defect, 2: reversible defect'''
                                                                        
        '''14: Target, Diagnosis of heart disease, 0 = no disease, 1: disease'''
                                                                           
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
    st.write("Quelques représentations graphiques.")
    fig1=plt.figure(figsize=(6,6))
    sns.lineplot(data=df,x='class_ages',y='chol',hue='sex')
    plt.title("Distribution du taux de choléstérol en fonction de l'âge et par sexe")
    st.pyplot(fig1)
    st.write("On note que le taux de cholestérol pour les femmes augmente rapidement entre le premier et le deuxième quartile et se stabilise après.")
    st.write("Quant aux hommes,le taux de cholestérol augmente lentement entre le premier et le troisième quartile et plus rapidement après.")
    fig2=plt.figure(figsize=(6,6))
    sns.lineplot(data=df,x='class_ages',y='thalach',hue='sex')
    plt.title("Fréquence cardiaque maximale en fonction de l'âge et par sexe")
    st.pyplot(fig2)
    st.write('On remarque que la fréquence cardiaque diminue plus lentement entre le deuxième et le troisième quartile, et ce pour les deux sexes.')
    fig3=plt.figure(figsize=(6,6))
    sns.lineplot(data=df,x='class_ages',y='trestbps',hue='sex')
    plt.title("Tension artérielle au repos en fonction de l'âge et par sexe") 
    st.pyplot(fig3)
    st.write('On remarque que la tension artérielle augmente rapidement pour les femmes entre le premier et le deuxième quartile, puis moins après.')
    st.write("Quant aux hommes,la tension artérielle augmente plus rapidement entre le deuxième et le troisième quartile, puis décroit après.")
    fig4=plt.figure()
    sns.heatmap(data=X1.corr(),annot=True,cmap='coolwarm',fmt='.1f')
    plt.title("Matrice de corrélation de Pearson entre les variables explicatives.")
    st.pyplot(fig4)    

from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression, RFE, RFECV,f_classif
sel1 = SelectKBest(f_classif,k=5)
sel2 = SelectKBest(chi2,k=5)
sel3 = SelectKBest(f_regression,k=5)
X1_new = sel1.fit_transform(X1,y)
X2_new = sel2.fit_transform(X1,y)
X3_new = sel3.fit_transform(X1,y)

if page == pages[2] :
    st.write("3.Réduction de dimension, feature selection")
    st.write("On teste 3 méthodes de feature selection : \n - f_classif \n - chi2 \n - f_regression")
    st.write("Les variables sélectionnées par méthode sont : \n")
    st.dataframe(pd.DataFrame({"f_classif":sel1.get_support()*sel1.feature_names_in_,
                                "chi2": sel2.get_support()*sel2.feature_names_in_,
                                "f_regression":sel3.get_support()*sel3.feature_names_in_})                        
                )
    st.write("On choisit la méthode f_classif mais les 3 méthodes sélectionnent les mêmes variables pertinentes.")
    st.write(sel1.get_feature_names_out())
    st.write("X shape :", X1.shape)
    st.write("X_new shape :", X1_new.shape)

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
    st.write("On compare la performance des 3 méthodes sélectionnées.")
    st.dataframe(pd.DataFrame(data.round(3)))
    st.write("La méthode RF donne de meilleurs résultats.")
    st.write('Matrice de confusion :\n')
    st.dataframe(pd.crosstab(y_ntest,y_pred))
    print('-'*100)
    st.write('Classification report :\n') 
    st.dataframe(pd.DataFrame(classification_report(y_ntest,y_pred,output_dict=True)).round(2))
    
    st.write("ROC-AUC :",roc_auc.round(2))
    fig5=plt.figure()
    plt.plot(fpr,tpr,c="orange",label="Modèle clf(auc = 0.99)");
    plt.xlim(0,1);
    plt.xlabel("Taux de faux positifs");
    plt.plot(fpr,fpr,c="blue", linestyle ='--', label="Aléatoire (auc = 0.72)");
    plt.ylim(0,1.05);
    plt.ylabel('Taux de vrais positifs');
    plt.title("Courbe ROC");
    plt.legend(loc="lower right");
    st.pyplot(fig5)

if page == pages[4] :
    st.write("5.Prédiction")
    st.write("Entrez les valeurs requises ci-dessous :")
    val1=st.selectbox("Douleur thoracique (cp)",[0,1,2,3])
    ''' Value 0: angine de poitrine typique,
        Value 1: angine de poitrine atypique,
        Value 2: absence de douleur thoracique,
        Value 3: asymptomatique'''
    val2=st.number_input("Fréquence cardiaque maximale (thalach)")
    val3=st.selectbox("Exercise-induced angina (exang) value",[0,1])
    ''' Value 1 = yes,
        Value 0 = no'''
    val4=st.number_input("Stress test depression induced by exercise relative to rest (oldpeak) value")
    ''' 0 to 6.2'''
    val5=st.selectbox("Number of major vessels (ca) value",[0,1,2,3])
    '''Number of major vessels (0–3) colored by fluoroscopy'''
    
    
    X_submitted = np.array([val1,val2,val3,val4,val5])
    pred = rf.predict(X_submitted.reshape(1,-1))
    st.write("Heart disease prediction :",pred)
    '''0 : no disease, 1 : disease'''