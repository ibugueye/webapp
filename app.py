# Importations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import codecs
import requests
import shap
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import sweetviz as sv
import streamlit as st
import joblib  # Pour charger le modèle



# Configuration initiale de Streamlit
st.sidebar.title("Sommaire")
pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "EDA automatique",
         "MLflow experience", "Modèles Comparés", "Modélisation", "Choix Classifiers", "Prédictions",
         "Explication SHAP", "Outil de Décision"]
page = st.sidebar.radio("Aller vers la page :", pages)

# Chargement des données
#df= pd.read_csv("data/application_train.csv")
df = pd.read_csv("data/df_final.csv")
df["CODE_GENDER"].fillna(0, inplace=True)

st.sidebar.title("Sommaire")
pages = [
    "Contexte du projet", "Exploration des données", "Analyse de données",
    "EDA automatique", "MLflow experience", "Modèles Comparés", 
    "Modélisation", "Choix Classifiers", "Prédictions", "Explication SHAP", 
    "Outil de Décision"
]
 

# Fonctions pour gérer le contenu de chaque page
def contexte_projet():
    st.write('### Contexte du projet')
    st.write("L’entreprise souhaite mettre en œuvre un outil de “scoring crédit”...")
    # Ajoutez votre logique pour cette page ici

def exploration_donnees():
    st.write("### Exploration des données ")
    
  
     #Exemple: Charger et afficher les premières lignes de votre DataFrame
    # df = pd.read_csv("chemin_vers_votre_fichier.csv")
    st.dataframe(df.head())
    st.write("Dimensions du dataFrame : ")
    st.write("Nombre de lignes : " ,  df.shape[0])
    st.write("Nombre de colonnes : ",df.shape[1])
    st.write("Missing values")
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        return mis_val_table_ren_columns
     
def analyse_donnees():
    st.write("### Analyse de données")
    
  
    
    # Calcul des valeurs manquantes
    missing_values = missing_values_table(df)
    
    # Afficher le tableau des valeurs manquantes
    st.write("Tableau des Valeurs Manquantes :")
    st.table(missing_values)
    
    if st.checkbox("Afficher les valeurs manquantes"):
        st.write("Nombre de missing value : ",df.isna().sum().sum())
    if st.checkbox("Afficher les doublons"):
        st.write("Nombre de doublons ",df.duplicated().sum())
       

# Ajoutez d'autres fonctions pour les pages restantes de manière similaire

# Dictionnaire des fonctions de page
page_functions = {
    "Contexte du projet": contexte_projet,
    "Exploration des données": exploration_donnees,
    "Analyse de données": analyse_donnees,
    # Continuez à mapper vos pages aux fonctions correspondantes ici
}

# Exécuter la fonction de page sélectionnée
if page in page_functions:
    page_functions[page]()  # Appelle la fonction sans passer df si pas utilisé globalement
else:
    st.write("Erreur: Page non trouvée.")  # Message d'erreur si la page n'est pas dans le dictionnaire

# Exécution de la fonction de page sélectionnée
 

# Définitions des fonctions pour chaque page
def contexte_projet(df, df_train):
    st.write('### Contexte du projet')
    st.write("L’entreprise souhaite mettre en œuvre un outil de “scoring crédit”...")
    st.image("credit_score.png")

def exploration_donnees(df, df_train):
    st.write("### Exploration des données ")
    st.dataframe(df.head())
    # Ajoutez plus d'analyses exploratoires ici

def analyse_donnees(df, df_train):
    st.title(" Analyse de Données")
    # Insérez votre logique d'analyse des données ici

def eda_automatique(df, df_train):
    
    st.write("## Analyse exploration automatique avec Sweetviz")   
    def st_display_sweetviz(report_html,width=1000,height=500):
        report_file = codecs.open(report_html,'r')
        page = report_file.read()
        components.html(page,width=width,height=height,scrolling=True) # type: ignore

def mlflow_experience(df, df_train):
    st.title("Résultats MLflow dans Streamlit")
    # Afficher les résultats MLflow

def modeles_comparés(df, df_train):
    st.title('Comparaison des Classificateurs')
    # Comparaison des modèles de classification

def modelisation(df, df_train):
    st.title("Modélisation")
    # Logique de modélisation

def choix_classifiers(df, df_train):
    st.title("Choix Classifiers")
    # Choix et évaluation des classificateurs

def predictions(df, df_train):
    st.title(" Prédiction de Défaut de Paiement")
    # Logique pour effectuer des prédictions

def explication_shap(df, df_train):
    st.title("L’interprétabilité globale et locale")
    # Explication SHAP des prédictions du modèle

def outil_decision(df, df_train):
    st.title("Outil de Décision")
    

    # Chargement du modèle sérialisé
    model_path = 'best_model.joblib'
    pipeline = joblib.load(model_path)

    # Extraire le modèle de classification du pipeline
     
    classifier_model = pipeline.named_steps['classifier']

    # Préparation des données (comme précédemment)
    df = pd.read_csv('df_final.csv')  # Remplacez par votre chemin de fichier correct
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialiser l'explainer SHAP avec le modèle extrait
    explainer = shap.Explainer(classifier_model, X_train)
    # Calcul des valeurs SHAP pour l'ensemble d'entraînement (pour les explications globales)
    shap_values = explainer(X_train)

    # Interface Streamlit
    st.title("Prédiction de Non-Paiement de Prêt avec le Meilleur Modèle")
t
    # Explications globales avec SHAP
    st.header("Importance Globale des Caractéristiques")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    st.pyplot(fig)

    # Sélection de l'indice de l'observation à expliquer par un utilisateur
    index_to_explain = st.slide("Sélectionnez l'indice de l'observation à expliquer", 0, len(X_test)-1, 0)

    # Extraction de l'observation spécifique à expliquer
    observation_to_explain = X_test.iloc[index_to_explain:index_to_explain+1]

