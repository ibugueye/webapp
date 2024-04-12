 
            
    

# URL of the Flask API
API_URL = 'https://flask-deploement.onrender.com/predict'  # Adjust this to the URL of your Flask API

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
import joblib  # Pour charger le modèle
    
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
st.title("Prédiction de Non-Paiement de Prêt ")
        
            # Explications globales avec SHAP
st.header("Importance Globale des Caractéristiques")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_train, plot_type="bar")
st.pyplot(fig)
        
        
        
        # Création d'un dictionnaire pour mapper SK_ID_CURR à l'indice du DataFrame
id_to_index = pd.Series(df.index, index=df['SK_ID_CURR']).to_dict()
        
        
        # Sélection de SK_ID_CURR dans Streamlit
selected_sk_id_curr = st.selectbox("# Sélectionnez l'identifiant du client à expliquer", X_test['SK_ID_CURR'].unique())
        
        # Trouver l'indice correspondant dans le DataFrame original
selected_index = id_to_index[selected_sk_id_curr]
        
observation_to_explain = X_test.loc[selected_index:selected_index]
            # Sélection de l'indice de l'observation à expliquer par un utilisateur
         
predicted_class =classifier_model .predict(observation_to_explain)[0]
probability_of_default = classifier_model.predict_proba(observation_to_explain)[0, 1]  # Probabilité de défaut de paiement
        
            # Affichage des résultats
st.write(f"Prédiction  : {'Non-Paiement  Risk' if predicted_class else 'Paiement no Risk '}")
st.write(f"Probabilité de non-paiement: {probability_of_default:.4f}")
        
        # Décision basée sur un seuil spécifique
decision_threshold = 0.428  # Ajustez ce seuil selon vos critères
loan_decision = "Prêt Accordé" if probability_of_default < decision_threshold else "Prêt Refusé"
st.write(f"Décision du prêt basée sur le seuil de probabilité de {decision_threshold}: {loan_decision}")
        
            # Affichage des explications SHAP pour l'observation sélectionnée
st.header(f"Explications SHAP local ")
shap_values_observation = explainer(observation_to_explain)
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values_observation[0], max_display=10)
st.pyplot(fig)
          
          
# Supposons que 'df' est déjà chargé et que 'selected_sk_id_curr' est l'ID du client sélectionné
client_data = df[df['SK_ID_CURR'] == selected_sk_id_curr ]

# Calcul des moyennes
#CNT_CHILDREN,AMT_INCOME_TOTAL,AMT_CREDIT,
mean_children = df['CNT_CHILDREN'].mean()
mean_income = df['AMT_INCOME_TOTAL'].mean()
mean_credit = df['AMT_CREDIT'].mean()
mean_annuity = df['AMT_ANNUITY'].mean()

# Affichage des caractéristiques du client
st.header("Client")
st.write(f"Enfants : {client_data['CNT_CHILDREN'].values[0]}")
st.write(f"Revenu: {client_data['AMT_INCOME_TOTAL'].values[0]}")
st.write(f"Crédit  : {client_data['AMT_CREDIT'].values[0]}")
st.write(f"Annuité : {client_data['AMT_ANNUITY'].values[0]}")

# Visualisation des caractéristiques
fig, axs = plt.subplots(4, 1, figsize=(5, 15))  # Ajustez les dimensions selon vos besoins

# Enfants
axs[0].bar(['Client', 'Moyenne'], [client_data['CNT_CHILDREN'].values[0], mean_children])
axs[0].set_title('Nombre d\'enfants')

# Revenu
axs[1].bar(['Client', 'Moyenne'], [client_data['AMT_INCOME_TOTAL'].values[0], mean_income])
axs[1].set_title('Revenu')

# Crédit
axs[2].bar(['Client', 'Moyenne'], [client_data['AMT_CREDIT'].values[0], mean_credit])
axs[2].set_title('Montant du crédit')

# Annuité
axs[3].bar(['Client', 'Moyenne'], [client_data['AMT_ANNUITY'].values[0], mean_annuity])
axs[3].set_title('Annuité')

plt.tight_layout()
st.pyplot(fig)         