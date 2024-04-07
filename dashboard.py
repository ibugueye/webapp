import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
import joblib
import requests
import numpy as np

# Chargement du modèle et des données
model_path = 'best_model.joblib'
pipeline = joblib.load(model_path)
df = pd.read_csv('df_final.csv')  # Assurez-vous que ce chemin est correct
X = df.drop(columns=['TARGET'])
y = df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier_model = pipeline.named_steps['classifier']
explainer = shap.Explainer(classifier_model, X_train)

# Interface utilisateur
st.title("Prédiction de Non-Paiement de Prêt")

# Partie 1 : Explications globales avec SHAP
st.header("Importance Globale des Caractéristiques")
fig, ax = plt.subplots()
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
st.pyplot(fig)

# Récupération de l'identifiant client et affichage des prédictions
st.header("Informations sur le client")

response = requests.get("https://flask-deploement.onrender.com/get_ids")
if response.status_code == 200:
    ids = response.json()
    sk_id_curr = st.sidebar.selectbox('Choisissez SK_ID_CURR pour obtenir des informations détaillées et une prédiction:', ids)
    if st.sidebar.button('Obtenir la prédiction'):
        data = {'SK_ID_CURR': sk_id_curr}
        response = requests.post("https://flask-deploement.onrender.com/prediction", json=data)
        if response.status_code == 200:
            prediction_data = response.json()

            # Partie 2 : Affichage des informations de prédiction
            decision_threshold = 0.44
            probability = float(prediction_data['probability'])
            loan_decision = "Prêt Accordé" if probability > decision_threshold else "Prêt Refusé"
            st.write(f"Prédiction : {prediction_data['prediction']}, Probabilité : {probability:.2f}")
            st.write(f"Décision basée sur le seuil de {decision_threshold} : ")

            if loan_decision == "</h1> Prêt Accordé </h1>":
                st.markdown(f"<h4 style='color:green;'>{loan_decision}</h4>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h4 style='color:red;'>{loan_decision}</h4>", unsafe_allow_html=True)

            # Nouvelle Partie : Comparaison avec les moyennes
            if response.status_code == 200:
                prediction_data = response.json()
            
            # Displaying the results in two columns
            
        col1, col2,col3= st.columns([1,1,4])
            
            
            # Affichage des informations et des prédictions
        #st.write(f"Prédiction : {prediction_data['prediction']}, Probabilité : {prediction_data['probability']:.2f}")
        #st.write(f"### Décision basée sur le seuil de 0.44 : {prediction_data['decision']}")
            
            
        with col1:
                st.header("Client")
                st.write(f"Enfants : {prediction_data['client_children']}")
                st.write(f"Revenu: {prediction_data['client_income']}")
                st.write(f"Crédit  : {prediction_data['client_credit']}")
                st.write(f"Annuité : {prediction_data['client_annuity']}")
                
                
                
               
        with col2:
                st.header("Moyen")
            
                st.write(f"Enfants : {prediction_data['mean_children']:.2f}")
                st.write(f"Revenu   : {prediction_data['mean_income']:.2f}")
                st.write(f"Crédit: {prediction_data['mean_credit']:.2f}")
                st.write(f"Annuité  : {prediction_data['mean_annuity']:.2f}")
                
                
        with col3:
                st.write("Comparaiso, des caracteristuqes clients avec la moyenne")
                # Création de la visualisation
                fig, ax = plt.subplots()
                labels = ["Enfants", "Revenu", "Crédit", "Annuité"]
                client_values = [prediction_data['client_children'], prediction_data['client_income'], prediction_data['client_credit'], prediction_data['client_annuity']]
                mean_values = [prediction_data['mean_children'], prediction_data['mean_income'], prediction_data['mean_credit'], prediction_data['mean_annuity']]
                
                x = range(len(labels))  # les labels de l'axe x
                
                ax.bar(x, client_values, width=0.4, label='Client', align='center')
                ax.bar(x, mean_values, width=0.4, label='Moyenne', align='edge')
                
                ax.set_xlabel('Caractéristiques')
                ax.set_ylabel('Valeurs')
                ax.set_title('Comparaison des caractéristiques du client avec les moyennes')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()
                
                st.pyplot(fig)
    else:
            st.error("Une erreur s'est produite lors de l'obtention de la prédiction.")
           

 
        # Partie 3 : Explications SHAP pour l'observation sélectionnée
    st.header("Explications SHAP locale")
    if sk_id_curr in df['SK_ID_CURR'].values:
        observation_to_explain = X[df['SK_ID_CURR'] == int(sk_id_curr)]
        shap_values_observation = explainer(observation_to_explain)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values_observation[0], max_display=10)
        st.pyplot(fig)
    else:
        median_observation = X_train.median().to_frame().T
        shap_values_median = explainer(median_observation)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values_median[0], max_display=10)
        st.pyplot(fig)
     
else:
    st.error("Impossible de charger la liste des identifiants SK_ID_CURR.")
