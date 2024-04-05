import streamlit as st
import requests
import matplotlib.pyplot as plt

st.header("Informations sur le client")
     
# Récupération de la liste des identifiants SK_ID_CURR depuis l'API Flask
response = requests.get("https://flask-deploement.onrender.com/get_ids")
if response.status_code == 200:
     ids = response.json()
     sk_id_curr = st.sidebar.selectbox('Choisissez SK_ID_CURR pour obtenir des informations détaillées et une prédiction:', ids)
else:
     st.error("Impossible de charger la liste des identifiants SK_ID_CURR.")
     ids = []
     sk_id_curr = None

if st.sidebar.button('Obtenir la prédiction') and sk_id_curr:
     data = {'SK_ID_CURR': sk_id_curr}
     response = requests.post("https://flask-deploement.onrender.com/prediction", json=data)
        
     if response.status_code == 200:
          prediction_data = response.json()
            
            # Displaying the results in two columns
            
          col1, col2,col3= st.columns([1,1,4])
            
            
            # Affichage des informations et des prédictions
          st.write(f"Prédiction : {prediction_data['prediction']}, Probabilité : {prediction_data['probability']:.2f}")
          st.write(f"### Décision basée sur le seuil de 0.44 : {prediction_data['decision']}")
            
            
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
