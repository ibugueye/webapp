import streamlit as st
import requests
import matplotlib.pyplot as plt

st.title('Consultation de Prédiction de Crédit Client')

# Récupération de la liste des identifiants SK_ID_CURR depuis l'API Flask
response = requests.get("https://flask-deploement.onrender.com/get_ids")
if response.status_code == 200:
    ids = response.json()
    sk_id_curr = st.selectbox('Choisissez SK_ID_CURR pour obtenir la prédiction, la probabilité, et le nombre d’enfants:', ids)
else:
    st.error("Impossible de charger la liste des identifiants SK_ID_CURR.")
    ids = []
    sk_id_curr = None

if st.button('Obtenir la prédiction') and sk_id_curr:
    data = {'SK_ID_CURR': sk_id_curr}
    
    response = requests.post("https://flask-deploement.onrender.com/prediction", json=data)
    
    if response.status_code == 200:
        prediction_data = response.json()
        
        # Afficher les informations de prédiction
        st.write(f"Prédiction : {prediction_data['prediction']}, Probabilité : {prediction_data['probability']}")
        st.write(f"Décision basée sur le seuil de 0.39 : {prediction_data['decision']}")
        
        # Afficher le nombre d'enfants du client et la moyenne
        st.write(f"Nombre d'enfants du client : {prediction_data['client_children']}")
        st.write(f"Moyenne du nombre d'enfants parmi tous les clients : {prediction_data['mean_children']:.2f}")
        st.write(prediction_data['comparison'])
        
        # Création d'une visualisation
        fig, ax = plt.subplots()
        ax.barh(["Client", "Moyenne"], [prediction_data['client_children'], prediction_data['mean_children']], color=['blue', 'orange'])
        ax.set_xlabel('Nombre d\'enfants')
        st.pyplot(fig)
    else:
        st.error("Une erreur s'est produite lors de l'obtention de la prédiction.")
