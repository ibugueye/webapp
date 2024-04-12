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

# Préparation des données
df = pd.read_csv('df_final.csv')  # Assurez-vous que ce chemin est correct
X = df.drop(columns=['TARGET'])
y = df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser l'explainer SHAP avec le modèle extrait
explainer = shap.Explainer(classifier_model, X_train)

# Interface Streamlit
st.title("Score de Credit Client")

# Sélection de SK_ID_CURR dans Streamlit
selected_sk_id_curr = st.sidebar.selectbox("Sélectionnez l'identifiant du client à expliquer", X_test['SK_ID_CURR'].unique())

st.write('### Contexte du projet')
st.write("L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.")
st.image("credit_score.png")


# Bouton de prédiction
if st.sidebar.button('Prédire'):
    
           
            
    #  DF est déjà chargé et que 'selected_sk_id_curr' est l'ID du client sélectionné
    client_data = df[df['SK_ID_CURR'] == selected_sk_id_curr ]


    col1, col2,col3= st.columns([1,1,4])

    # Calcul des moyennes
    #CNT_CHILDREN,AMT_INCOME_TOTAL,AMT_CREDIT,
    mean_children = df['CNT_CHILDREN'].mean()
    mean_income = df['AMT_INCOME_TOTAL'].mean()
    mean_credit = df['AMT_CREDIT'].mean()
    mean_annuity = df['AMT_ANNUITY'].mean()

    # Affichage des caractéristiques du client

    
        
    # Supposons que les données du client et les moyennes sont déjà calculées
    client_data = df[df['SK_ID_CURR'] == selected_sk_id_curr]
    mean_children = df['CNT_CHILDREN'].mean()
    mean_income = df['AMT_INCOME_TOTAL'].mean()
    mean_credit = df['AMT_CREDIT'].mean()
    mean_annuity = df['AMT_ANNUITY'].mean()

    cols = st.columns(3)  # Crée trois colonnes

    with cols[0]:  # Première colonne pour les caractéristiques du client
        st.header("Client")
        st.write(f"Enfants : {client_data['CNT_CHILDREN'].values[0]}")
        st.write(f"Revenu: {client_data['AMT_INCOME_TOTAL'].values[0]}")
        st.write(f"Crédit  : {client_data['AMT_CREDIT'].values[0]}")
        st.write(f"Annuité : {client_data['AMT_ANNUITY'].values[0]}")

    with cols[1]:  # Deuxième colonne pour les moyennes
        st.header("Moyennes")
        st.write(f"Enfants Moy. : {mean_children:.2f}")
        st.write(f"Revenu Moy. : {mean_income:.2f}")
        st.write(f"Crédit Moy. : {mean_credit:.2f}")
        st.write(f"Annuité Moy. : {mean_annuity:.2f}")
        
        
        
        with cols[2]:  # Troisième colonne pour les graphiques comparatifs
            st.header("Comparaison ")

            # Préparer les données pour le graphique
            labels = ['Enfants', 'Revenu', 'Crédit', 'Annuité']
            client_values = [
                client_data['CNT_CHILDREN'].values[0],
                client_data['AMT_INCOME_TOTAL'].values[0],
                client_data['AMT_CREDIT'].values[0],
                client_data['AMT_ANNUITY'].values[0]
            ]
            mean_values = [mean_children, mean_income, mean_credit, mean_annuity]

            # Paramètres pour le positionnement des barres
            x = range(len(labels))  # labels pour les axes
            width = 0.4  # Largeur des barres

            fig, ax = plt.subplots()
            ax.bar(x, client_values, width=width, label='Client', color='b', align='center')
            ax.bar([p + width for p in x], mean_values, width=width, label='Moyenne', color='r', align='center')

            ax.set_xlabel('Caractéristiques')
            ax.set_ylabel('Valeurs')
            ax.set_title('Comparaison des caractéristiques du client avec les moyennes')
            ax.set_xticks([p + width / 2 for p in x])  # Positionner les étiquettes au centre entre les barres
            ax.set_xticklabels(labels)
            ax.legend()

            st.pyplot(fig)

    
    
    
    # Trouver l'indice correspondant dans le DataFrame original
    selected_index = pd.Series(df.index, index=df['SK_ID_CURR']).to_dict()[selected_sk_id_curr]
    observation_to_explain = X_test.loc[selected_index:selected_index]

    # Prédiction et probabilité de défaut de paiement
    predicted_class = classifier_model.predict(observation_to_explain)[0]
    probability_of_default = classifier_model.predict_proba(observation_to_explain)[0, 1]

    # Affichage des résultats
    st.sidebar.write(f"Prédiction  : {'Non-Paiement  Risk' if predicted_class else 'Paiement no Risk '}")
    st.sidebar.write(f"Probabilité de non-paiement: {probability_of_default:.4f}")

    # Décision basée sur un seuil spécifique
    decision_threshold = 0.428
    loan_decision = "Prêt Accordé" if probability_of_default < decision_threshold else "Prêt Refusé"
    st.sidebar.write(f" Le seuil de probabilité de {decision_threshold}")
    
    
    st.write("### la decision prise pour ce client :" )
    
    # Utiliser du HTML pour le style
    decision_color = "green" if loan_decision == "Prêt Accordé" else "red"
    st.markdown(f"<span style='color: {decision_color}; font-weight: bold;'>Décision du prêt basée sur le seuil de probabilité de {decision_threshold}: {loan_decision}</span>", unsafe_allow_html=True)

    # Affichage des explications SHAP globales pour l'ensemble d'entraînement
    st.header("Explications SHAP globales")
    fig, ax = plt.subplots()
    shap.summary_plot(explainer.shap_values(X_train), X_train, plot_type="bar")
    st.pyplot(fig)
    
    
 
            
            # Création d'un dictionnaire pour mapper SK_ID_CURR à l'indice du DataFrame
    id_to_index = pd.Series(df.index, index=df['SK_ID_CURR']).to_dict()
            
            

            # Trouver l'indice correspondant dans le DataFrame original
    selected_index = id_to_index[selected_sk_id_curr]
            
    observation_to_explain = X_test.loc[selected_index:selected_index]
                # Sélection de l'indice de l'observation à expliquer par un utilisateur
            
    predicted_class =classifier_model .predict(observation_to_explain)[0]
    probability_of_default = classifier_model.predict_proba(observation_to_explain)[0, 1]  # Probabilité de défaut de paiement
            
   
            
                # Affichage des explications SHAP pour l'observation sélectionnée
    st.header(f"Explications SHAP local ")
    shap_values_observation = explainer(observation_to_explain)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values_observation[0], max_display=10)
    st.pyplot(fig)
     