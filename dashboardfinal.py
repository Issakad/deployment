#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px


from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression




from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score, roc_curve, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_curve,PrecisionRecallDisplay
from sklearn.metrics import precision_score,recall_score, accuracy_score 

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

st.set_option('deprecation.showPyplotGlobalUse', False)



from PIL import Image
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap
from sklearn import model_selection










    






import shap
#from streamlit_shap import st_shap
shap.initjs()

#Import du modèle entrainé et des données pré-traitées

with open('lgbm_model.pickle', 'rb') as f:
    #lgbm_model = pickle.load(f)
    lgbm_model =pd.read_pickle(f)
with open('X_test_id.pickle', 'rb') as f:
    X_test_id = pickle.load(f)
    #X_test_id = pd.read_pickle(f)
    


# Prédiction du modèle
y_pred = lgbm_model.predict(X_test_id)

#lgbm_model = LGBMClassifier(max_depth=7, learning_rate=0.1, random_state=42)


# Préparation des données pour l'interface utilisateur
#data_for_display = pd.concat([pd.DataFrame(X_test_id.index, columns=['SK_ID_CURR']), pd.DataFrame(y_pred, columns=['Predicted_TARGET'])], axis=1)

score = lgbm_model.predict_proba(X_test_id)[:,1]
#data_for_display['Score'] = score

#X_with_score = pd.merge(X_test_id, data_for_display, on='SK_ID_CURR')



data_for_display =X_test_id.copy()
data_for_display.fillna(data_for_display.mean(), inplace=True)
data_for_display['Predicted_TARGET']=y_pred
data_for_display['Score']=score
data_for_display=data_for_display.reset_index()
data_for_display=data_for_display.rename(columns={"index": "SK_ID_CURR"})
#data_for_display=data_for_display.set_index('SK_ID_CURR')
X_with_score=data_for_display.sample(100)

st.title("Dashboard pour l'octroi de crédits")
st.subheader("Auteur: Issaka Dialga")
# Glossaire des features utilisées:


import streamlit as st

# Définir le glossaire
glossary = {
    "CREDIT_INCOME_PERCENT": "Pourcentage du montant du crédit par rapport au revenu d'un client",
    "ANNUITY_INCOME_PERCENT": "Pourcentage de la rente de prêt par rapport au revenu d'un client",
    "CREDIT_TERM": "Durée du paiement en mois",
    "DAYS_EMPLOYED_PERCENT": "Pourcentage des jours employés par rapport à l'âge du client",
    "NAME_INCOME_TYPE_Businessman": "Revenu de provenant des activités d'affaires",
    "NAME_EDUCATION_TYPE_Higher education":"Niveau d'éducation supérieur",
    "NAME_EDUCATION_TYPE_Incomplete higher": "Education supérieure non achevée",
    "NAME_EDUCATION_TYPE_Lower secondary": "Niveau d'éducation collège",
    "ANNUITY_INCOME_PERCENT": "Montant du remboursement du prêt (annuitées) en % du revenu du client",
    "EXT_SOURCE_1": "Note du client par rapport à son historique de prêts auprès d'autres banques (1)", 
    "EXT_SOURCE_2":"Note du client par rapport à son historique de prêts auprès d'autres banques (2)",
    "EXT_SOURCE_3": "Note du client par rapport à son historique de prêts auprès d'autres banques (3)",
    "DAYS_EMPLOYED_ANOM":"Nombre  jours entre le début du contrat d'emploi du client et la date de sa demande de crédit"
}

# Ajouter une option pour afficher/cacher le glossaire
glossary_visibility = st.sidebar.checkbox("Afficher/Cacher le glossaire")

# Si l'option "Afficher/Cacher" est cochée, afficher le glossaire
if glossary_visibility:
    # Afficher le glossaire en liste déroulante
    selected_term = st.sidebar.selectbox("Sélectionnez un terme :", list(glossary.keys()))

    # Afficher la définition du terme sélectionné
    if selected_term in glossary:
        st.sidebar.write(f"## {selected_term}")
        st.sidebar.write(glossary[selected_term])
    else:
        st.sidebar.write("Sélectionnez un terme dans la liste déroulante.")
else:
    st.sidebar.write("Le glossaire est caché. Cochez la case 'Afficher/Cacher le glossaire' pour le voir.")

    
# Sélection d'un client via un système de filtre
st.write("Sélection d'un client :")
selected_client = st.selectbox('Sélectionnez un client', X_with_score['SK_ID_CURR'].unique())
selected_client_data = X_with_score[X_with_score['SK_ID_CURR'] == selected_client]
st.write(selected_client_data)


# Affichage des informations descriptives relatives à un client
st.subheader("Score, Interprétation et Décision de la banque sur le client sélectionné :")
client_score = float(X_with_score[X_with_score['SK_ID_CURR'] == selected_client]['Score'])
st.write(f"Score du client sélectionné : {client_score}")
if client_score < 0.1:
    st.write("Le client a de bonnes chances de rembourser son prêt.")
elif client_score < 0.2:
    st.write("Le client a des chances moyennes de rembourser son prêt.")
else:
    st.write("Le client a très peu de chances de rembourser son prêt." )
st.write("Décision de la banque :")
if client_score < 0.1:
    st.write("Prêt accordé !👋✍️🤝") 
else:
    st.write("Crédit refusé.😕😕😕")


# Afficher les feature importances
st.subheader("Caractéristiques expliquant le score du client:")
# Sélectionner l'identifiant du client
#client_id = st.selectbox('Sélectionnez un client', list(X_with_score[''SK_ID_CURR'].unique()))
client_id  = st.selectbox('Sélectionnez un client', X_with_score['SK_ID_CURR'].unique(), key='client_selector')
# Obtenir les données du client sélectionné
client_data = X_with_score.loc[X_with_score['SK_ID_CURR'] == client_id]

# Sélectionner les caractéristiques du client
client_features = client_data.drop([ 'Score'], axis=1)

# Obtenir les importances des caractéristiques pour le modèle
importances_df = pd.DataFrame({
    'feature': lgbm_model.feature_name_,
    'importance': lgbm_model.feature_importances_
})

# Sélectionner les importances des caractéristiques correspondant aux caractéristiques du client
client_importances = importances_df[importances_df['feature'].isin(client_features.columns)]

# Afficher les importances des caractéristiques sous forme de graphique à barres
importances_chart = alt.Chart(client_importances).mark_bar().encode(
    x=alt.X('importance:Q', title='Importance'),
    y=alt.Y('feature:N', sort='-x', title='Feature'),
    color=alt.Color('importance:Q', legend=None)
).properties(
    width=800,
    height=400,
    title=f'Importance des caractéristiques pour le client {client_id}'
)

# Afficher le graphique
st.write(importances_chart)


# Afficher les importances des caractéristiques sous forme de graphique à barres
importances_df = pd.DataFrame({
    'feature': lgbm_model.feature_name_,
    'importance': lgbm_model.feature_importances_
})
importances_chart = alt.Chart(importances_df).mark_bar().encode(
    x=alt.X('importance:Q', title='Importance'),
    y=alt.Y('feature:N', sort='-x', title='Feature'),
    color=alt.Color('importance:Q', legend=None)
).properties(
    width=800,
    height=400,
    title='Importance des caractéristiques pour tous les clients'
)
st.write(importances_chart)







st.subheader('Comparaison des informations du client sélectionné avec les clients similaires ou tous les clients')

#Features
features=['SK_ID_CURR','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_EMPLOYED_ANOM', 'AMT_REQ_CREDIT_BUREAU_YEAR', 
          'NAME_INCOME_TYPE_Businessman', 'NAME_EDUCATION_TYPE_Higher education', 'NAME_EDUCATION_TYPE_Incomplete higher', 
          'NAME_EDUCATION_TYPE_Lower secondary', 'ANNUITY_INCOME_PERCENT', 'DAYS_EMPLOYED_PERCENT', 'CREDIT_TERM']
X = X_with_score[features]


# Comparaison des informations descriptives relatives à un client à l'ensemble des clients ou à un groupe de clients similaires
#st.write("Comparaison avec l'ensemble des clients ou un groupe de clients similaires :")
#comparison_choice = st.selectbox("Sélectionnez l'ensemble de clients à comparer", ['Tous les
comparison_choice = st.selectbox("Sélectionnez l'ensemble de clients à comparer", ['Tous les clients', 'Clients similaires'])
#if comparison_choice == "Clients similaires":
    #st.write("Sélectionnez les critères de similitude :")
sim_criteria = st.multiselect("Critères", ['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3','DAYS_EMPLOYED_ANOM','AMT_REQ_CREDIT_BUREAU_YEAR',
         'NAME_INCOME_TYPE_Businessman','NAME_EDUCATION_TYPE_Higher education','NAME_EDUCATION_TYPE_Incomplete higher',
         'NAME_EDUCATION_TYPE_Lower secondary','ANNUITY_INCOME_PERCENT','DAYS_EMPLOYED_PERCENT','CREDIT_TERM','Score'])
sim_values = []
for crit in sim_criteria:
    values = list(set(X_with_score[crit]))
    selected_values = st.multiselect(crit, values)
    sim_values.append(selected_values)

# filtrage des clients similaires
filter_df = X_with_score.copy()
for i, crit in enumerate(sim_criteria):
    filter_df = filter_df[filter_df[crit].isin(sim_values[i])]

# comparaison avec le client sélectionné
client_id  = st.selectbox('Sélectionnez un client', X_with_score['SK_ID_CURR'].unique(), key=f'client_selector_{client_id}')



if client_id != "":
    client = X_with_score[X_with_score['SK_ID_CURR'] == int(client_id)]
    st.write("Informations du client sélectionné :")
    st.write(client)
    if comparison_choice == "Clients similaires":
        st.write("Informations des clients similaires :")
    else:
        st.write("Informations de tous les clients :")
    st.write(filter_df)


    import plotly.express as px

    # sélection des caractéristiques à comparer
    compare_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3','ANNUITY_INCOME_PERCENT','DAYS_EMPLOYED_PERCENT','CREDIT_TERM', 'Score']

    # calcul des moyennes pour les clients similaires
    sim_mean = filter_df[compare_features].mean()

    # moyennes pour le client sélectionné
    client_mean = client[compare_features].iloc[0]

    # création d'un DataFrame pour la comparaison
    df_compare = pd.DataFrame({'Feature': compare_features, 'Similar clients': sim_mean, 'Selected client': client_mean})

    # mise en forme des données pour la visualisation
    df_compare = df_compare.melt(id_vars=['Feature'], var_name='Group', value_name='Value')

    # création du graphique
    fig = px.bar(df_compare, x='Feature', y='Value', color='Group', barmode='group')

    # affichage du graphique
    st.plotly_chart(fig)



    
 

# Affichage des stastistiques descriptives 



# sélection des variables pour les statistiques descriptives
desc_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'ANNUITY_INCOME_PERCENT', 'DAYS_EMPLOYED_PERCENT', 'CREDIT_TERM']

# calcul des statistiques descriptives pour les clients similaires
sim_desc = filter_df[desc_features].describe().reset_index()
sim_desc.rename(columns={'index': 'Statistiques'}, inplace=True)
sim_desc['Groupe'] = 'Clients similaires'

# calcul des statistiques descriptives pour le client sélectionné
client_desc = client[desc_features].describe().reset_index()
client_desc.rename(columns={'index': 'Statistiques'}, inplace=True)
client_desc['Groupe'] = 'Client sélectionné'

# concaténation des deux DataFrames
df_desc = pd.concat([sim_desc, client_desc], axis=0)

# affichage du tableau de statistiques descriptives
st.write('Tableau des statistiques descriptives')
st.write(df_desc)







# Afficher les SHAP values
st.subheader("Impact des caractéristiques sur le score du client:")
# Sélectionner l'identifiant du client
client_id = st.selectbox('Sélectionnez un client', X_with_score['SK_ID_CURR'].unique(), key='client_selector_1')

# Obtenir les données du client sélectionné
client_data = X_with_score.loc[X_with_score['SK_ID_CURR'] == client_id]

# Sélectionner les caractéristiques du client
client_features = client_data.drop(columns = ['Score','Predicted_TARGET'], axis=1)
#client_features=client_features.set_index('SK_ID_CURR')
client_ids=list(client_features['SK_ID_CURR'])
i =client_ids.index(client_id)
st.write(client_features)

import streamlit as st
import shap
import pandas as pd



# Créer un selecteur de clients
#client_id = st.selectbox("Sélectionnez un client", client_features['SK_ID_CURR'])

# Obtenir les SHAP values pour le client sélectionné
explainer = shap.Explainer(lgbm_model)
shap_values = explainer.shap_values(client_features.drop('SK_ID_CURR', axis=1))

# Re-indexer le shap_values DataFrame en utilisant le client_id comme index
shap_values_df = pd.DataFrame(shap_values[0], columns=client_features.drop('SK_ID_CURR', axis=1).columns)
shap_values_df.index = [client_id]

# Afficher le plot SHAP
st.write("SHAP values pour le client sélectionné")
shap_plot=shap.summary_plot(shap_values[i], client_features.drop('SK_ID_CURR', axis=1), max_display=12)

plt.tight_layout()  # Réduire les marges entre les subplots
st.pyplot()  # Afficher le plot SHAP
# Afficher les SHAP values dans un graphique
 # index du client dans les données
#shap.plots.waterfall(shap_values[0], max_display=12)
summary_plot=shap.summary_plot(shap_values, shap_values_df)
plt.tight_layout()  # Réduire les marges entre les subplots
st.pyplot()




#sum_force_plot=shap.force_plot(explainer.expected_value[1], shap_values[1][:1000,:],  shap_values_df.iloc[:1000,:])
#plt.tight_layout()  # Réduire les marges entre les subplots
#st.pyplot()

