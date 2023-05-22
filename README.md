# OpenClassRooms_Projet7
# CREDIT SCORING PREDICTION DASHBOARD

#### <i>Implémentez un modèle de scoring</i>

## Présentation


    La société financière, nommée "Prêt à dépenser", propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

    L’entreprise souhaite développer un modèle de scoring de la probabilité de défaut de paiement du client pour étayer la décision d'accorder ou non un prêt à un client potentiel en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

    De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

    La société décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.



Ce projet consiste à créer une API web avec un Dashboard interactif. Celui-ci devra a minima contenir les fonctionnalités suivantes :

 - Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
 - Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
 - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

Pour l'analyse exploratoire et le feature ingeniering, il nous est proposé d'utiliser un des Notebooks disponibles sur le site de Kaggle, dont le lien est disponible au bas de cette présentation.

## Construction

<u>Dans ce dépôt, vous trouverez :</u>

 - Un dossier avec le Notebook Jupyter pour l'analyse exploratoire et le feature ingeniering, l'entraînement et la configuration du modèle de classification avec les hyperparamètres optimaux. Le notebook inclut également deux sections d'analyse de data drift avec la librairie Evidently et la déterminantion d'une fonction de coût métier qui minimise les perte de la société financière "Prêt à dépenser".
 - Un dossier avec la note technique qui explique en détails la construction et les résultats du modèle.
 - Un dossier avec la configuration de l'API et le déploiement de l'application sur Streamlit. 
   
 - Un dossier Mlflow permettant d'accéder aux différentes versions des modèles entrainés et optimisés sur trois optimisateurs: RandomizedSearchCV, GridSearch et Optuna
## Modèle de classification
Le modèle retenu pour développer le dashbord est le modèle LigthGBM. En comparant les différentes métriques de performance des quatre modèles que nous avons testés, à savoir Xgboost, LogisticRegression, RandomForestClassifier et LigthGBM, ce dernier se révèle être le modèle adéquat.  

## Dashboard / API
Le dashboard est entièrement développé et déployé sur Streamlit
 

## Données d'entrées
 - Lien de téléchargement des données d'entrées : https://www.kaggle.com/c/home-credit-default-risk/data 
 - Notebook de départ pour la partie Features Engineering : https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
