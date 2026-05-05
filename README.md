# Analyse Comportementale Clientèle – Atelier Machine Learning (GI2)

## Contexte & Mission

Ce projet est un atelier pratique de Machine Learning réalisé dans le cadre du module ML (GI2). Il simule une mission de **Data Scientist** au sein d'une entreprise e-commerce de cadeaux. L'objectif est d'analyser une base de données complexe de **52 features** issues de transactions réelles afin de :

- **Personnaliser** les stratégies marketing.
- **Réduire** le taux de départ des clients (churn).
- **Optimiser** le chiffre d'affaires.

La base de données est intentionnellement imparfaite (valeurs manquantes, aberrantes, formats inconsistants) pour maîtriser l'ensemble de la chaîne de traitement : **Exploration → Préparation → Modélisation → Évaluation → Déploiement**.

---

## Architecture du Projet

Le projet est structuré en trois couches principales :

1. **Pipeline ML (Python / scikit-learn)** : nettoyage, feature engineering, ACP, entraînement et tracking avec MLflow.
2. **API REST (Flask)** : exposition des modèles via une API avec prédiction individuelle, batch, statistiques et réentraînement.
3. **Frontend (Vue.js + Tailwind CSS)** : tableau de bord interactif pour visualiser les prédictions de churn et les statistiques client.

### Structure des dossiers

```
ML/
├── data/
│   ├── raw/                    # Données brutes (retail_customers_COMPLETE_CATEGORICAL.csv)
│   ├── processed/              # Données nettoyées et encodées
│   ├── train_test/             # Splits train/test + versions PCA
│   └── external/               # Données externes (GeoIP, etc.)
├── notebooks/                  # Notebooks Jupyter (prototypage)
├── src/                        # Scripts Python (production)
│   ├── preprocessing.py        # Nettoyage, imputation, parsing des dates
│   ├── utils.py                # Analyse de corrélation, visualisations
│   ├── split_data.py           # Séparation stratifiée train/test
│   ├── pca_transform.py        # Réduction de dimension (ACP)
│   ├── Train_Model.py          # Entraînement de 3 classifieurs + tracking MLflow
│   ├── feature_selector.py     # Modèle Multi-Output (RandomForest) pour inférer 40+ features
│   ├── predict.py              # Script de prédiction autonome
│   └── api.py                  # API Flask (endpoints REST + intégration MLflow)
├── app_Vue/                    # Application frontend (Vue 3 + Vite + Tailwind)
│   ├── src/
│   │   ├── App.vue             # Layout avec sidebar
│   │   ├── views/
│   │   │   ├── Dashboard.vue   # Vue d'ensemble (KPIs, graphiques)
│   │   │   ├── Predict.vue     # Formulaire de prédiction churn
│   │   │   └── History.vue     # Historique des prédictions
│   │   └── router/             # Vue Router
│   └── package.json
├── models/                     # Modèles sérialisés (.pkl / .joblib)
├── reports/                    # Rapports textuels de comparaison de modèles
├── outputs/                    # Visualisations (matrices de corrélation, etc.)
├── mlflow.db                   # Base SQLite de tracking MLflow
├── mlruns/                     # Artefacts MLflow
├── run_pipeline.py             # Orchestrateur du pipeline complet
├── requirements.txt            # Dépendances Python
└── project_tasks.txt           # Consignes pédagogiques détaillées
```

---

## Jeu de Données (52 Features)

Le dataset comporte **52 colonnes** décrivant le comportement client :

- **Numériques (1–34)** : `Recency`, `Frequency`, `MonetaryTotal`, `MonetaryAvg`, `MonetaryStd`, `TotalQuantity`, `CustomerTenure`, `Age`, `SatisfactionScore`, `Churn` (binaire, target), etc.
- **Catégorielles (35–52)** : `RFMSegment`, `AgeCategory`, `SpendingCat`, `CustomerType`, `FavoriteSeason`, `PreferredTime`, `Region`, `LoyaltyLevel`, `ChurnRisk`, `Gender`, `AccountStatus`, `Country`, etc.

### Problèmes de qualité résolus

| Problème | Features concernées | Traitement |
|----------|---------------------|------------|
| Valeurs manquantes | `Age`, `SupportTicketsCount`, `SatisfactionScore` | Imputation (moyenne, médiane, mode selon la distribution) |
| Valeurs aberrantes | `SupportTicketsCount`, `SatisfactionScore` | Filtrage par seuil métier + remplacement |
| Formats inconsistants | `RegistrationDate` (UK, ISO, US) | Parsing robuste avec `pd.to_datetime` |
| Features inutiles | `NewsletterSubscribed` (100% constant), `LastLoginIP` | Suppression ou extraction de sous-features (IP → privée/publique) |
| Déséquilibre de classes | `AccountStatus`, `Churn` | Rééquilibrage via `class_weight="balanced"` |

---

## Pipeline Machine Learning

Le pipeline complet est exécuté via `python run_pipeline.py` :

### 1. Prétraitement (`src/preprocessing.py`)
- Chargement des données brutes.
- Correction de l'âge (imputation par la moyenne, distribution quasi-symétrique).
- Correction de `SupportTicketsCount` (médiane sur [0,7]) et `SatisfactionScore` (mode sur [0,5]).
- Parsing robuste des dates d'inscription (`RegistrationDate`).
- Extraction de features depuis l'IP de connexion (`LastLoginIP`).
- Encodage des variables catégorielles (`LabelEncoder`, `One-Hot`).

### 2. Analyse de Corrélation (`src/utils.py`)
- Génération de la matrice de corrélation (`correlation_matrix.png`).
- Détection de la multicolinéarité (seuil |corrélation| > 0.8).
- Visualisations exportées dans `outputs/`.

### 3. Split Train/Test (`src/split_data.py`)
- Séparation **80/20** stratifiée sur la target `Churn` (`random_state=42`).
- Sauvegarde en CSV dans `data/train_test/`.

### 4. Réduction de Dimension – ACP (`src/pca_transform.py`)
- Application du `StandardScaler` (fit sur train uniquement, transform sur test).
- **PCA** pour réduire les features à un nombre optimal de composantes tout en conservant la variance expliquée.
- Sauvegarde du scaler et du modèle PCA dans `models/`.

### 5. Modélisation (`src/Train_Model.py`)
Trois algorithmes sont entraînés et comparés sur les données PCA :

- **Logistic Regression** (`class_weight='balanced'`, `max_iter=1000`)
- **Random Forest Classifier** (`n_estimators=100`, `max_depth=10`)
- **Gradient Boosting Classifier** (`n_estimators=100`, `learning_rate=0.1`)

Chaque modèle est loggé dans **MLflow** (params, metrics : accuracy, precision, recall, f1, roc_auc, confusion matrix). Le meilleur modèle (selon **ROC-AUC**) est sauvegardé localement en `models/model.pkl`.

### 6. Feature Selector (`src/feature_selector.py`)
Un modèle **MultiOutputRegressor (RandomForest)** est entraîné pour inférer **~40 features cibles** à partir de **9 features d'entrée** connues :

`Recency`, `Age`, `Region`, `MonetaryTotal`, `Frequency`, `SatisfactionScore`, `LoyaltyLevel`, `CustomerType`, `AccountStatus`.

Ce modèle est utilisé par l'API pour enrichir un profil client à partir de données partielles.

---

## API Flask (`src/api.py`)

L'API expose les endpoints suivants (CORS activé) :

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/health` | GET | Vérification de santé + URI MLflow |
| `/api/inputs` | GET | Métadonnées des 9 features d'entrée (min/max, options catégorielles) |
| `/api/predict` | POST | Prédiction churn + inférence des features manquantes. Logging MLflow par prédiction. |
| `/api/batch` | POST | Prédiction batch sur plusieurs clients. Logging MLflow agrégé. |
| `/api/dataset_stats` | GET | Statistiques globales (total clients, churn rate, etc.) |
| `/api/churn_summary` | GET | Distribution churn (Loyal vs Churn) pour graphique doughnut |
| `/api/risk_distribution` | GET | Distribution des risques (Low / Medium / High) sur un échantillon |
| `/api/sample_predictions` | GET | 10 premières lignes avec prédictions pour tableau de bord |
| `/api/region_churn` | GET | Taux de churn par région (graphique en barres) |
| `/api/train` | POST | Réentraînement des 3 modèles via l'API, résultats loggés dans MLflow |
| `/api/predict_with_mlflow_model` | POST | Prédiction via un modèle sklearn chargé directement depuis MLflow |

---

## Frontend Vue.js (`app_Vue/`)

Application **SPA (Single Page Application)** développée avec :

- **Vue 3** (Composition API)
- **Vue Router** (navigation Dashboard / Predict / History)
- **Tailwind CSS** (styling utilitaire)
- **Chart.js + vue-chartjs** (graphiques interactifs)
- **Lucide Vue Next** (icônes)

### Pages

- **Dashboard** : KPIs (total clients, churn rate, loyal customers), graphiques doughnut (distribution churn), bar chart (churn par région), tableau des prédictions récentes.
- **Predict** : Formulaire dynamique basé sur les 9 features d'entrée. Appel à `/api/predict`. Affichage du résultat churn (probabilité, label, niveau de risque) et des features inférées catégorisées (Monetary, Behaviour, Preferences, Profile).
- **History** : Historique des prédictions avec filtrage et détails.

---

## Tracking & MLOps – MLflow

Le projet intègre **MLflow** avec une base SQLite locale (`mlflow.db`) pour :

- Tracker chaque run d'entraînement (hyperparamètres, métriques, modèles).
- Comparer les 3 algorithmes (LogisticRegression, RandomForest, GradientBoosting).
- Logger les prédictions API (inputs, probabilités, classes prédites).
- Charger dynamiquement le meilleur modèle pour inférence via `/api/predict_with_mlflow_model`.

---

## Technologies Utilisées

### Backend / Data Science
- **Python 3**
- **pandas, numpy** – manipulation de données
- **scikit-learn** – prétraitement, ACP, classification, régression multi-sortie
- **Flask, flask-cors** – API REST
- **MLflow** – tracking d'expériences et de modèles
- **joblib** – sérialisation de modèles
- **matplotlib, seaborn** – visualisations
- **geoip2, python-dotenv** – GeoIP et variables d'environnement

### Frontend
- **Vue 3** + **Vite**
- **Vue Router**
- **Tailwind CSS** + **PostCSS** + **Autoprefixer**
- **Chart.js** + **vue-chartjs**
- **lucide-vue-next**

---

## Installation & Utilisation

### 1. Environnement Python

```bash
# Création et activation du venv
python -m venv venv
venv\Scripts\activate        # Windows

# Installation des dépendances
pip install -r requirements.txt
```

### 2. Exécution du Pipeline ML

```bash
# Lancer le pipeline complet (preprocessing → split → PCA → training)
python run_pipeline.py
```

Fichiers générés :
- `data/processed/retail_customers_processed.csv`
- `data/train_test/X_train_pca.csv`, `X_test_pca.csv`, `y_train.csv`, `y_test.csv`
- `models/model.pkl`, `scaler.pkl`, `pca.pkl`
- `reports/model_results.txt`
- `outputs/correlation_*.png`

### 3. Lancement de l'API

```bash
python src/api.py
```
L'API est accessible sur `http://127.0.0.1:5000`.

### 4. Lancement du Frontend

```bash
cd app_Vue
npm install
npm run dev
```
Le frontend est accessible sur l'URL affichée par Vite (généralement `http://localhost:5173`).

---

## Auteur

Atelier Machine Learning – GI2  
Année universitaire 2025-2026
