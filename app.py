import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.cluster import AgglomerativeClustering

# Try importing XGBRegressor safely
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df['wip'].fillna(value=0, inplace=True)
    df['department'] = df['department'].str.replace("sweing", "sewing")
    df = df.sort_values(by='date').reset_index(drop=True)
    df['department'] = df.apply(lambda row: 'finishing' if row['wip'] == 0 else 'sewing' if pd.isna(row['department']) and row['wip'] > 0 else row['department'], axis=1)
    df = df.dropna(subset=['date'])
    for i in range(1, len(df) - 1):
        if pd.isna(df.loc[i, 'quarter']):
            previous_quarter = df.loc[i - 1, 'quarter']
            next_quarter = df.loc[i + 1, 'quarter']
            previous_date = df.loc[i - 1, 'date']
            next_date = df.loc[i + 1, 'date']
            if (previous_quarter == next_quarter and previous_date.month == next_date.month and previous_date.day == next_date.day):
                df.loc[i, 'quarter'] = previous_quarter
    df = df.dropna(subset=['quarter'])
    for i in range(1, len(df) - 1):
        if pd.isna(df.iloc[i]['day']):
            previous_date = df.iloc[i - 1]['date']
            next_date = df.iloc[i + 1]['date']
            if previous_date.date() == next_date.date():
                df.iloc[i, df.columns.get_loc('day')] = df.iloc[i - 1]['day']
    df = df.dropna(subset=['day'])
    def fill_idle_columns(row):
        if pd.isna(row['idle_time']) and pd.isna(row['idle_men']):
            row['idle_time'] = 0
            row['idle_men'] = 0
        if pd.isna(row['idle_time']):
            if row['idle_men'] == 0:
                row['idle_time'] = 0
        if pd.isna(row['idle_men']):
            if row['idle_time'] == 0:
                row['idle_men'] = 0
        return row
    df = df.apply(fill_idle_columns, axis=1)
    df['incentive'] = df['incentive'].fillna(0)
    df['over_time'] = df['over_time'].fillna(0)
    df = df.dropna(subset=['no_of_workers'])
    df = df.dropna(subset=['team'])
    def set_no_of_style_change(row):
        if pd.isna(row['no_of_style_change']) and pd.notna(row['smv']):
            if row['smv'] == 11.41:
                return 2
            elif row['smv'] == 30.1:
                return 1
        return row['no_of_style_change']
    df['no_of_style_change'] = df.apply(set_no_of_style_change, axis=1)
    df = df.dropna(subset=['no_of_style_change'])
    df = df.dropna(subset=['smv'])
    df['no_of_workers'] = df['no_of_workers'].apply(lambda x: int(x))
    df['actual_productivity'] = df['actual_productivity'].astype(float)
    df = df.dropna(subset=['actual_productivity'])
    df = df.dropna(subset=['targeted_productivity'])
    df['department'] = df['department'].str.replace("sweing", "sewing")
    df['quarter'] = df['quarter'].astype(str).str.replace('Quarter','')
    df['quarter'] = pd.to_numeric(df['quarter'], errors='coerce')
    df['department'] = df['department'].str.replace('finishing ','finishing')
    df['day'] = df['day'].replace({
        'Monday': 0, 'Tuesday': 1, 'Wednesday':2, 'Thursday':3, 'Saturday':4, 'Sunday':5
    })
    df['department'] = df['department'].replace({'sewing':0, 'finishing':1})
    df = df.drop(columns=['date'], errors='ignore')
    return df

def train_and_evaluate_models(df):
    df_enc = df.copy()
    X = df_enc.drop(['actual_productivity'], axis=1)
    y = df_enc['actual_productivity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=0)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    y_train = pd.to_numeric(y_train, errors='coerce').fillna(y_train.mean())
    y_test = pd.to_numeric(y_test, errors='coerce').fillna(y_test.mean())
    results = []

    # Linear Regression
    regression = LinearRegression()
    regression.fit(X_train_sc, y_train)
    y_pred = regression.predict(X_test_sc)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results.append(('Linear Regression', rmse))

    # Ridge Regression
    ridge = Ridge(alpha=1.9)
    ridge.fit(X_train_sc, y_train)
    y_pred = ridge.predict(X_test_sc)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results.append(('Ridge Regression', rmse))

    # KNN Regression
    knn = KNeighborsRegressor(n_neighbors=7, metric='manhattan')
    knn.fit(X_train_sc, y_train)
    y_pred = knn.predict(X_test_sc)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results.append(('KNN Regression', rmse))

    # Random Forest
    forest = RandomForestRegressor(n_estimators=50, random_state=0, min_samples_split=10, max_depth=6)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results.append(('Random Forest', rmse))

    # XGBoost Regression (only if available)
    if xgb_available:
        try:
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=0, verbosity=0)
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            results.append(('XGBoost Regression', rmse))
        except Exception as e:
            results.append(('XGBoost Regression', f'Error: {e}'))
    else:
        results.append(('XGBoost Regression', 'XGBoost not installed'))

    # Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=5, random_state=0)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results.append(('Gradient Boosting', rmse))

    # Find best model (ignore models with string errors)
    numeric_results = [r for r in results if isinstance(r[1], (int, float, np.floating))]
    best_model = min(numeric_results, key=lambda x: x[1]) if numeric_results else ("None", "N/A")
    return results, best_model

def clustering_results(df):
    X = df.drop(['actual_productivity'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clustering_info = []

    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
    kmeans_n_clusters = len(set(kmeans_labels))
    clustering_info.append({
        'name': 'KMeans',
        'silhouette': kmeans_sil,
        'n_clusters': kmeans_n_clusters
    })

    # DBSCAN
    dbscan = DBSCAN(eps=0.7, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_sil = silhouette_score(X_scaled, dbscan_labels) if dbscan_n_clusters > 1 else None
    clustering_info.append({
        'name': 'DBSCAN',
        'silhouette': dbscan_sil,
        'n_clusters': dbscan_n_clusters
    })

    # GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_sil = silhouette_score(X_scaled, gmm_labels)
    gmm_n_clusters = len(set(gmm_labels))
    clustering_info.append({
        'name': 'GMM',
        'silhouette': gmm_sil,
        'n_clusters': gmm_n_clusters
    })

    # CAH (Agglomerative Clustering)
    cah = AgglomerativeClustering(n_clusters=2)
    cah_labels = cah.fit_predict(X_scaled)
    cah_sil = silhouette_score(X_scaled, cah_labels)
    cah_n_clusters = len(set(cah_labels))
    clustering_info.append({
        'name': 'CAH',
        'silhouette': cah_sil,
        'n_clusters': cah_n_clusters
    })

    return clustering_info

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            try:
                df = preprocess_data(df)
                results, best_model = train_and_evaluate_models(df)
                clustering_info = clustering_results(df)
                return render_template('results.html', results=results, best_model=best_model, clustering_info=clustering_info)
            except Exception as e:
                flash(f'Error processing file: {e}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)