import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, silhouette_score
from scipy.stats import gaussian_kde
import geopandas as gpd
import warnings

class CrimeAnalysisDashboard:
    def __init__(self, data_path):
        """
        Initialize the Crime Analysis Dashboard with crime data
        
        :param data_path: Path to the CSV file containing crime data
        """
        warnings.filterwarnings('ignore')
        
        # Load and preprocess crime data
        self.raw_data = pd.read_csv(data_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        """
        Preprocess and clean the crime data
        """
        # Convert date columns
        self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
        
        # Handle missing values
        self.raw_data.fillna({
            'latitude': self.raw_data['latitude'].mean(),
            'longitude': self.raw_data['longitude'].mean()
        }, inplace=True)
        
        # Create additional features
        self.raw_data['month'] = self.raw_data['date'].dt.month
        self.raw_data['day_of_week'] = self.raw_data['date'].dt.dayofweek
        self.raw_data['hour'] = self.raw_data['date'].dt.hour
        
    def spatial_clustering(self, n_clusters=5):
        """
        Perform spatial clustering of crime locations using K-Means
        
        :param n_clusters: Number of clusters to create
        :return: Clustered data and clustering metrics
        """
        # Prepare spatial features
        spatial_features = self.raw_data[['latitude', 'longitude']]
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(spatial_features)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.raw_data['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_features, self.raw_data['cluster'])
        
        return {
            'clustered_data': self.raw_data,
            'cluster_centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette_avg
        }
    
    def temporal_trend_analysis(self):
        """
        Analyze temporal trends in crime data
        
        :return: Dictionary of temporal trend insights
        """
        # Monthly crime frequency
        monthly_crimes = self.raw_data.groupby('month')['crime_type'].count()
        
        # Day of week analysis
        day_of_week_crimes = self.raw_data.groupby('day_of_week')['crime_type'].count()
        
        # Hourly crime distribution
        hourly_crimes = self.raw_data.groupby('hour')['crime_type'].count()
        
        return {
            'monthly_trends': monthly_crimes,
            'day_of_week_trends': day_of_week_crimes,
            'hourly_trends': hourly_crimes
        }
    
    def crime_type_prediction(self, test_size=0.2):
        """
        Predict crime type likelihood using Random Forest
        
        :param test_size: Proportion of data to use for testing
        :return: Prediction model and performance metrics
        """
        # Prepare features for prediction
        features = ['latitude', 'longitude', 'month', 'day_of_week', 'hour']
        X = self.raw_data[features]
        y = self.raw_data['crime_type']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'model': rf_model,
            'mean_squared_error': mse,
            'feature_importance': dict(zip(features, rf_model.feature_importances_))
        }
    
    def density_hotspot_analysis(self):
        """
        Create density-based hotspot analysis
        
        :return: Kernel Density Estimation results
        """
        # Extract coordinates
        coords = self.raw_data[['latitude', 'longitude']].values
        
        # Calculate kernel density
        kde = gaussian_kde(coords.T)
        
        # Generate grid for density estimation
        x_min, x_max = coords[:,0].min(), coords[:,0].max()
        y_min, y_max = coords[:,1].min(), coords[:,1].max()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        # Compute density
        z = np.reshape(kde(positions).T, xx.shape)
        
        return {
            'density_grid': z,
            'x_grid': xx,
            'y_grid': yy
        }
    
    def visualize_crime_insights(self, cluster_results, temporal_trends, hotspot_data):
        """
        Create comprehensive visualizations of crime analysis results
        
        :param cluster_results: Spatial clustering results
        :param temporal_trends: Temporal trend analysis results
        :param hotspot_data: Density hotspot analysis results
        """
        plt.figure(figsize=(20,15))
        
        # Spatial Clustering Subplot
        plt.subplot(2,2,1)
        scatter = plt.scatter(
            cluster_results['clustered_data']['longitude'], 
            cluster_results['clustered_data']['latitude'],
            c=cluster_results['clustered_data']['cluster'], 
            cmap='viridis'
        )
        plt.title('Spatial Crime Clusters')
        plt.colorbar(scatter)
        
        # Monthly Trends Subplot
        plt.subplot(2,2,2)
        temporal_trends['monthly_trends'].plot(kind='bar')
        plt.title('Monthly Crime Frequency')
        plt.xlabel('Month')
        plt.ylabel('Crime Count')
        
        # Hourly Crime Distribution
        plt.subplot(2,2,3)
        temporal_trends['hourly_trends'].plot(kind='line')
        plt.title('Hourly Crime Distribution')
        plt.xlabel('Hour of Day')
        plt.ylabel('Crime Count')
        
        # Crime Density Heatmap
        plt.subplot(2,2,4)
        plt.contourf(
            hotspot_data['x_grid'], 
            hotspot_data['y_grid'], 
            hotspot_data['density_grid'], 
            cmap='YlOrRd'
        )
        plt.title('Crime Density Hotspots')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('crime_analysis_insights.png')
        plt.close()
    
    def run_comprehensive_analysis(self):
        """
        Execute a comprehensive crime data analysis pipeline
        """
        # Perform spatial clustering
        cluster_results = self.spatial_clustering()
        
        # Analyze temporal trends
        temporal_trends = self.temporal_trend_analysis()
        
        # Perform crime type prediction
        prediction_results = self.crime_type_prediction()
        
        # Create density hotspot analysis
        hotspot_data = self.density_hotspot_analysis()
        
        # Visualize insights
        self.visualize_crime_insights(cluster_results, temporal_trends, hotspot_data)
        
        return {
            'spatial_clusters': cluster_results,
            'temporal_trends': temporal_trends,
            'prediction_results': prediction_results,
            'hotspot_data': hotspot_data
        }