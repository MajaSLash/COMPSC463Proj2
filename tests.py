import unittest
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from algorithms import CrimeAnalysisDashboard  # Assuming the original class is in this module

class TestCrimeAnalysisDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up test data and initialize the dashboard for testing
        """
        # Create a sample test CSV file with minimal required columns
        test_data = {
            'DATE OCC': ['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-02'],
            'TIME OCC': [1200, 1300, 1400, 1500],
            'LAT': [34.0522, 34.0523, 34.0524, 34.0525],
            'LON': [-118.2437, -118.2438, -118.2439, -118.2440],
            'DR_NO': [1, 2, 3, 4],
            'Crm Cd Desc': ['Theft', 'Assault', 'Robbery', 'Burglary'],
            'AREA NAME': ['Central', 'West LA', 'Hollywood', 'South']
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_csv('test_crime_data.csv', index=False)
        
        # Initialize dashboard with test data
        cls.dashboard = CrimeAnalysisDashboard('test_crime_data.csv')

    @classmethod
    def tearDownClass(cls):
        """
        Clean up test data file after tests complete
        """
        if os.path.exists('test_crime_data.csv'):
            os.remove('test_crime_data.csv')
        if os.path.exists('crime_analysis_insights.png'):
            os.remove('crime_analysis_insights.png')

    def test_initialization(self):
        """
        Test that the dashboard initializes correctly
        """
        self.assertIsNotNone(self.dashboard.raw_data)
        self.assertEqual(len(self.dashboard.raw_data), 4)

    def test_preprocess_data(self):
        """
        Test data preprocessing methods
        """
        # Check datetime conversion
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.dashboard.raw_data['DATE OCC']))
        
        # Check additional features
        self.assertIn('month', self.dashboard.raw_data.columns)
        self.assertIn('day_of_week', self.dashboard.raw_data.columns)
        self.assertIn('hour', self.dashboard.raw_data.columns)
        
        # Check encoding
        self.assertTrue('Crm Cd Desc' in self.dashboard.raw_data.columns)
        self.assertTrue(len(self.dashboard.crime_mapping) > 0)

    def test_spatial_clustering(self):
        """
        Test spatial clustering method
        """
        cluster_results = self.dashboard.spatial_clustering()
        
        # Check cluster results
        self.assertIn('clustered_data', cluster_results)
        self.assertIn('cluster_centers', cluster_results)
        self.assertIn('silhouette_score', cluster_results)
        
        # Check clusters were added to the dataframe
        self.assertTrue('cluster' in self.dashboard.raw_data.columns)
        
        # Validate silhouette score
        self.assertTrue(0 <= cluster_results['silhouette_score'] <= 1)

    def test_temporal_trend_analysis(self):
        """
        Test temporal trend analysis method
        """
        temporal_trends = self.dashboard.temporal_trend_analysis()
        
        # Check trend analysis results
        self.assertIn('monthly_trends', temporal_trends)
        self.assertIn('day_of_week_trends', temporal_trends)
        self.assertIn('hourly_trends', temporal_trends)
        
        # Validate trends are not empty
        self.assertTrue(len(temporal_trends['monthly_trends']) > 0)
        self.assertTrue(len(temporal_trends['day_of_week_trends']) > 0)
        self.assertTrue(len(temporal_trends['hourly_trends']) > 0)

    def test_crime_type_prediction(self):
        """
        Test crime type prediction method
        """
        prediction_results = self.dashboard.crime_type_prediction()
        
        # Check prediction results
        self.assertIsInstance(prediction_results['model'], RandomForestRegressor)
        self.assertIsNotNone(prediction_results['mean_squared_error'])
        
        # Check feature importance
        feature_importance = prediction_results['feature_importance']
        self.assertEqual(len(feature_importance), 5)  # 5 features
        self.assertTrue(all(0 <= val <= 1 for val in feature_importance.values()))

    def test_density_hotspot_analysis(self):
        """
        Test density hotspot analysis method
        """
        hotspot_data = self.dashboard.density_hotspot_analysis()
        
        # Check hotspot data
        self.assertIn('density_grid', hotspot_data)
        self.assertIn('x_grid', hotspot_data)
        self.assertIn('y_grid', hotspot_data)
        
        # Validate grid shapes
        self.assertEqual(hotspot_data['density_grid'].shape, (100, 100))

    def test_top_frequent_crime_areas(self):
        """
        Test top frequent crime areas method
        """
        top_areas = self.dashboard.top_frequent_crime_areas()
        
        # Check top areas result
        self.assertTrue(isinstance(top_areas, list))
        self.assertTrue(all(isinstance(area, tuple) and len(area) == 2 for area in top_areas))

    def test_run_comprehensive_analysis(self):
        """
        Test the comprehensive analysis pipeline
        """
        results = self.dashboard.run_comprehensive_analysis()
        
        # Check all key components are present
        self.assertIn('spatial_clusters', results)
        self.assertIn('temporal_trends', results)
        self.assertIn('prediction_results', results)
        self.assertIn('hotspot_data', results)
        
        # Check that visualization was created
        self.assertTrue(os.path.exists('crime_analysis_insights.png'))

    def test_edge_cases(self):
        """
        Test edge cases and error handling
        """
        # Test with empty data or invalid data
        with self.assertRaises(Exception):
            CrimeAnalysisDashboard('nonexistent_file.csv')

        # Test clustering with different number of clusters
        cluster_results_3 = self.dashboard.spatial_clustering(n_clusters=3)
        self.assertEqual(len(np.unique(cluster_results_3['clustered_data']['cluster'])), 3)

if __name__ == '__main__':
    unittest.main()