from algorithms import CrimeAnalysisDashboard

def main():
    # Example usage
    crime_dashboard = CrimeAnalysisDashboard('crime_data.csv')
    analysis_results = crime_dashboard.run_comprehensive_analysis()
    
    # Print key insights
    print("Spatial Clustering Silhouette Score:", 
          analysis_results['spatial_clusters']['silhouette_score'])
    print("Crime Type Prediction MSE:", 
          analysis_results['prediction_results']['mean_squared_error'])

if __name__ == "__main__":
    main()