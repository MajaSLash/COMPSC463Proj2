from algorithms import CrimeAnalysisDashboard

def main():
    # Example usage
    crime_dashboard = CrimeAnalysisDashboard('Crime_Data_from_2020_to_Present.csv')
    analysis_results = crime_dashboard.run_comprehensive_analysis()
    
    # Print key insights
    print("Spatial Clustering Silhouette Score:", 
          analysis_results['spatial_clusters']['silhouette_score'])
    print("Crime Type Prediction MSE:", 
          analysis_results['prediction_results']['mean_squared_error'])
    
    # Display top 10 frequent crime areas
    top_areas = crime_dashboard.top_frequent_crime_areas(top_n=10)
    print("\nTop 10 Frequent Crime Areas:")
    for area, count in top_areas:
        print(f"{area}: {count} crimes")

if __name__ == "__main__":
    main()
