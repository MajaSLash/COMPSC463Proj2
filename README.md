Project Repository for COMPSC463 Project 2.

The program we chose to develop is a Crime Analysis Application with Live Data.

# Project Description

In this project, students will have the opportunity to propose and develop their own project related to the theme of crime/violence. The goal is to encourage students to apply their knowledge of algorithms to a relevant and meaningful problem in the context of crime/violence. In this group project assignment, you and your team member will embark on a creative and meaningful journey to develop a software application that reflects the knowledge and skills you have acquired in this course. This project serves as a culmination of your learning experience, aiming to apply what you've learned to solve real-world problems through software development and teamwork. Students are encouraged to explore various aspects of crime/violence such as crime and violence, such as crime detection and monitoring, resource allocation for law enforcement, strategies for crime prevention and community safety, data logging and real-time analysis of crime patterns, simulation and forecasting of crime trends, and emergency communication systems for crisis response, or any other related area of interest. Through this project, students will have the flexibility to showcase their creativity, critical thinking, and technical skills while addressing important challenges in the crime/violence problem.

 

*Project Groups*

To facilitate collaboration and ensure that every student has the opportunity to work with a team member, group formation will follow the following guidelines:

- Two-Person Groups: Students are encouraged to form their own groups consisting of two individuals who share a common interest or preference in the project topic. These groups should be formed voluntarily.
- Remaining Students: If there is an odd number of students or if an individual is unable to find a partner, they will be assigned to a group by the faculty. The faculty will make reasonable efforts to create groups that align with students' interests and skills.
- Faculty-Assigned Groups: In cases that students are unable to find a partner or request faculty assistance, the faculty can assign groups to ensure that every student is part of a team.


*Expected Skills Gained*

- Critical and creative thinking
- Developing a user-friendly interface
- Collaborative coding
- Real-world application
- Problem solving 


*Deliverables*

- Source Code: Submit the source code, including comments that explain the design and implementation. Store the code on a GitHub repository.

- Project Report: Prepare a comprehensive project report that includes the following.

    1. Project goals: Clearly define the project's objectives.

    2. Significance and novelty of the project: Background information and Why the project is meaningful and novel.

    3. Installation and usage instructions: Provide clear instructions for installing and using the software.

    4. Code structure: Present a systematic flow-chart of the code's structure and explanations for easy understanding.

    5. List of functionalities and verification results: Describe the functionalities and present testing results for verification.

    6. Showcasing the achievement of project goals: Provide some execution results and discuss your result on how your project achieves the project goal

    7. Discussion and Conclusions: Address project issues, limitations, and how your course learning were applied.

- GitHub Repository Link: Share a GitHub repository link with the project report as a README (Markdown file, Word, or pdf) and the source code as a separate file.
 

*Grading Criteria*

Your project will be assessed according to the following criteria:

- Code (15%)
    - Code Quality: Code should be well-structured and readable.

- Report (75%)
- Goal of the project (5%)
    - Clearly state the project's objectives.

- Significance of the project (10%)
    - Explain the project's meaningfulness and novelty

- Installation and Instruction to use (5%)
    - Provide clear installation and usage instructions.

- Structure of the code (5%)
    - Include a systematic code structure diagram and clear explanations.

- Functionalities and Test Results (15%)
    - Present functionalities and testing results for verification.

- Showcasing the achievement of project goals (15%)
    - Present the results and discuss the implementation to show off the achievement of the project goal

- Discussion and Conclusions (10%)
    - Discuss project issues, limitations, and the application of course learning.

- Overall quality of report (10%)
    - editing and quality of writing, 

- GitHub (10%)
    - Ensure all project components are on the GitHub repository.
    - Set the GitHub repository as public initially.

 

*Submission*

- Your project submission should include a GitHub repository link containing all the required deliverables for evaluation.
- Each student needs to submit their project on Canvas individually even though they submit the same link.

 

*Note*

- Ensure that within your group, every member respects each other's contributions, including your own. If your team demonstrates harmonious dynamics and all members, including yourself, contributed equally, the same score will be assigned to all members of the group. In the event that you or another team member argues that your own or another member's contribution is weak, the faculty will listen to the perspectives of all group members, including yours, and decide the scores accordingly. This ensures fair assessment of individual contributions within the team.

- Everything here can be subject to change with common-sense reasoning

# Code Structure

COMPSC463Proj2<br>
    └── main.py<br>
    └── algorithms.py<br>
    └── tests.py<br>
    └── Crime_Data_from_2020_to_Present.csv<br>

- main.py: Driver code for the application.

- algorithm.py: Contains the various algorithms used to produce the analysis results based on the data.
  - Classes:
    - CrimeAnalysisDashboard: Master Class to run all generative processes and analysis
  - Methods:
    - preprocess_data: Preprocess and clean the crime data
    - spatial_clustering: Perform spatial clustering of crime locations using K-Means
        - :param n_clusters: Number of clusters to create
        - :return: Clustered data and clustering metrics
    - temporal_trend_analysis: Analyze temporal trends in crime data
    - crime_type_prediction: Predict crime type likelihood using Random Forest
    - def density_hotspot_analysis: Create density-based hotspot analysis
    - top_frequent_crime_areas: List the top N crime hotspots using a greedy algorithm
    - visualize_crime_insights: Create comprehensive visualizations of crime analysis results
    - run_comprehensive_analysis: Execute a comprehensive crime data analysis pipeline

- tests.py: Test cases for application functions

- Crime_Data_from_2020_to_Present.csv: Dataset downloaded from Kaggle

# Tutorial

*Required Installs*
Download the dataset [here](https://www.kaggle.com/datasets/haseefalam/crime-dataset?resource=download)

*Guide*
- Run `git clone https://github.com/MajaSLash/COMPSC463Proj2.git` in your terminal to download this repository.
- Download the dataset from Kaggle to a CSV file (Instructions linked [here](https://www.kaggle.com/discussions/getting-started/58426)).
- Move the file into this project's directory.
- Run the program from main.py
- To use the test case suite, run `python -m unittest tests.py` in your terminal.

- Example Usage:
  - Output: ![alt text](report_images/image_2.png)
  - Chart: ![alt text](report_images/image_3.png)
  
# Test Cases

*Test Cases for Individual Parts*
Indiviudal Cases for each function are located in test.py
- Example Usage:
  - Output: ![alt text](report_images/image.png)

 
# Discussion and Conclusion

### Issues and Limitations:
- Missing or incomplete data, especially for geographic coordinates, can affect the quality of analysis.
- Due to the large size of the dataset, only 500 rows were used for this analysis. This may limit the generalizability and accuracy of the model, as a larger dataset could provide more reliable insights.

### Course Learning Applied:
- **Greedy Algorithm**: Used a greedy algorithm to find and list the top frequent crime areas based on their crime count. This approach ensures that the most crime-prone areas are prioritized for resource allocation.

This project effectively showcases the practical application of the concepts learned in COMPSC463 throughout the semester. It successfully delivers a comprehensive crime analysis dashboard, providing valuable insights through spatial, temporal, and predictive analyses. The project meets its goals of analyzing crime data and improving resource allocation, while also integrating algorithms like greedy algorithms and machine learning models for practical solutions. 
