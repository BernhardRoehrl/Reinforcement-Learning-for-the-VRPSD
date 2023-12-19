# Project Files for My Masterthesis

### Simple Policy  

- A detailed version for specifically examining a single or multiple instances. Saves detailed Output to P_Results for analyzing specific instances.
- An efficient version returning results for the Final Complete Instance Analysis.

### Reinforcement Learning

- A detailed version for specifically examining a single or multiple instances. Saves to RL_Results
- An efficient version returning results for the Final Complete Instance Analysis.

### Instance Prerequisites

Class Instance File. To create a class you need the name of the Solomonfile, the customer-size, the capacity, and the demand range. These are mandatory and define an Instance. Refer to Solomon Instances Folder to see all available.

### Apriori

Handling everything related to the Google OR Tool solving. You can change the Google OR Tools FirstSolutionStrategy.PATH_CHEAPEST_ARC to whatever you prefer in Line 101. Sets Customer Order via a priori and creates the distance matrix.

### Mega Instance Generator

Nested for loop and result generation with xlsx output via pandas and openpyxl. Currently, creates the 1920 instances. Saves to results.xlsx with advanced pre-processed columns for easier analysis. Uses the Reinforcement/SimplePolicy_Efficient.py

### XLSX-Files

Results, P_Results, and RL_Results are from output. ANOVA_Appendix is containing all the Analysis, Tables, Figures featured in the Master-thesis.

### Archived

Old Project Files & Tutorial and Learning Related Residuents 
