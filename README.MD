# Project Files for My Masterthesis

## Final Code 

Contains the actual Project. The Final Code is split between:

### Simple Policy

- A detailed version for specifically examining a single or multiple instances. Saves to P_Results.
- A efficient version for fast Quick Results

### Reinforcement Learning

- A detailed version for specifically examining a single or multiple instances. Saves to RL_Results
- A efficient version for fast Quick Results

## Instance Prerequisites

Class Instance File. To create a class you need the name of the Solomonfile, the customer-size, the capacity, and the demand range. These are mandatory and define an Instance. Refer to Solomon Instances Folder to see all available.

## Apriori

Basically handling the "routing". You can change the Google OR Tools FirstSolutionStrategy.PATH_CHEAPEST_ARC to whatever you prefer in Line 101. Sets Customer Order via a priori and creates the distance matrix.

## Mega Instance Generator

Nested for loop and result generation with xlsx output via pandas and openpyxl. Currently, creates the 1920 instances. Saves to results.xlsx with advanced pre-processed columns for easier analysis.

## XLSX

Results, P_Results, and RL_Results are from output. ANOVA_Appendix is containing all the Analysis, Tables, Figures featured in the Master-thesis.

## Archived

Refers to stuff I used as practice and to learn Python+RL. Nothing in there is used in the final implementation.
