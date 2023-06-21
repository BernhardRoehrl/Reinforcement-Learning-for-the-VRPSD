These are my project files for my Masterthesis.
Archived refers to stuff I used as practice and to learn everything.
Final Code Contains the actual Project.

The Final clode is split between:

-Simple Policy

-Reinforcement Learning

The input_string.txt has to contain any Solomon1987 Instance from the Capacitated Vehicle Routing Problem Libary.
Prerequisites """Profile""" needs to be set up (demands, capacity, etc.) for the desired Instance manually because --> 
Only the X and Y Coordinate are extracted from the input_string.txt and processed to cut_string.txt when the Policy of
your choice (Reinforcement_Learning.py or Simple_Policy.py) is run. Apriori.py is just a modification from the Google OR
VRP Tool and sets the customer order and the beginning of each policy. Results get captured in both .xlsx files. 
Within the Reinforcement_Learning.py you can adjust some learning parameters. I did my best to find scalable hyperparameters
that work on all kinds of SolomonInstances.

TLDR:

-Put Solomon Instance into input_string.txt

-Set Up Instance Profile in Prerequisites

-Run Reinforcement_Learning.py or Simple_Policy.py