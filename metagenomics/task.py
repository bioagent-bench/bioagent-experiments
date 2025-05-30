prompt = """The metagenomes that we will use were collected in Cuatro Ciénegas, 
in a study about the response of the Cuatro Cienegas' bacterial community to nutrient
 enrichment. In this study, authors compared the differences between the microbial 
 community in its natural, oligotrophic, phosphorus-deficient environment, a pond from 
 the Cuatro Ciénegas Basin (CCB), and the same microbial community under a fertilization 
 treatment. Sample data is Control mecocosm (JC1A) and fertilized pond (JP4D).

Think about which steps are necessary to produce this analysis and generate a plan before starting.
The dataset files are provided in the ./data/ directory.
Provide the output processing and results in the ./outputs directory, for each separate step of
analysis create an output subdirectory and name them in order for example step_1, step_2, etc...
The final result should be the output of the percentage change from the 5 most abundant
species in the control data to the fertilization treatment
"""