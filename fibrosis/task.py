prompt = """The sample dataset is a simulated dataset for finding the generic 
cause of Cystic fibrosis. The dataset is real sequencing data from CEPH_1463 
dataset provided by the Complete Genomics Diversity Panel. It consists of sequencing of a 
family: 4 grandparents, 2 parents and 11 siblings. A known Mandelian disease mutation has been 
added on three siblings, taking care to be consistent with the underlying heplotype structure. 
The goal is to find the mutation causing the mendalian recessive trait - Cystic Fibrosis. 
The samples with the observed phenotype are NA12885, NA12886, NA12879. Family tree:
Siblings: NA12879, NA12880, NA12881, NA12882, NA12883, NA12884, NA12885, NA12886, NA12887, NA12888, NA12893
Parents: NA12877, NA12878
Parents of NA12877: NA12889, NA12890
Parents of NA12878: NA12891, NA12892

Think about which steps are necessary to produce this analysis and generate a plan before starting.
Install all necessary tools and packages to find variants responsible for this pathology.
The dataset files are provided in the ./data/ directory.
Provide the output processing and results in the ./outputs/ directory.
Output the variant responsible for the pathology.
"""