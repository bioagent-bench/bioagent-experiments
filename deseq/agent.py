from smolagents import CodeAgent
from models import create_azure_model
import logging
from tools import bioinformatics_tools

logger = logging.getLogger(__name__)

prompt = """The dataset consists of RNA-Seq samples from Candida parapsilosis wild-type (WT) 
strains grown in planktonic and biofilm conditions, generated as part of a study on gene 
expression and biofilm formation. The samples were sequenced on the Illumina 
HiSeq 2000 platform. The goal of this analysis is to perform differential expression analysis to 
identify genes that are significantly up- or down-regulated between planktonic and biofilm 
conditions, providing insights into biofilm-associated transcriptional changes.
Think about which steps are necessary to produce this analysis and generate a plan before starting.
The dataset files are provided in the ./data/ directory.
Provide the output processing and results in the ./outputs directory, for each separate step of
analysis create an output subdirectory and name them in order for example step_1, step_2, etc...
The final output is a list of differentialy expressed genes between planktonic and biofilm
conditions.
"""


model = create_azure_model()
bioagent = CodeAgent(
    name="bioagent",
    max_steps=30,
    model=model,
    tools=[bioinformatics_tools],
    planning_interval=1,
    add_base_tools=True,
    additional_authorized_imports=["*"],
    executor_type="local",
)
result = bioagent.run(prompt)
print(result)
