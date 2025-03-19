from smolagents import CodeAgent
from models import create_azure_model
import logging
from tools import bioinformatics_tools

logger = logging.getLogger(__name__)

prompt = """The experiment follows a similar strategy as in what is called an
 “experimental evolution” experiment. The final aim is to identify the genome 
 variations in evolved lines of E. coli. The data is composed of a single ancestor line and 
 two evolved lines. The data is from a paired-end sequencing run data from an 
 Illumina HiSeq. This data has been post-processed in two ways already. All sequences that were 
 identified as belonging to the PhiX genome have been removed. Illumina adapters have been removed 
 as well already.
Think about which steps are necessary to produce this analysis and generate a plan before starting.
The dataset files are provided in the ./data/ directory.
Provide the output processing and results in the ./outputs directory, for each separate step of
analysis create an output subdirectory and name them in order for example step_1, step_2, etc...
The final output is a list of differentialy expressed genes between planktonic and biofilm
conditions.
The goal is to find to find and annotate the genome variations in the evolved lines of E.coli. 
Only output those variants which are shared across both evolved lines.
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
