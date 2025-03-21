from smolagents import CodeAgent
from models import (
    create_azure_model,
    create_llama_model,
    create_claude_model,
    create_gemini_model,
)
import logging
from tools import bioinformatics_tools

logger = logging.getLogger(__name__)

prompt = """The datasets consists FASTA sequences and GFF annotations of a microbial genome 
for Micrococcus. The main goal. The goal of is to do phylogenetic reconstruction of 
clusters of orthologous co-evolving genes. The COGs needs to be filtered based on the 
following quality criteria:
1.No paralogs
2.Clusters present in all 4 organisms
3.Only present in the coding regions
4.Must have at least 1 high confidence annotation.

The final result is clustering of the co-evolving genes into functional (annotated clusters)
The dataset files are provided in the ./data directory.
Provide the output processing and results in the ./outputs directory, for each separate step of
analysis create an output subdirectory and name them in order for example step_1, step_2, etc...
"""

# model = create_azure_model()
# model = create_claude_model()
# model = create_gemini_model()
model = create_llama_model()
bioagent = CodeAgent(
    name="bioagent",
    max_steps=30,
    model=model,
    tools=bioinformatics_tools,
    planning_interval=1,
    add_base_tools=True,
    additional_authorized_imports=["*"],
    executor_type="local",
)
result = bioagent.run(prompt)
print(result)
