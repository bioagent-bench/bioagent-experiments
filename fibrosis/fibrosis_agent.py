from smolagents import CodeAgent
from models import create_azure_model
import logging

logger = logging.getLogger(__name__)

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
The dataset files are provided in the ./fibrosis/data/ directory.
Provide the output processing and results in the ./fibrosis/outputs/ directory.
Install all necessary tools and packages, execute the required tools to obtain the variant that explains
the phenotype. You can wrap command line tool calls with Python code like so:
result = subprocess.run(
    ["fastqc", input_file, "-o", output_dir], capture_output=True, text=True
)
return {
    "stdout": result.stdout,
    "stderr": result.stderr,
    "returncode": result.returncode,
}
You can provide sudo password like so if you need it
echo 5hygs5nf | sudo -S apt-get install package_name
"""


def run_black_box_fibrosis_agent():
    model = create_azure_model()
    bioagent = CodeAgent(
        name="bioagent",
        max_steps=50,
        model=model,
        tools=[],
        planning_interval=3,
        add_base_tools=True,
        additional_authorized_imports=["*"],
        executor_type="local",
    )

    model = create_azure_model()
    result = bioagent.run(prompt)
    print(result)
