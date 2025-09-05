from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
import os
import json
from datetime import datetime
from pathlib import Path

from src.docker_sandbox import DockerSandbox
from src.dataset import DataSet

register()
SmolagentsInstrumentor().instrument()

def create_run_directory(task_id: str) -> Path:
    """Create a timestamped run directory for experiment logs."""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(f"run-logs/{timestamp}_{task_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_run_metadata(run_dir: Path, task_id: str, task_name: str, task_description: str, task_prompt: str):
    """Save metadata about the experiment run."""
    metadata = {
        "task_id": task_id,
        "task_name": task_name,
        "task_description": task_description,
        "task_prompt": task_prompt,
        "timestamp": datetime.now().isoformat(),
        "status": "started"
    }
    
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

def run_single_experiment(dataset: DataSet, run_dir: Path) -> dict:
    """Run a single experiment for a dataset."""
    print(f"Processing task: {dataset.task_id} at {dataset.path}")
    
    # Create sandbox
    sandbox = DockerSandbox()
    
    try:
        # Simple test agent code to verify mounts are working
        agent_code = """
import os
import sys
print("=== Python path ===")
print(sys.executable)
print("\\n=== Current working directory ===")
print(os.getcwd())
print("\\n=== Contents of /workspace ===")
try:
    for item in os.listdir('/workspace'):
        print(f"- {item}")
except Exception as e:
    print(f"Error listing workspace: {e}")

print("\\n=== Contents of /workspace/data (if exists) ===")
try:
    if os.path.exists('/workspace/data'):
        for item in os.listdir('/workspace/data'):
            print(f"- data/{item}")
    else:
        print("No data directory found")
except Exception as e:
    print(f"Error listing data: {e}")

print("\\n=== Contents of /workspace/reference (if exists) ===")
try:
    if os.path.exists('/workspace/reference'):
        for item in os.listdir('/workspace/reference'):
            print(f"- reference/{item}")
    else:
        print("No reference directory found")
except Exception as e:
    print(f"Error listing reference: {e}")

print("\\n=== Environment variables ===")
for key in ['HF_TOKEN']:
    value = os.getenv(key)
    if value:
        print(f"{key}: {'*' * min(len(value), 10)}...")
    else:
        print(f"{key}: Not set")
"""
        
        # Run the code in the sandbox with proper mounts
        result = sandbox.run_code(
            agent_code, 
            task_data_path=dataset.path,
            output_path=str(run_dir / "output")
        )
        
        # Save results
        output_file = run_dir / "agent_output.txt"
        with open(output_file, "w") as f:
            if result.output:
                f.write(result.output)
            if result.error:
                f.write(f"\n\nERROR:\n{result.error.traceback}")
        
        # Create result summary
        result_summary = {
            "success": result.error is None,
            "output_length": len(result.output) if result.output else 0,
            "error": result.error.traceback if result.error else None
        }
        
        print(f"Result for {dataset.task_id}:")
        print(f"  Success: {result_summary['success']}")
        if result.output:
            print(f"  Output preview: {result.output[:200]}...")
        if result.error:
            print(f"  Error: {result.error.traceback}")
        
        return result_summary
        
    finally:
        # Always cleanup the sandbox
        sandbox.cleanup()

def main():
    """Main experiment runner."""
    print("Starting bioagent experiments...")
    
    # Load all datasets
    datasets = DataSet.load_all()
    print(f"Found {len(datasets)} datasets to process")
    
    # Create overall run logs directory
    Path("run-logs").mkdir(exist_ok=True)
    
    results = {}
    
    for dataset in datasets:
        if dataset.path is None:
            print(f"Skipping {dataset.task_id} - no data path available")
            continue
            
        # Create run directory for this experiment
        run_dir = create_run_directory(dataset.task_id)
        
        # Save metadata
        save_run_metadata(run_dir, dataset.task_id, dataset.name, dataset.description, dataset.task_prompt)
        
        # Run the experiment
        try:
            result = run_single_experiment(dataset, run_dir)
            results[dataset.task_id] = result
            
            # Update metadata with final status
            metadata_file = run_dir / "run_meta.json"
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            metadata["status"] = "completed" if result["success"] else "failed"
            metadata["result_summary"] = result
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Failed to run experiment for {dataset.task_id}: {e}")
            results[dataset.task_id] = {"success": False, "error": str(e)}
    
    # Save overall results
    overall_results_file = Path("run-logs") / "overall_results.json"
    with open(overall_results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(datasets),
            "successful_experiments": sum(1 for r in results.values() if r.get("success", False)),
            "results": results
        }, f, indent=2)
    
    print(f"\nExperiments completed. Results saved to {overall_results_file}")
    print(f"Individual run logs available in run-logs/ directory")

if __name__ == "__main__":
    main()

