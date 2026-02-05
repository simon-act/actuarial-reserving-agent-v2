import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.orchestrator import Orchestrator

def main():
    """
    Main entry point for the Multi-Agent Reserving System.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Reserving System")
    parser.add_argument("--request", type=str, default="full analysis with stress tests", 
                        help="Natural language request for the analysis")
    
    args = parser.parse_args()
    
    # Run Orchestrator
    try:
        orch = Orchestrator()
        result = orch.run_workflow(args.request)
        
        # Save output
        output_dir = Path("outputs/agent_runs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "final_report.md", "w") as f:
            f.write(result["report"])
            
        print(f"\n📄 Report saved to: {output_dir}/final_report.md")
        
        # Print preview
        print("\n" + "="*80)
        print("FINAL REPORT PREVIEW")
        print("="*80)
        print(result["report"])
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
