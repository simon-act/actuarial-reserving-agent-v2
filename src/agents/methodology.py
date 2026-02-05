from typing import Tuple
from agents.schemas import ReservingInput, ReservingConfigFile, AnalysisType, AgentRole, AgentLog

class MethodologyAgent:
    """
    SENIOR ACTUARY (Methodology)
    - Interprets the request
    - Decides on assumptions and methods
    - outputs configuration for the execution agent
    """
    
    def __init__(self):
        self.role = AgentRole.METHODOLOGY

    def plan_analysis(self, request_type: str = "standard") -> Tuple[ReservingConfigFile, AgentLog]:
        """
        Decide the analysis strategy.
        In a real LLM scenario, this would parse a natural language request.
        Here we map string triggers to configs.
        """
        print(f"[{self.role}] 🧠 Planning analysis for request: '{request_type}'")
        
        if "quick" in request_type.lower():
            analysis_type = AnalysisType.QUICK
            config = ReservingConfigFile(
                analysis_type=analysis_type,
                run_model_selection=False,
                run_bootstrap=False,
                run_diagnostics=False
            )
            reasoning = "Quick estimate requested. Only Chain Ladder selected."
            
        elif "stress" in request_type.lower() or "full" in request_type.lower():
            analysis_type = AnalysisType.FULL
            config = ReservingConfigFile(
                analysis_type=analysis_type,
                run_model_selection=True,
                run_cross_validation=True,
                run_bootstrap=True,
                n_bootstrap_simulations=2000,
                run_diagnostics=True,
                run_stress_testing=True
            )
            reasoning = "Full deep dive requested. Enabling Stress Testing, Cross Validation and Diagnositcs."
            
        else:
            # Default / Standard
            analysis_type = AnalysisType.STANDARD
            config = ReservingConfigFile(
                analysis_type=analysis_type,
                run_model_selection=True,
                run_bootstrap=True,
                run_diagnostics=True
            )
            reasoning = "Standard analysis. Chain Ladder + Mack + Bootstrap + Basic Diagnostics."

        log = AgentLog(
            agent=self.role,
            action="Plan Analysis",
            details=f"Selected {analysis_type} strategy. Reasoning: {reasoning}"
        )
        
        print(f"[{self.role}] 📝 Plan created: {reasoning}")
        return config, log
