"""
Intelligent Methodology Agent
=============================

Transforms the MethodologyAgent into an intelligent agent that uses LLM reasoning
to decide the analysis strategy instead of hardcoded rules.

The agent:
- INTERPRETS complex user requests using LLM
- DECIDES the optimal analysis configuration
- EXPLAINS why this configuration was chosen
- CONSIDERS context and requirements
"""

import json
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from agents.intelligent_base import IntelligentAgent, ConfidenceLevel, AgentThought
from agents.schemas import ReservingConfigFile, AnalysisType, AgentRole, AgentLog


@dataclass
class MethodologyDecision:
    """Decision made by the MethodologyAgent."""

    analysis_type: AnalysisType
    run_model_selection: bool
    run_cross_validation: bool
    run_bootstrap: bool
    n_bootstrap_simulations: int
    run_diagnostics: bool
    run_stress_testing: bool
    reasoning: str
    confidence: ConfidenceLevel


class MethodologyAgent(IntelligentAgent):
    """
    INTELLIGENT SENIOR ACTUARY (Methodology)

    Uses LLM reasoning to:
    - Interpret complex user requests
    - Decide on analysis strategy and assumptions
    - Explain the methodology choices
    - Output configuration for the execution agent
    """

    def __init__(self):
        super().__init__(
            role=AgentRole.METHODOLOGY, name="Intelligent_Methodology_Agent"
        )

    def _get_system_prompt(self) -> str:
        return """You are a SENIOR ACTUARIAL METHODOLOGIST with expertise in:
- Loss reserving strategy and planning
- Statistical method selection
- Risk assessment and conservatism levels
- Computational trade-offs (speed vs accuracy)
- Regulatory and business requirements

YOUR ROLE:
Interpret the user's request and decide the optimal analysis configuration.
Consider:
1. User intent (what they really need)
2. Data characteristics (if mentioned)
3. Time constraints (quick vs thorough)
4. Risk profile (conservative vs best estimate)
5. Business context (pricing, reserving, forecasting)

CONFIGURATION OPTIONS:
- analysis_type: "quick" (2 min), "standard" (5 min), "full" (10+ min)
- run_model_selection: Compare different factor estimators (yes/no)
- run_cross_validation: Validate with hold-out periods (yes/no)
- run_bootstrap: Simulation-based uncertainty (yes/no)
- n_bootstrap_simulations: 500 (standard) or 1000 (thorough)
- run_diagnostics: Model quality tests (yes/no)
- run_stress_testing: Scenario analysis (yes/no)

RULES:
- Always explain your reasoning
- Consider the trade-off between speed and accuracy
- Match the analysis complexity to the business need
- Conservative analyses need more methods and simulations"""

    def _format_data_for_analysis(self, data: Dict) -> str:
        """Format the user request for LLM analysis."""
        return f"""USER REQUEST: {data.get("request", "Standard analysis")}

CONTEXT:
- Has existing results: {data.get("has_context", False)}
- Previous analysis type: {data.get("previous_type", "None")}

DECIDE the optimal configuration and explain WHY."""

    def plan_analysis(
        self,
        request_type: str = "standard",
        has_context: bool = False,
        previous_type: Optional[str] = None,
    ) -> Tuple[ReservingConfigFile, AgentLog]:
        """
        Use LLM reasoning to decide the analysis strategy.

        Args:
            request_type: User's natural language request
            has_context: Whether there's existing analysis context
            previous_type: Type of previous analysis if any

        Returns:
            Tuple of (config, log)
        """
        self._log(f"🧠 Planning analysis for request: '{request_type}'")

        # Prepare data for LLM
        data = {
            "request": request_type,
            "has_context": has_context,
            "previous_type": previous_type,
        }

        # Use LLM to reason about the request
        if self.llm.is_available():
            try:
                thought = self.think(
                    data=data,
                    options=["quick", "standard", "full"],
                    task="Decide analysis methodology",
                    context="Choose based on user intent, time constraints, and risk requirements",
                )

                decision = self._parse_llm_decision(thought)

                config = ReservingConfigFile(
                    analysis_type=decision.analysis_type,
                    run_model_selection=decision.run_model_selection,
                    run_cross_validation=decision.run_cross_validation,
                    run_bootstrap=decision.run_bootstrap,
                    n_bootstrap_simulations=decision.n_bootstrap_simulations,
                    run_diagnostics=decision.run_diagnostics,
                    run_stress_testing=decision.run_stress_testing,
                )

                reasoning = decision.reasoning

                log = AgentLog(
                    agent=self.role,
                    action="Intelligent Plan Analysis",
                    details=f"Selected {decision.analysis_type.value} strategy. Confidence: {decision.confidence.value}. Reasoning: {reasoning[:200]}...",
                )

                self._log(f"📝 Plan created via LLM: {reasoning[:100]}...")
                return config, log

            except Exception as e:
                self._log(f"⚠️ LLM reasoning failed: {e}. Using fallback.")
                return self._fallback_plan(request_type)
        else:
            self._log("⚠️ LLM not available. Using fallback rules.")
            return self._fallback_plan(request_type)

    def _parse_llm_decision(self, thought: AgentThought) -> MethodologyDecision:
        """Parse the LLM's thought into a structured decision."""
        # Default values
        decision = MethodologyDecision(
            analysis_type=AnalysisType.STANDARD,
            run_model_selection=True,
            run_cross_validation=False,
            run_bootstrap=True,
            n_bootstrap_simulations=500,
            run_diagnostics=True,
            run_stress_testing=False,
            reasoning="Standard analysis selected",
            confidence=ConfidenceLevel.MEDIUM,
        )

        if thought.decision:
            choice = thought.decision.choice.lower()

            # Map choice to analysis type
            if "quick" in choice or "fast" in choice:
                decision.analysis_type = AnalysisType.QUICK
                decision.run_model_selection = False
                decision.run_bootstrap = False
                decision.run_diagnostics = False
                decision.n_bootstrap_simulations = 0
            elif "full" in choice or "deep" in choice or "comprehensive" in choice:
                decision.analysis_type = AnalysisType.FULL
                decision.run_model_selection = True
                decision.run_cross_validation = True
                decision.run_bootstrap = True
                decision.n_bootstrap_simulations = 1000
                decision.run_diagnostics = True
                decision.run_stress_testing = True

            decision.reasoning = thought.decision.reasoning
            decision.confidence = thought.decision.confidence

            # Parse additional settings from evidence/risks if present
            if thought.decision.evidence:
                evidence_text = " ".join(thought.decision.evidence).lower()
                if "stress" in evidence_text or "scenario" in evidence_text:
                    decision.run_stress_testing = True
                if "cross-validation" in evidence_text or "validation" in evidence_text:
                    decision.run_cross_validation = True
                if "2000" in evidence_text or "thorough" in evidence_text:
                    decision.n_bootstrap_simulations = 1000

        return decision

    def plan_analysis_stream(
        self,
        request_type: str = "standard",
        has_context: bool = False,
        previous_type: Optional[str] = None,
    ):
        """
        Streaming version that yields thought process in real-time.

        Yields:
            Dict with 'phase', 'status', 'content', 'data'
        """
        self._log(f"🧠 Starting intelligent planning for: '{request_type}'")

        # Phase 1: Analysis
        yield {
            "phase": "analysis",
            "status": "running",
            "content": "🔍 Interpreting user request and context...",
            "data": None,
        }

        data = {
            "request": request_type,
            "has_context": has_context,
            "previous_type": previous_type,
        }

        if not self.llm.is_available():
            yield {
                "phase": "analysis",
                "status": "complete",
                "content": "⚠️ LLM not available, using fallback rules",
                "data": None,
            }
            config, log = self._fallback_plan(request_type)
            yield {
                "phase": "complete",
                "status": "complete",
                "content": f"📋 Plan: {config.analysis_type.value} analysis (fallback mode)",
                "data": {"config": config, "log": log},
            }
            return

        try:
            # Stream the thinking process
            for thought_update in self.think_stream(
                data=data,
                options=["quick", "standard", "full"],
                task="Decide analysis methodology",
                context="Choose based on user intent, time constraints, and risk requirements",
            ):
                phase_name = {
                    "analysis": "interpretation",
                    "decision": "strategy_selection",
                    "critique": "validation",
                }.get(thought_update["phase"], thought_update["phase"])

                yield {
                    "phase": f"methodology_{phase_name}",
                    "status": thought_update["status"],
                    "content": thought_update["content"],
                    "data": thought_update.get("data"),
                }

            # Get the final thought
            if self.history:
                final_thought = self.history[-1]
                decision = self._parse_llm_decision(final_thought)

                config = ReservingConfigFile(
                    analysis_type=decision.analysis_type,
                    run_model_selection=decision.run_model_selection,
                    run_cross_validation=decision.run_cross_validation,
                    run_bootstrap=decision.run_bootstrap,
                    n_bootstrap_simulations=decision.n_bootstrap_simulations,
                    run_diagnostics=decision.run_diagnostics,
                    run_stress_testing=decision.run_stress_testing,
                )

                log = AgentLog(
                    agent=self.role,
                    action="Intelligent Plan Analysis",
                    details=f"Selected {decision.analysis_type.value}. Confidence: {decision.confidence.value}",
                )

                yield {
                    "phase": "complete",
                    "status": "complete",
                    "content": f"📋 Plan: {decision.analysis_type.value} analysis - {decision.reasoning[:100]}...",
                    "data": {
                        "config": config,
                        "log": log,
                        "decision": decision,
                        "thought": final_thought,
                    },
                }
            else:
                raise ValueError("No thought history available")

        except Exception as e:
            yield {
                "phase": "error",
                "status": "error",
                "content": f"⚠️ LLM reasoning failed: {e}. Using fallback.",
                "data": None,
            }
            config, log = self._fallback_plan(request_type)
            yield {
                "phase": "complete",
                "status": "complete",
                "content": f"📋 Plan: {config.analysis_type.value} analysis (fallback mode)",
                "data": {"config": config, "log": log},
            }

    def _fallback_plan(self, request_type: str) -> Tuple[ReservingConfigFile, AgentLog]:
        """Fallback to simple rules when LLM is unavailable."""
        request_lower = request_type.lower()

        if "quick" in request_lower:
            analysis_type = AnalysisType.QUICK
            config = ReservingConfigFile(
                analysis_type=analysis_type,
                run_model_selection=False,
                run_bootstrap=False,
                run_diagnostics=False,
            )
            reasoning = "Quick estimate requested. Only Chain Ladder selected."

        elif "stress" in request_lower or "full" in request_lower:
            analysis_type = AnalysisType.FULL
            config = ReservingConfigFile(
                analysis_type=analysis_type,
                run_model_selection=True,
                run_cross_validation=True,
                run_bootstrap=True,
                n_bootstrap_simulations=1000,
                run_diagnostics=True,
                run_stress_testing=True,
            )
            reasoning = "Full deep dive requested. Enabling all features."

        else:
            analysis_type = AnalysisType.STANDARD
            config = ReservingConfigFile(
                analysis_type=analysis_type,
                run_model_selection=True,
                run_bootstrap=True,
                run_diagnostics=True,
            )
            reasoning = "Standard analysis. Chain Ladder + Mack + Bootstrap + Basic Diagnostics."

        log = AgentLog(
            agent=self.role,
            action="Plan Analysis (Fallback)",
            details=f"Selected {analysis_type.value} strategy. Reasoning: {reasoning}",
        )

        self._log(f"📝 Plan created (fallback): {reasoning}")
        return config, log
