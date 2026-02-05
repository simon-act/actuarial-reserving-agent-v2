from agents.schemas import ReservingInput, AgentRole, AgentLog
from agents.methodology import MethodologyAgent
from agents.reserving import ReservingExecutionAgent
from agents.validation import ValidationAgent
from agents.reporting import ReportingAgent
from agents.qa import QASpecialistAgent
from agents.intelligent_selection import IntelligentSelectionAgent
from agents.code_agent import CodeAgent
from agents.llm_utils import LLMClient

import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationMessage:
    """Single message in conversation history."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[str] = None


class ConversationMemory:
    """
    Conversation buffer for context continuity.

    Enables:
    - "e per il 2022?" (references previous context)
    - "mostrami di più" (continuation)
    - "perché?" (follow-up)
    """

    def __init__(self, max_messages: int = 20):
        self.messages: List[ConversationMessage] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str, intent: str = None):
        """Add message to history."""
        self.messages.append(
            ConversationMessage(role=role, content=content, intent=intent)
        )
        # Trim if too long
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get_recent(self, n: int = 5) -> List[Dict]:
        """Get last n messages as dicts."""
        return [{"role": m.role, "content": m.content} for m in self.messages[-n:]]

    def get_context_string(self, n: int = 5) -> str:
        """Get recent conversation as string for LLM context."""
        recent = self.get_recent(n)
        if not recent:
            return "No previous conversation."

        lines = []
        for msg in recent:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content'][:200]}")
        return "\n".join(lines)

    def clear(self):
        """Clear history."""
        self.messages = []


class Orchestrator:
    """
    ENHANCED QUESTION-DRIVEN ORCHESTRATOR

    Improvements:
    1. Multi-label intent detection (NEW_ANALYSIS, Q_AND_A, CODE_QUERY, HYBRID)
    2. Conversation memory for context continuity
    3. CodeAgent for dynamic calculations
    4. Smart routing based on query complexity
    """

    def __init__(self):
        self.role = AgentRole.ORCHESTRATOR
        self.logs = []

        # Collaborative Team
        self.methodology = MethodologyAgent()
        self.selector = IntelligentSelectionAgent()
        self.actuary = ReservingExecutionAgent()
        self.validator = ValidationAgent()
        self.reporter = ReportingAgent()
        self.qa_specialist = QASpecialistAgent()
        self.code_agent = CodeAgent()  # NEW: Dynamic code execution

        # Brain
        self.llm = LLMClient()

        # Memory
        self.memory = ConversationMemory()

        # Cache for triangle (for code agent)
        self._triangle_cache = None

    def determine_intent(self, query: str, has_context: bool) -> Dict[str, Any]:
        """
        MULTI-LABEL intent detection.

        Returns:
            {
                "primary": "NEW_ANALYSIS" | "Q_AND_A" | "CODE_QUERY",
                "needs_code": bool,  # Requires dynamic calculation
                "is_followup": bool,  # References previous conversation
                "confidence": float
            }
        """
        result = {
            "primary": "Q_AND_A",
            "needs_code": False,
            "is_followup": False,
            "confidence": 0.5,
        }

        query_lower = query.lower()

        # Check for follow-up patterns
        followup_patterns = [
            "e per",
            "e il",
            "e la",
            "e nel",
            "invece",
            "perché",
            "why",
            "come mai",
            "mostra",
            "show me",
            "più dettagli",
            "more detail",
            "spiega",
            "explain",
            "quello",
            "questo",
            "that",
            "this",
            "the same",
        ]
        result["is_followup"] = any(p in query_lower for p in followup_patterns)

        # Check for code-requiring patterns (complex calculations)
        code_patterns = [
            "correlazione",
            "correlation",
            "covarianza",
            "covariance",
            "trend",
            "regressione",
            "regression",
            "forecast",
            "predict",
            "distribuzione",
            "distribution",
            "histogram",
            "percentile",
            "somma",
            "sum",
            "media",
            "mean",
            "average",
            "std",
            "deviazione",
            "massimo",
            "max",
            "minimo",
            "min",
            "range",
            "confronta",
            "compare",
            "differenza",
            "difference",
            "filtra",
            "filter",
            "where",
            "maggiore",
            "greater",
            "less",
            "calcola",
            "calculate",
            "compute",
            "quanto",
            "how much",
        ]
        result["needs_code"] = any(p in query_lower for p in code_patterns)

        # Check for new analysis patterns
        analysis_patterns = [
            "analizza",
            "analyze",
            "run",
            "esegui",
            "calcola riserve",
            "calculate reserves",
            "update",
            "aggiorna",
            "nuovo",
            "new analysis",
            "full analysis",
            "stress test",
            "bootstrap",
        ]

        if not self.llm.is_available():
            # Fallback heuristic
            if any(k in query_lower for k in analysis_patterns) or not has_context:
                result["primary"] = "NEW_ANALYSIS"
                result["confidence"] = 0.7
            elif result["needs_code"] and has_context:
                result["primary"] = "CODE_QUERY"
                result["confidence"] = 0.8
            else:
                result["primary"] = "Q_AND_A"
                result["confidence"] = 0.6
            return result

        # Use LLM for better classification
        conversation_context = self.memory.get_context_string(3)

        system_prompt = """You are the Brain of an Actuarial Orchestrator.
Classify the User Query into ONE of these intents:

1. 'NEW_ANALYSIS': User wants to run a new calculation or analysis (e.g., "Analyze 2024", "Run stress test")
2. 'Q_AND_A': User asks about existing results or explanations (e.g., "What is the IBNR?", "Why is it high?")
3. 'CODE_QUERY': User asks a complex question that requires dynamic calculation (e.g., "What's the correlation between factors?", "Calculate the mean reserve")

Also determine:
- needs_code: true if the answer requires computing something not pre-calculated
- is_followup: true if the query references previous conversation

Output JSON only:
{"primary": "...", "needs_code": true/false, "is_followup": true/false, "confidence": 0.0-1.0}"""

        user_prompt = f"""Recent conversation:
{conversation_context}

Current query: {query}
Has existing context: {has_context}

Classify this query."""

        try:
            response = self.llm.get_completion(system_prompt, user_prompt)

            # Parse JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                result.update(parsed)

            # Override: force NEW_ANALYSIS if no context and not a general question
            if not has_context and result["primary"] != "Q_AND_A":
                result["primary"] = "NEW_ANALYSIS"

        except Exception as e:
            print(f"[Orchestrator] Intent detection error: {e}")

        return result

    def route_request(
        self,
        message: str,
        current_result: dict = None,
        inputs: ReservingInput = ReservingInput(),
    ):
        """
        Unified entry point for the UI.
        Enhanced with multi-label routing and code execution.
        """
        has_context = current_result is not None

        # Save to memory
        self.memory.add("user", message)

        # 1. Decide Intent (multi-label)
        yield {
            "step": "router",
            "status": "running",
            "message": "🤔 Analyzing query...",
        }
        intent = self.determine_intent(message, has_context)

        primary = intent["primary"]
        needs_code = intent.get("needs_code", False)
        is_followup = intent.get("is_followup", False)

        # Log intent
        intent_msg = f"👉 Intent: {primary}"
        if needs_code:
            intent_msg += " (needs code)"
        if is_followup:
            intent_msg += " (follow-up)"

        yield {"step": "router", "status": "done", "message": intent_msg}

        # 2. Route based on intent
        if primary == "NEW_ANALYSIS" or (not has_context and primary != "Q_AND_A"):
            # Run full analysis pipeline
            final_res = None
            for update in self.stream_workflow(message, inputs):
                yield update
                if update["step"] == "complete":
                    final_res = update["result"]

            # Auto-followup synthesis
            if final_res:
                yield {
                    "step": "final_synthesis",
                    "status": "running",
                    "message": "✍️ Synthesizing answer...",
                }

                synthesis, log = self.qa_specialist.answer_query(
                    f"Summarize the results for: {message}",
                    final_res,
                    conversation_history=self.memory.get_recent(3),
                )
                self.memory.add("assistant", synthesis, intent=primary)
                yield {"step": "final_synthesis", "status": "done", "data": synthesis}

        elif primary == "CODE_QUERY" and has_context:
            # Use CodeAgent for dynamic calculation
            for update in self.execute_code_query(message, current_result):
                yield update

        else:  # Q_AND_A
            # Standard Q&A with memory context
            # Ensure context is a dict, not None
            safe_context = current_result if current_result is not None else {}
            for update in self.ask_question(message, safe_context):
                yield update

    def execute_code_query(self, query: str, context: dict):
        """Execute dynamic code to answer complex questions."""
        yield {
            "step": "code",
            "status": "running",
            "message": "🐍 Generating and executing code...",
        }

        structured = context.get("structured_results")
        if not structured:
            yield {
                "step": "code",
                "status": "done",
                "data": "No analysis results available for code execution.",
            }
            return

        # Execute code
        response, log = self.code_agent.answer_with_code(
            query, structured, triangle_df=self._triangle_cache
        )

        self.logs.append(log)
        self.memory.add("assistant", response, intent="CODE_QUERY")

        yield {"step": "code", "status": "done", "data": response, "log": log}

    def stream_workflow(self, request: str, inputs: ReservingInput = ReservingInput()):
        """Generator that yields updates for the UI."""
        import pandas as pd
        from pathlib import Path

        self.logs = []

        # 1. Methodology
        yield {
            "step": "methodology",
            "status": "running",
            "message": "Analyzing request...",
        }
        config, log1 = self.methodology.plan_analysis(request)
        self.logs.append(log1)
        yield {"step": "methodology", "status": "done", "data": config, "log": log1}

        # 2. LLM Method Selection (Intelligent) + One-Shot Validation Feedback
        yield {
            "step": "selection",
            "status": "running",
            "message": "🧠 Intelligent agent is selecting optimal method...",
        }
        try:
            # Load data for selection
            triangle = pd.read_csv(inputs.triangle_path, index_col=0)
            self._triangle_cache = triangle  # Cache for CodeAgent

            premium = None
            if inputs.premium_path:
                premium = pd.read_csv(inputs.premium_path, index_col=0).iloc[:, 0]

            # Stream the thinking process
            initial_selection = None
            try:
                for thought_update in self.selector.analyze_and_select_stream(
                    triangle, verbose=True
                ):
                    # Yield the thinking process to UI
                    yield {
                        "step": f"selection_thought_{thought_update['phase']}",
                        "status": thought_update["status"],
                        "message": thought_update["content"],
                        "thought_data": thought_update.get("data"),
                    }
                    # Keep the final result
                    if thought_update["phase"] == "complete":
                        initial_selection = thought_update["data"]["result"]
            except Exception as e:
                # Fallback to non-streaming version
                yield {
                    "step": "selection_thought_fallback",
                    "status": "running",
                    "message": f"⚠️ Streaming failed, using standard method...",
                }
                initial_selection = self.selector.analyze_and_select(
                    triangle, verbose=True
                )

            # One-shot peer review (no loop)
            pattern_analysis = {
                "smoothing_applied": initial_selection.pattern_analysis.smoothing_applied,
                "smoothing_method": initial_selection.pattern_analysis.smoothing_method,
                "smoothing_weight": initial_selection.pattern_analysis.smoothing_weight,
            }

            feedback = self.validator.review_selection(
                selection_result={
                    "selected_estimator": initial_selection.selected_estimator,
                    "confidence": initial_selection.confidence.value
                    if initial_selection.confidence
                    else "uncertain",
                    "reasoning": (
                        initial_selection.thought_process.decision.reasoning
                        if initial_selection.thought_process
                        and initial_selection.thought_process.decision
                        else initial_selection.explanation
                    ),
                },
                estimator_results=initial_selection.all_estimator_results,
                diagnostics=initial_selection.diagnostics_summary,
                pattern_analysis=pattern_analysis,
                verbose=True,
            )

            final_selection = self.selector.consider_validation_feedback(
                initial_selection, feedback, verbose=True
            )

            selection_result = {
                "selected_estimator": final_selection.selected_estimator,
                "estimator_reason": (
                    final_selection.thought_process.decision.reasoning
                    if final_selection.thought_process
                    and final_selection.thought_process.decision
                    else final_selection.explanation
                ),
                "bf_years": [],
                "bf_reason": "",
                "summary": final_selection.explanation,
                "all_estimators": final_selection.all_estimator_results,
                "maturity_by_year": {},
                "validation_metrics": final_selection.validation_metrics.get(
                    "detailed_metrics", {}
                ),
                "pattern_analysis": pattern_analysis,
                "prudential_adjustments": final_selection.prudential_adjustments,
                "adjusted_factors": {
                    str(k): round(float(v), 4)
                    for k, v in final_selection.adjusted_factors.items()
                },
            }

            log_sel = AgentLog(
                agent=AgentRole.METHODOLOGY,
                action="Intelligent Method Selection",
                details=f"Selected: {selection_result.get('selected_estimator', 'Unknown')}",
            )
            self.logs.append(log_sel)

            yield {
                "step": "selection",
                "status": "done",
                "data": selection_result,
                "log": log_sel,
            }
        except Exception as e:
            selection_result = {
                "selected_estimator": "Volume-Weighted",
                "estimator_reason": "Default (selection failed)",
                "bf_years": [],
                "bf_reason": "",
                "summary": f"Using default method due to error: {e}",
            }
            yield {"step": "selection", "status": "done", "data": selection_result}

        # 3. Execution
        yield {
            "step": "execution",
            "status": "running",
            "message": "Actuary is running models...",
        }

        # Use adjusted_factors if available, otherwise use default factors
        selected_factors = None
        try:
            selected_factors = final_selection.adjusted_factors
        except NameError:
            # final_selection not defined due to error in selection - use default
            pass

        results = self.actuary.execute(
            inputs, config, selected_factors=selected_factors
        )

        # Attach selection results
        from agents.schemas import MethodSelection

        results.method_selection = MethodSelection(
            selected_estimator=selection_result.get("selected_estimator", "Unknown"),
            estimator_reason=selection_result.get("estimator_reason", ""),
            bf_years=selection_result.get("bf_years", []),
            bf_reason=selection_result.get("bf_reason", ""),
            summary=selection_result.get("summary", ""),
            all_estimators=selection_result.get("all_estimators", {}),
            maturity_by_year=selection_result.get("maturity_by_year", {}),
            validation_metrics=selection_result.get("validation_metrics", {}),
            prudential_adjustments=selection_result.get("prudential_adjustments", []),
            adjusted_factors=selection_result.get("adjusted_factors", {}),
        )

        log2 = AgentLog(
            agent=AgentRole.EXECUTION, action="Execution", details="Calculated reserves"
        )
        self.logs.append(log2)
        yield {"step": "execution", "status": "done", "data": results, "log": log2}

        # 4. Validation
        yield {
            "step": "validation",
            "status": "running",
            "message": "Peer reviewer is checking assumptions...",
        }
        validation_report, log3 = self.validator.validate(results)
        self.logs.append(log3)
        yield {
            "step": "validation",
            "status": "done",
            "data": validation_report,
            "log": log3,
        }

        # 5. Reporting
        yield {
            "step": "reporting",
            "status": "running",
            "message": "Drafting final report...",
        }
        final_report, log4 = self.reporter.generate_report(results, validation_report)
        self.logs.append(log4)
        yield {"step": "reporting", "status": "done", "data": final_report, "log": log4}

        # Final Bundle
        yield {
            "step": "complete",
            "result": {
                "report": final_report,
                "structured_results": results,
                "validation": validation_report,
                "method_selection": selection_result,
                "audit_trail": self.logs,
            },
        }

    def ask_question(self, query: str, context: dict):
        """Enhanced Q&A with conversation memory."""
        yield {"step": "qa", "status": "running", "message": "Consulting results..."}

        # Pass conversation history for context
        response, log = self.qa_specialist.answer_query(
            query, context, conversation_history=self.memory.get_recent(5)
        )

        self.logs.append(log)
        self.memory.add("assistant", response, intent="Q_AND_A")

        yield {"step": "qa", "status": "done", "data": response, "log": log}

    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()
        print("[Orchestrator] Conversation memory cleared.")
