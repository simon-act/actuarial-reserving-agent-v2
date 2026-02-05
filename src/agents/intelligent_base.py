"""
Intelligent Agent Base Framework
================================

Base class for all intelligent agents that reason about data
instead of applying hardcoded rules.

Philosophy:
- NO hardcoded thresholds
- NO if-then rules for decisions
- ALL decisions go through LLM reasoning
- LLM sees raw data + context
- LLM explains reasoning
- LLM critiques itself
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agents.llm_utils import LLMClient
from agents.schemas import AgentRole, AgentLog


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent decisions."""
    HIGH = "high"           # >80% confident
    MEDIUM = "medium"       # 50-80% confident
    LOW = "low"             # <50% confident
    UNCERTAIN = "uncertain" # Cannot determine


@dataclass
class Analysis:
    """Result of agent's analysis of data."""
    observations: List[str]          # What the agent sees
    patterns: List[str]              # Patterns identified
    anomalies: List[str]             # Anomalies/concerns found
    confidence: ConfidenceLevel      # How confident in analysis
    raw_reasoning: str               # Full LLM reasoning


@dataclass
class Decision:
    """A decision made by an agent."""
    choice: str                      # The decision made
    reasoning: str                   # Why this decision
    evidence: List[str]              # Specific evidence cited
    alternatives_considered: List[str]  # Other options considered
    risks: List[str]                 # Risks of this decision
    confidence: ConfidenceLevel      # Confidence in decision
    conditions_to_reconsider: List[str]  # When would agent change mind


@dataclass
class Critique:
    """Self-critique of a decision."""
    weaknesses: List[str]            # Weaknesses in reasoning
    alternative_interpretations: List[str]  # Other valid views
    additional_data_needed: List[str]  # What would help confirm
    revised_confidence: ConfidenceLevel  # Confidence after critique
    final_recommendation: str        # Final recommendation


@dataclass
class AgentThought:
    """Complete thought process of an agent."""
    timestamp: datetime = field(default_factory=datetime.now)
    task: str = ""
    analysis: Optional[Analysis] = None
    decision: Optional[Decision] = None
    critique: Optional[Critique] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "task": self.task,
            "analysis": self.analysis.__dict__ if self.analysis else None,
            "decision": self.decision.__dict__ if self.decision else None,
            "critique": self.critique.__dict__ if self.critique else None
        }


class IntelligentAgent(ABC):
    """
    Base class for all intelligent agents.

    Agents that inherit from this class will:
    1. See raw data (not just summaries)
    2. Reason about data using LLM
    3. Make decisions with explanations
    4. Critique their own decisions
    5. Be transparent about uncertainty
    """

    def __init__(self, role: AgentRole, name: str = None):
        self.role = role
        self.name = name or f"{role.value}_Agent"
        self.llm = LLMClient()
        self.thought_history: List[AgentThought] = []

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for this agent's specialty."""
        pass

    @abstractmethod
    def _format_data_for_analysis(self, data: Dict) -> str:
        """Format raw data for LLM to see and analyze."""
        pass

    def _log(self, message: str):
        """Print log message with agent identifier."""
        print(f"[{self.role.value}] {message}")

    def analyze(self, data: Dict, focus: str = None) -> Analysis:
        """
        LLM sees raw data and analyzes it.

        Args:
            data: Raw data dictionary
            focus: Optional specific aspect to focus on

        Returns:
            Analysis with observations, patterns, anomalies
        """
        self._log("🔍 Analyzing data...")

        if not self.llm.is_available():
            self._log("⚠️ LLM not available, using basic analysis")
            return self._fallback_analysis(data)

        formatted_data = self._format_data_for_analysis(data)

        focus_instruction = f"\nFocus particularly on: {focus}" if focus else ""

        prompt = f"""
{self._get_system_prompt()}

=== RAW DATA ===
{formatted_data}

=== TASK: ANALYZE ==={focus_instruction}

Analyze this data thoroughly. You must:

1. **OBSERVATIONS**: What specific patterns, values, or characteristics do you see?
   - Reference actual numbers
   - Note any trends
   - Identify relationships

2. **PATTERNS**: What recurring patterns or structures exist?
   - Are they expected or unexpected?
   - How strong/consistent are they?

3. **ANOMALIES**: What concerns, outliers, or unusual aspects exist?
   - Why are they concerning?
   - Could they be real or noise?

4. **CONFIDENCE**: How confident are you in this analysis?
   - high (>80%): Clear patterns, sufficient data
   - medium (50-80%): Some uncertainty but reasonable conclusions
   - low (<50%): Significant uncertainty
   - uncertain: Cannot determine

Respond in this JSON format:
{{
    "observations": ["observation 1 with specific numbers", "observation 2", ...],
    "patterns": ["pattern 1", "pattern 2", ...],
    "anomalies": ["anomaly 1 with explanation", "anomaly 2", ...],
    "confidence": "high|medium|low|uncertain",
    "reasoning": "Your detailed reasoning process..."
}}
"""

        if not self.llm.is_available():
            self._log("⚠️ LLM not available, using basic analysis")
            return self._fallback_analysis(data)

        try:
            response = self.llm.get_completion(
                system_prompt="You are an expert analyst. Always respond in valid JSON.",
                user_prompt=prompt
            )

            # Parse JSON from response
            result = self._parse_json_response(response)

            analysis = Analysis(
                observations=result.get("observations", []),
                patterns=result.get("patterns", []),
                anomalies=result.get("anomalies", []),
                confidence=ConfidenceLevel(result.get("confidence", "uncertain")),
                raw_reasoning=result.get("reasoning", "")
            )

            self._log(f"✓ Analysis complete. Confidence: {analysis.confidence.value}")
            return analysis

        except Exception as e:
            self._log(f"✗ Analysis failed: {e}")
            return self._fallback_analysis(data)

    def decide(self, analysis: Analysis, options: List[str], context: str = "") -> Decision:
        """
        LLM makes decision based on analysis.

        Args:
            analysis: Previous analysis of data
            options: Available options to choose from
            context: Additional context for decision

        Returns:
            Decision with choice, reasoning, and risks
        """
        self._log("🧠 Making decision...")

        if not self.llm.is_available():
            self._log("⚠️ LLM not available, using fallback decision")
            return self._fallback_decision(options)

        prompt = f"""
{self._get_system_prompt()}

=== YOUR ANALYSIS ===
Observations: {json.dumps(analysis.observations)}
Patterns: {json.dumps(analysis.patterns)}
Anomalies: {json.dumps(analysis.anomalies)}
Confidence: {analysis.confidence.value}
Reasoning: {analysis.raw_reasoning}

=== ADDITIONAL CONTEXT ===
{context}

=== AVAILABLE OPTIONS ===
{json.dumps(options)}

=== TASK: DECIDE ===

Based on your analysis, make a decision. You must:

1. **CHOICE**: Which option do you choose?

2. **REASONING**: Why this choice? Cite specific evidence from your analysis.
   - Reference actual numbers/patterns you observed
   - Explain the logic

3. **EVIDENCE**: List the specific pieces of evidence supporting this choice.

4. **ALTERNATIVES**: What other options did you seriously consider? Why not those?

5. **RISKS**: What are the risks of this decision?
   - What could go wrong?
   - What assumptions are you making?

6. **CONDITIONS TO RECONSIDER**: Under what conditions would you change this decision?
   - What new information would change your mind?

7. **CONFIDENCE**: How confident are you?
   - high: Strong evidence, clear choice
   - medium: Good evidence but some uncertainty
   - low: Weak evidence, close call
   - uncertain: Cannot confidently decide

Respond in JSON format:
{{
    "choice": "your chosen option",
    "reasoning": "detailed reasoning with evidence",
    "evidence": ["evidence 1", "evidence 2", ...],
    "alternatives_considered": ["option X - rejected because...", ...],
    "risks": ["risk 1", "risk 2", ...],
    "conditions_to_reconsider": ["condition 1", "condition 2", ...],
    "confidence": "high|medium|low|uncertain"
}}
"""

        if not self.llm.is_available():
            self._log("⚠️ LLM not available, using fallback decision")
            return self._fallback_decision(options)

        try:
            response = self.llm.get_completion(
                system_prompt="You are an expert decision maker. Always respond in valid JSON.",
                user_prompt=prompt
            )

            result = self._parse_json_response(response)

            decision = Decision(
                choice=result.get("choice", options[0] if options else "unknown"),
                reasoning=result.get("reasoning", ""),
                evidence=result.get("evidence", []),
                alternatives_considered=result.get("alternatives_considered", []),
                risks=result.get("risks", []),
                confidence=ConfidenceLevel(result.get("confidence", "uncertain")),
                conditions_to_reconsider=result.get("conditions_to_reconsider", [])
            )

            self._log(f"✓ Decision: {decision.choice} (confidence: {decision.confidence.value})")
            return decision

        except Exception as e:
            self._log(f"✗ Decision failed: {e}")
            return self._fallback_decision(options)

    def critique(self, decision: Decision, original_data: Dict = None) -> Critique:
        """
        LLM critiques its own decision.

        Args:
            decision: The decision to critique
            original_data: Original data for reference

        Returns:
            Critique with weaknesses and revised confidence
        """
        self._log("🔎 Self-critiquing decision...")

        if not self.llm.is_available():
            self._log("⚠️ LLM not available, skipping critique")
            return self._fallback_critique(decision)

        data_context = ""
        if original_data:
            data_context = f"\n=== ORIGINAL DATA FOR REFERENCE ===\n{self._format_data_for_analysis(original_data)}"

        prompt = f"""
{self._get_system_prompt()}

=== THE DECISION MADE ===
Choice: {decision.choice}
Reasoning: {decision.reasoning}
Evidence cited: {json.dumps(decision.evidence)}
Risks identified: {json.dumps(decision.risks)}
Confidence: {decision.confidence.value}
{data_context}

=== TASK: CRITIQUE ===

Now critique this decision harshly but fairly. Play devil's advocate.

1. **WEAKNESSES**: What are the weaknesses in this reasoning?
   - What assumptions might be wrong?
   - What evidence was overlooked or misinterpreted?
   - What logical gaps exist?

2. **ALTERNATIVE INTERPRETATIONS**: What other valid interpretations exist?
   - Could the same evidence support a different conclusion?
   - What would someone who disagrees say?

3. **ADDITIONAL DATA NEEDED**: What additional information would help?
   - What would confirm this decision?
   - What would refute it?

4. **REVISED CONFIDENCE**: After this critique, what's your confidence?
   - Did the decision hold up to scrutiny?
   - Should confidence be raised or lowered?

5. **FINAL RECOMMENDATION**: After critique, what's your final recommendation?
   - Proceed with decision as-is?
   - Modify the decision?
   - Seek more information?

Respond in JSON format:
{{
    "weaknesses": ["weakness 1", "weakness 2", ...],
    "alternative_interpretations": ["interpretation 1", ...],
    "additional_data_needed": ["data 1", "data 2", ...],
    "revised_confidence": "high|medium|low|uncertain",
    "final_recommendation": "your final recommendation with reasoning"
}}
"""

        if not self.llm.is_available():
            self._log("⚠️ LLM not available, skipping critique")
            return self._fallback_critique(decision)

        try:
            response = self.llm.get_completion(
                system_prompt="You are a critical reviewer. Be harsh but fair. Always respond in valid JSON.",
                user_prompt=prompt
            )

            result = self._parse_json_response(response)

            critique = Critique(
                weaknesses=result.get("weaknesses", []),
                alternative_interpretations=result.get("alternative_interpretations", []),
                additional_data_needed=result.get("additional_data_needed", []),
                revised_confidence=ConfidenceLevel(result.get("revised_confidence", "uncertain")),
                final_recommendation=result.get("final_recommendation", "")
            )

            self._log(f"✓ Critique complete. Revised confidence: {critique.revised_confidence.value}")
            return critique

        except Exception as e:
            self._log(f"✗ Critique failed: {e}")
            return self._fallback_critique(decision)

    def think(self, data: Dict, options: List[str], task: str = "",
              focus: str = None, context: str = "") -> AgentThought:
        """
        Complete thinking process: analyze → decide → critique.

        Args:
            data: Raw data to analyze
            options: Available options for decision
            task: Description of the task
            focus: Specific aspect to focus on
            context: Additional context

        Returns:
            Complete AgentThought with analysis, decision, and critique
        """
        self._log(f"💭 Starting thought process for: {task}")

        thought = AgentThought(task=task)

        # Step 1: Analyze
        thought.analysis = self.analyze(data, focus=focus)

        # Step 2: Decide
        thought.decision = self.decide(thought.analysis, options, context=context)

        # Step 3: Critique
        thought.critique = self.critique(thought.decision, original_data=data)

        # Store in history
        self.thought_history.append(thought)

        self._log(f"💭 Thought process complete")
        return thought

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response, handling common issues."""
        # Try to find JSON in response
        start = response.find('{')
        end = response.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)

        # If no JSON found, try to parse entire response
        return json.loads(response)

    def _fallback_analysis(self, data: Dict) -> Analysis:
        """Fallback analysis when LLM is not available."""
        return Analysis(
            observations=["LLM not available - basic analysis only"],
            patterns=[],
            anomalies=[],
            confidence=ConfidenceLevel.UNCERTAIN,
            raw_reasoning="Fallback mode - no LLM reasoning available"
        )

    def _fallback_decision(self, options: List[str]) -> Decision:
        """Fallback decision when LLM is not available."""
        return Decision(
            choice=options[0] if options else "unknown",
            reasoning="LLM not available - defaulting to first option",
            evidence=[],
            alternatives_considered=[],
            risks=["Decision made without LLM reasoning"],
            confidence=ConfidenceLevel.UNCERTAIN,
            conditions_to_reconsider=["LLM becomes available"]
        )

    def _fallback_critique(self, decision: Decision) -> Critique:
        """Fallback critique when LLM is not available."""
        return Critique(
            weaknesses=["Could not perform LLM critique"],
            alternative_interpretations=[],
            additional_data_needed=["LLM reasoning"],
            revised_confidence=ConfidenceLevel.UNCERTAIN,
            final_recommendation=f"Proceed with {decision.choice} but verify manually"
        )

    def get_log(self) -> AgentLog:
        """Get the most recent thought as an AgentLog."""
        if not self.thought_history:
            return AgentLog(
                agent=self.role,
                action="No actions yet",
                details=""
            )

        last_thought = self.thought_history[-1]
        return AgentLog(
            agent=self.role,
            action=last_thought.task,
            details=last_thought.decision.choice if last_thought.decision else ""
        )
