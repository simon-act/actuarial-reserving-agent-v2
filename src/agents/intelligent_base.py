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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional

from agents.llm_utils import LLMClient
from agents.schemas import AgentRole, AgentLog


# =====================================================
# Core Types
# =====================================================


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class Analysis:
    observations: List[str]
    patterns: List[str]
    anomalies: List[str]
    confidence: ConfidenceLevel
    reasoning: str


@dataclass
class Decision:
    choice: str
    reasoning: str
    evidence: List[str]
    alternatives: List[str]
    risks: List[str]
    confidence: ConfidenceLevel
    reconsider: List[str]


@dataclass
class Critique:
    weaknesses: List[str]
    alternatives: List[str]
    data_needed: List[str]
    confidence: ConfidenceLevel
    recommendation: str


@dataclass
class AgentThought:
    task: str
    analysis: Analysis
    decision: Decision
    critique: Critique
    timestamp: datetime = field(default_factory=datetime.now)


# =====================================================
# Base Agent
# =====================================================


class IntelligentAgent(ABC):
    def __init__(self, role: AgentRole, name: str = None):
        self.role = role
        self.name = name or f"{role.value}_Agent"

        self.llm = LLMClient()
        self.history: List[AgentThought] = []

    # -------------------------------------------------
    # Interface
    # -------------------------------------------------

    @abstractmethod
    def _get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def _format_data_for_analysis(self, data: Dict) -> str:
        pass

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------

    def _log(self, msg):
        print(f"[{self.role.value}] {msg}")

    # =================================================
    # Public API
    # =================================================

    def think(
        self,
        data: Dict,
        options: List[str],
        task: str = "",
        focus: str = None,
        context: str = "",
    ) -> AgentThought:
        self._log(f"Thinking: {task}")

        analysis = self._analyze(data, focus)
        decision = self._decide(analysis, options, context)
        critique = self._critique(decision, data)

        thought = AgentThought(
            task=task, analysis=analysis, decision=decision, critique=critique
        )

        self.history.append(thought)

        return thought

    # Public API methods for backward compatibility with tests
    def analyze(self, data: Dict, focus: str = None) -> Analysis:
        """Public wrapper for _analyze."""
        return self._analyze(data, focus)

    def decide(
        self, analysis: Analysis, options: List[str], context: str = ""
    ) -> Decision:
        """Public wrapper for _decide."""
        return self._decide(analysis, options, context)

    # =================================================
    # LLM Steps
    # =================================================

    def _analyze(self, data: Dict, focus: str = None) -> Analysis:
        if not self.llm.is_available():
            return self._fallback_analysis()

        payload = {"data": self._format_data_for_analysis(data), "focus": focus}

        prompt = f"""
{self._get_system_prompt()}

Analyze the following data.

{json.dumps(payload, indent=2)}

Return JSON:
{{
 "observations": [],
 "patterns": [],
 "anomalies": [],
 "confidence": "high|medium|low|uncertain",
 "reasoning": ""
}}
"""

        res = self._call_llm(prompt)

        return Analysis(
            observations=res.get("observations", []),
            patterns=res.get("patterns", []),
            anomalies=res.get("anomalies", []),
            confidence=ConfidenceLevel(res.get("confidence", "uncertain")),
            reasoning=res.get("reasoning", ""),
        )

    # -------------------------------------------------

    def _decide(self, analysis: Analysis, options: List[str], context: str) -> Decision:
        if not self.llm.is_available():
            return self._fallback_decision(options)

        payload = {
            "analysis": analysis.__dict__,
            "options": options,
            "context": context,
        }

        prompt = f"""
{self._get_system_prompt()}

Make a decision.

{json.dumps(payload, indent=2)}

Return JSON:
{{
 "choice": "",
 "reasoning": "",
 "evidence": [],
 "alternatives": [],
 "risks": [],
 "confidence": "high|medium|low|uncertain",
 "reconsider": []
}}
"""

        res = self._call_llm(prompt)

        return Decision(
            choice=res.get("choice"),
            reasoning=res.get("reasoning", ""),
            evidence=res.get("evidence", []),
            alternatives=res.get("alternatives", []),
            risks=res.get("risks", []),
            confidence=ConfidenceLevel(res.get("confidence", "uncertain")),
            reconsider=res.get("reconsider", []),
        )

    # -------------------------------------------------

    def _critique(self, decision: Decision, data: Dict) -> Critique:
        if not self.llm.is_available():
            return self._fallback_critique(decision)

        payload = {
            "decision": decision.__dict__,
            "data": self._format_data_for_analysis(data),
        }

        prompt = f"""
{self._get_system_prompt()}

Critique the decision.

{json.dumps(payload, indent=2)}

Return JSON:
{{
 "weaknesses": [],
 "alternatives": [],
 "data_needed": [],
 "confidence": "high|medium|low|uncertain",
 "recommendation": ""
}}
"""

        res = self._call_llm(prompt)

        return Critique(
            weaknesses=res.get("weaknesses", []),
            alternatives=res.get("alternatives", []),
            data_needed=res.get("data_needed", []),
            confidence=ConfidenceLevel(res.get("confidence", "uncertain")),
            recommendation=res.get("recommendation", ""),
        )

    # =================================================
    # Utilities
    # =================================================

    def _call_llm(self, prompt: str) -> Dict:
        txt = self.llm.get_completion(
            system_prompt="Always reply in valid JSON.", user_prompt=prompt
        )

        return self._parse_json(txt)

    # -------------------------------------------------

    def _parse_json(self, txt: str) -> Dict:
        s = txt.find("{")
        e = txt.rfind("}") + 1

        return json.loads(txt[s:e])

    # =================================================
    # Fallbacks
    # =================================================

    def _fallback_analysis(self) -> Analysis:
        return Analysis(
            observations=["LLM unavailable"],
            patterns=[],
            anomalies=[],
            confidence=ConfidenceLevel.UNCERTAIN,
            reasoning="",
        )

    # -------------------------------------------------

    def _fallback_decision(self, options) -> Decision:
        return Decision(
            choice=options[0] if options else "unknown",
            reasoning="Fallback mode",
            evidence=[],
            alternatives=[],
            risks=["No LLM reasoning"],
            confidence=ConfidenceLevel.UNCERTAIN,
            reconsider=["LLM available"],
        )

    # -------------------------------------------------

    def _fallback_critique(self, decision) -> Critique:
        return Critique(
            weaknesses=["No critique available"],
            alternatives=[],
            data_needed=["LLM access"],
            confidence=ConfidenceLevel.UNCERTAIN,
            recommendation=f"Review {decision.choice}",
        )

    # =================================================
    # Logging
    # =================================================

    def get_log(self) -> AgentLog:
        if not self.history:
            return AgentLog(agent=self.role, action="idle", details="")

        t = self.history[-1]

        return AgentLog(agent=self.role, action=t.task, details=t.decision.choice)
