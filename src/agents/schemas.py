from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime

class AgentRole(str, Enum):
    ORCHESTRATOR = "Orchestrator"
    METHODOLOGY = "Methodology"
    EXECUTION = "Execution"
    VALIDATION = "Validation"
    REPORTING = "Reporting"

class AnalysisType(str, Enum):
    QUICK = "quick"             # Just Chain Ladder
    STANDARD = "standard"       # CL + Mack + Bootstrap
    FULL = "full"              # All methods including BF, Cape Cod + Stress

class ReservingInput(BaseModel):
    """Input data references for the analysis."""
    triangle_path: str = "data/processed/reported_absolute_losses.csv"
    premium_path: Optional[str] = "data/processed/earned_premium.csv"
    loss_ratios_path: Optional[str] = None
    
class ReservingConfigFile(BaseModel):
    """Configuration for the Actuary (Execution Agent)."""
    analysis_type: AnalysisType
    run_model_selection: bool = True
    run_cross_validation: bool = False
    run_bootstrap: bool = True
    n_bootstrap_simulations: int = 1000
    run_diagnostics: bool = True
    run_stress_testing: bool = False

class MethodResult(BaseModel):
    """Summary result for a specific method."""
    method_name: str
    total_reserve: float
    ultimate_loss: float
    model_params: Dict[str, Any] = {}
    
class StochasticResult(MethodResult):
    """Result for stochastic methods."""
    standard_error: float
    cv: float
    percentiles: Dict[str, float] = {}

class DiagnosticsResult(BaseModel):
    """Health check of the models."""
    adequacy_score: float
    rating: str
    issues: List[str] = []

class TriangleMetadata(BaseModel):
    """Metadata about the input triangle."""
    n_accident_years: int
    n_development_periods: int
    first_accident_year: int
    last_accident_year: int
    currency: str = "USD"
    units: str = "millions"

class DetailedData(BaseModel):
    """Complete data for LLM analysis."""
    # Triangle data: {year: {period: value}} - None for missing values
    triangle: Dict[str, Dict[str, Optional[float]]] = {}
    # Development factors by period
    development_factors: Dict[str, float] = {}
    # Reserves by accident year
    reserves_by_year: Dict[str, float] = {}
    # Ultimates by accident year
    ultimates_by_year: Dict[str, float] = {}
    # Latest diagonal (most recent values per year)
    latest_diagonal: Dict[str, float] = {}

class MethodSelection(BaseModel):
    """LLM-driven method selection results."""
    selected_estimator: str
    estimator_reason: str
    bf_years: List[str] = []
    bf_reason: str = ""
    summary: str
    # All estimator results for comparison
    all_estimators: Dict[str, float] = {}  # {name: total_reserve}
    # Maturity by year
    maturity_by_year: Dict[str, float] = {}  # {year: maturity_pct}
    # Validation metrics
    validation_metrics: Dict[str, Dict[str, Optional[float]]] = {}  # {estimator: {mse, mae}}
    # Prudential adjustments to factors
    prudential_adjustments: List[Dict[str, Any]] = []
    # Adjusted factors after prudential rules
    adjusted_factors: Dict[str, float] = {}

class ReservingOutput(BaseModel):
    """Structured output from the Reserving Execution Agent."""
    timestamp: datetime = Field(default_factory=datetime.now)
    config_used: ReservingConfigFile

    # Triangle info
    triangle_info: Optional[TriangleMetadata] = None

    # Full data for LLM analysis
    detailed_data: Optional[DetailedData] = None

    # LLM Method Selection
    method_selection: Optional[MethodSelection] = None

    # Deterministic
    chain_ladder: MethodResult
    cape_cod: Optional[MethodResult] = None

    # Stochastic
    mack: Optional[StochasticResult] = None
    bootstrap: Optional[StochasticResult] = None

    # Diagnostics
    diagnostics: Optional[DiagnosticsResult] = None
    
    # Raw object reference (internal use only, not serialized usually)
    # detailed_results_path: str

class ValidationStatus(str, Enum):
    PASSED = "PASSED"
    WARNING = "WARNING"
    REJECTED = "REJECTED"

class ValidationIssue(BaseModel):
    severity: str # "INFO", "WARNING", "CRITICAL"
    message: str
    component: str

class ValidationReport(BaseModel):
    """Output from the Validation Agent."""
    status: ValidationStatus
    overall_confidence_score: int = Field(ge=0, le=100) # 0-100
    issues: List[ValidationIssue] = []
    comparison_summary: str # Short summary of method comparisons

class AgentLog(BaseModel):
    """Audit log entry."""
    timestamp: datetime = Field(default_factory=datetime.now)
    agent: AgentRole
    action: str
    details: str
