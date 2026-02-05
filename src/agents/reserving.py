import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from agents.settings import INPUT_DIR
from agents.schemas import (
    ReservingInput, ReservingConfigFile, ReservingOutput,
    MethodResult, StochasticResult, DiagnosticsResult, AgentRole, AgentLog,
    TriangleMetadata, DetailedData
)
# Import existing logic
import sys
sys.path.append("src") # Ensure src is in path if running from root
from enhanced_workflow import EnhancedReservingWorkflow

class ReservingExecutionAgent:
    """
    ACTUARY ROLE (Execution)
    - Receives instructions (config)
    - Runs calculations (using existing package)
    - Returns structured data
    - NO opinions, NO text generation
    """
    
    def __init__(self, name: str = "Actuary_Bot"):
        self.name = name
        self.role = AgentRole.EXECUTION

    def execute(
        self,
        inputs: ReservingInput,
        config: ReservingConfigFile,
        selected_factors: Optional[pd.Series] = None
    ) -> ReservingOutput:
        """Run the actuarial workflow based on config."""
        
        print(f"[{self.role}] ⚙️ Starting execution with config: {config.analysis_type}")
        
        # 1. Load Data
        triangle = self._load_data(inputs.triangle_path)
        premium = self._load_data(inputs.premium_path, is_series=True) if inputs.premium_path else None
        
        # 2. Init Workflow
        workflow = EnhancedReservingWorkflow(
            triangle=triangle,
            earned_premium=premium,
            verbose=False # Keep it quiet, we are agents
        )
        
        # 3. Execute Modules based on Config
        if selected_factors is not None:
            workflow.selected_factors = selected_factors

        # Always run CL
        workflow.run_chain_ladder()
        
        # Stochastic
        if config.analysis_type != "quick":
            workflow.run_mack_model()
            
        if config.run_bootstrap:
            workflow.run_bootstrap(n_simulations=config.n_bootstrap_simulations)
            
        # Alternative Methods
        if config.analysis_type == "full" and premium is not None:
            workflow.run_alternative_methods()
            
        if config.run_diagnostics:
            workflow.run_diagnostics()
            
        if config.run_stress_testing and config.analysis_type == "full":
            workflow.run_stress_testing()
            
        # 4. Extract Structured Output
        output = self._package_results(workflow, config, triangle)
        print(f"[{self.role}] ✅ Execution complete. Total Reserve (CL): ${output.chain_ladder.total_reserve:,.0f}")
        
        return output

    def _load_data(self, path: str, is_series: bool = False) -> Any:
        try:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            df = pd.read_csv(p, index_col=0)
            if is_series:
                return df.iloc[:, 0]
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {path}: {str(e)}")

    def _package_results(self, workflow: EnhancedReservingWorkflow, config: ReservingConfigFile, triangle: pd.DataFrame) -> ReservingOutput:
        results = workflow.results

        # Triangle metadata
        triangle_meta = TriangleMetadata(
            n_accident_years=len(triangle),
            n_development_periods=len(triangle.columns),
            first_accident_year=int(triangle.index[0]),
            last_accident_year=int(triangle.index[-1]),
            currency="USD",
            units="millions"
        )

        # Detailed data for LLM analysis
        cl_obj = results['chain_ladder']

        # Convert triangle to nested dict {year: {period: value}}
        triangle_dict = {}
        for year in triangle.index:
            triangle_dict[str(year)] = {
                str(col): round(float(val), 2) if pd.notna(val) else None
                for col, val in triangle.loc[year].items()
            }

        # Development factors
        dev_factors = {}
        if cl_obj.selected_factors is not None:
            dev_factors = {
                str(k): round(float(v), 4)
                for k, v in cl_obj.selected_factors.items()
            }

        # Reserves and ultimates by year
        reserves_by_year = {}
        ultimates_by_year = {}
        latest_diagonal = {}
        if cl_obj.ultimate_losses is not None:
            for year in cl_obj.ultimate_losses.index:
                reserves_by_year[str(year)] = round(float(cl_obj.ultimate_losses.loc[year, 'Reserve']), 2)
                ultimates_by_year[str(year)] = round(float(cl_obj.ultimate_losses.loc[year, 'Ultimate']), 2)
                latest_diagonal[str(year)] = round(float(cl_obj.ultimate_losses.loc[year, 'Latest_Value']), 2)

        detailed = DetailedData(
            triangle=triangle_dict,
            development_factors=dev_factors,
            reserves_by_year=reserves_by_year,
            ultimates_by_year=ultimates_by_year,
            latest_diagonal=latest_diagonal
        )

        # Chain Ladder (Always present)
        cl_sum = results['chain_ladder'].summary()
        cl_res = MethodResult(
            method_name="Chain Ladder",
            total_reserve=cl_sum['total_reserve'],
            ultimate_loss=cl_sum['total_ultimate'],
            model_params={"link_ratios": "calculated_internally"}
        )
        
        # Mack
        mack_res = None
        if results['mack']:
            m_tot = results['mack'].get_total_reserve_distribution()
            mack_res = StochasticResult(
                method_name="Mack Model",
                total_reserve=m_tot['Total_Reserve'],
                ultimate_loss=cl_sum['total_ultimate'], # Mack assumes same ultimate mean as CL usually
                standard_error=m_tot['Total_SE'],
                cv=m_tot['Total_CV']
            )
            
        # Bootstrap
        boot_res = None
        if results['bootstrap']:
            b_tot = results['bootstrap'].get_total_reserve_distribution()
            boot_res = StochasticResult(
                method_name="ODP Bootstrap",
                total_reserve=b_tot['Mean'], # Mean reserve
                ultimate_loss=b_tot['Mean'] + cl_sum['total_latest'],
                standard_error=b_tot['Std'],
                cv=b_tot['Std'] / b_tot['Mean'] if b_tot['Mean'] > 0 else 0,
                percentiles={
                    "75%": b_tot['P75'],
                    "95%": b_tot['P95'],
                    "99.5%": b_tot.get('P99.5', 0)
                }
            )
            
        # Cape Cod
        cc_res = None
        if results['cape_cod']:
            cc_sum = results['cape_cod'].summary()
            # CC Summary has total_reported
            total_cc_ultimate = cc_sum['total_cc_reserve'] + cc_sum['total_reported']
            
            cc_res = MethodResult(
                method_name="Cape Cod",
                total_reserve=cc_sum['total_cc_reserve'],
                ultimate_loss=total_cc_ultimate,
                model_params={"decay": "1.0", "elr": f"{cc_sum.get('cape_cod_elr', 0):.2%}"}
            )

        # Diagnostics
        diag_res = None
        if results['diagnostics'] and results['diagnostics'].get('tests'):
            adequacy = results['diagnostics']['tests'].get_model_adequacy_score()
            diag_res = DiagnosticsResult(
                adequacy_score=adequacy['adequacy_score'],
                rating=adequacy['rating'],
                issues=adequacy['issues']
            )

        return ReservingOutput(
            config_used=config,
            triangle_info=triangle_meta,
            detailed_data=detailed,
            chain_ladder=cl_res,
            cape_cod=cc_res,
            mack=mack_res,
            bootstrap=boot_res,
            diagnostics=diag_res
        )
