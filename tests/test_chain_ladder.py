import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from chain_ladder import ChainLadder

def test_initialization(sample_triangle):
    cl = ChainLadder(sample_triangle)
    assert cl.n_years == 4
    assert cl.n_periods == 4
    assert cl.triangle.equals(sample_triangle)

def test_age_to_age_factors(sample_triangle):
    """
    Sample data:
    0: [100, 110, 120, 130] -> Factors: 2.0 (100->200), 2.0 (110->220), 2.0 (120->240) ??
    WAIT. My sample data in conftest was:
        0: [100, 110, 120, 130]
        1: [200, 220, 240, nan]
    
    Cols in conftest were 0, 1, 2, 3.
    Row 2020: 100 -> 200 (fac 2.0), 200->400 (fac 2.0), 400->800 (fac 2.0)
    Row 2021: 110 -> 220 (fac 2.0), 220->440 (fac 2.0)
    Row 2022: 120 -> 240 (fac 2.0)
    
    So all factors should be exactly 2.0.
    """
    cl = ChainLadder(sample_triangle)
    factors = cl.calculate_age_to_age_factors()
    
    # Check dimensions (should have n_periods - 1 columns)
    assert factors.shape == (4, 3) 
    
    # Check values
    # Col 0 (0->1): 200/100=2.0, 220/110=2.0, 240/120=2.0
    assert np.allclose(factors[0].dropna(), 2.0)
    
    # Col 1 (1->2): 400/200=2.0, 440/220=2.0
    assert np.allclose(factors[1].dropna(), 2.0)
    
    # Col 2 (2->3): 800/400=2.0
    assert np.allclose(factors[2].dropna(), 2.0)

def test_select_development_factors_simple_average(sample_triangle):
    cl = ChainLadder(sample_triangle)
    selected = cl.select_development_factors(method='simple_average')
    
    # Since all raw factors are 2.0, average must be 2.0
    assert np.allclose(selected, 2.0)
    assert len(selected) == 3

def test_cumulative_factors(sample_triangle):
    cl = ChainLadder(sample_triangle)
    cl.select_development_factors()
    cum_factors = cl.calculate_cumulative_factors()
    
    # Factors are 2.0, 2.0, 2.0
    # Cum factors (to ultimate):
    # Age 2 (last dev period): 2.0 (to get to 3)
    # Age 1: 2.0 * 2.0 = 4.0
    # Age 0: 2.0 * 2.0 * 2.0 = 8.0
    
    expected = pd.Series({
        2: 2.0,
        1: 4.0,
        0: 8.0
    })
    # Note: indices in pandas are flexible, let's match what the code does
    
    # The code indexes selected_factors by the START period.
    # periods: 0, 1, 2
    # 2->3 (ultimate): factor 2.0. Cum from 2 = 2.0
    # 1->2: factor 2.0. Cum from 1 = 2.0 * 2.0 = 4.0
    # 0->1: factor 2.0. Cum from 0 = 2.0 * 4.0 = 8.0
    
    assert np.allclose(cum_factors[2], 2.0)
    assert np.allclose(cum_factors[1], 4.0)
    assert np.allclose(cum_factors[0], 8.0)

def test_project_ultimate_losses(sample_triangle):
    cl = ChainLadder(sample_triangle)
    results = cl.run_full_analysis()
    
    # Latest values:
    # 2020: 800 (at age 3 - ultimate) -> Ult 800
    # 2021: 440 (at age 2). Cum factor for 2 is 2.0. -> Ult 440 * 2 = 880
    # 2022: 240 (at age 1). Cum factor for 1 is 4.0. -> Ult 240 * 4 = 960
    # 2023: 130 (at age 0). Cum factor for 0 is 8.0. -> Ult 130 * 8 = 1040
    
    ultimates = results['ultimate_losses']['Ultimate']
    
    assert np.isclose(ultimates[2020], 800)
    assert np.isclose(ultimates[2021], 880)
    assert np.isclose(ultimates[2022], 960)
    assert np.isclose(ultimates[2023], 1040)
