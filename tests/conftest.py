import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_triangle():
    """
    Creates a simple 4x4 cumulative triangle for testing.
    Pattern: No development (all factors = 1.0) would mean values stay same across rows.
    Here we'll model a simple doubling pattern to make math easy to check.
    
    Periods: 0, 1, 2, 3
    """
    data = {
        0: [100, 110, 120, 130],
        1: [200, 220, 240, np.nan],
        2: [400, 440, np.nan, np.nan],
        3: [800, np.nan, np.nan, np.nan]
    }
    # Index = Accident Years
    df = pd.DataFrame(data, index=[2020, 2021, 2022, 2023])
    df.columns = [0, 1, 2, 3] # Development periods
    return df
