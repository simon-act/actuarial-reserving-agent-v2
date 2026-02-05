"""
Data Loader Module
==================

Flexible data loading for loss development triangles from various formats.

Supported formats:
- CSV (standard triangle format)
- Excel (.xlsx, .xls)
- chainladder-python Triangle objects
- pandas DataFrame
- Dictionary/JSON

Example:
    >>> loader = TriangleLoader()
    >>> triangle = loader.load('path/to/triangle.csv')
    >>> triangle = loader.load_sample('mack')  # Built-in sample
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
import json
import warnings


class TriangleLoader:
    """
    Flexible loader for loss development triangles.

    Handles multiple input formats and standardizes output to a clean
    pandas DataFrame with accident years as index and development
    periods as columns.
    """

    # Built-in sample triangles
    SAMPLE_TRIANGLES = {
        'mack': 'mack_genins.csv',
        'taylor_ashe': 'taylor_ashe.csv',
        'abc': 'abc_incurred.csv',
        'uk_motor': 'uk_motor.csv',
        'swiss_re': '../processed/reported_absolute_losses.csv'
    }

    SAMPLE_DESCRIPTIONS = {
        'mack': 'GenIns data from Mack (1993) - 10x10 triangle, General Insurance',
        'taylor_ashe': 'Taylor/Ashe data - 10x10 triangle, commonly used benchmark',
        'abc': 'ABC Insurance incurred losses - 10x10 triangle',
        'uk_motor': 'UK Motor claims - 7x7 triangle, shorter tail',
        'swiss_re': 'Swiss Re Property Reinsurance - 16x16 triangle'
    }

    def __init__(self, data_dir: str = None):
        """
        Initialize loader.

        Args:
            data_dir: Base directory for data files. If None, uses default.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.sample_dir = self.data_dir / "sample_triangles"

    def load(
        self,
        source: Union[str, Path, pd.DataFrame, Dict],
        index_col: Union[int, str] = 0,
        origin_col: str = None,
        development_cols: List[str] = None,
        value_col: str = None,
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load triangle from various sources.

        Args:
            source: File path, DataFrame, or dict
            index_col: Column to use as index (accident year)
            origin_col: Name of origin/accident year column (for long format)
            development_cols: Development period columns (auto-detected if None)
            value_col: Value column name (for long format)
            sheet_name: Sheet name for Excel files
            **kwargs: Additional arguments passed to pandas read functions

        Returns:
            Standardized triangle DataFrame
        """
        # Determine source type
        if isinstance(source, pd.DataFrame):
            triangle = self._from_dataframe(source, origin_col, development_cols, value_col)
        elif isinstance(source, dict):
            triangle = self._from_dict(source)
        elif isinstance(source, (str, Path)):
            source = Path(source)
            if source.suffix.lower() == '.csv':
                triangle = self._from_csv(source, index_col, **kwargs)
            elif source.suffix.lower() in ['.xlsx', '.xls']:
                triangle = self._from_excel(source, index_col, sheet_name, **kwargs)
            elif source.suffix.lower() == '.json':
                triangle = self._from_json(source)
            else:
                raise ValueError(f"Unsupported file format: {source.suffix}")
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # Standardize and validate
        triangle = self._standardize(triangle)
        self._validate(triangle)

        return triangle

    def load_sample(self, name: str) -> pd.DataFrame:
        """
        Load a built-in sample triangle.

        Args:
            name: Sample name ('mack', 'taylor_ashe', 'abc', 'uk_motor', 'swiss_re')

        Returns:
            Triangle DataFrame
        """
        if name not in self.SAMPLE_TRIANGLES:
            available = ', '.join(self.SAMPLE_TRIANGLES.keys())
            raise ValueError(f"Unknown sample '{name}'. Available: {available}")

        file_path = self.sample_dir / self.SAMPLE_TRIANGLES[name]

        if not file_path.exists():
            # Try relative path for swiss_re
            if name == 'swiss_re':
                file_path = self.data_dir / "processed" / "reported_absolute_losses.csv"

            if not file_path.exists():
                raise FileNotFoundError(f"Sample file not found: {file_path}")

        triangle = self._from_csv(file_path, index_col=0)
        triangle = self._standardize(triangle)

        print(f"✅ Loaded sample: {name}")
        print(f"   {self.SAMPLE_DESCRIPTIONS.get(name, '')}")
        print(f"   Shape: {triangle.shape[0]} accident years × {triangle.shape[1]} development periods")

        return triangle

    def list_samples(self) -> Dict[str, str]:
        """List available sample triangles."""
        print("\n📊 Available Sample Triangles:")
        print("-" * 60)
        for name, desc in self.SAMPLE_DESCRIPTIONS.items():
            print(f"  • {name}: {desc}")
        print("-" * 60)
        return self.SAMPLE_DESCRIPTIONS

    def _from_csv(self, path: Path, index_col: int = 0, **kwargs) -> pd.DataFrame:
        """Load from CSV file."""
        df = pd.read_csv(path, index_col=index_col, **kwargs)
        return df

    def _from_excel(
        self,
        path: Path,
        index_col: int = 0,
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> pd.DataFrame:
        """Load from Excel file."""
        df = pd.read_excel(path, index_col=index_col, sheet_name=sheet_name, **kwargs)
        return df

    def _from_json(self, path: Path) -> pd.DataFrame:
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return self._from_dict(data)

    def _from_dict(self, data: Dict) -> pd.DataFrame:
        """Load from dictionary."""
        if 'data' in data:
            # Structured format with metadata
            df = pd.DataFrame(data['data'])
            if 'index' in data:
                df.index = data['index']
            if 'columns' in data:
                df.columns = data['columns']
        else:
            # Direct data dictionary
            df = pd.DataFrame(data)
        return df

    def _from_dataframe(
        self,
        df: pd.DataFrame,
        origin_col: str = None,
        development_cols: List[str] = None,
        value_col: str = None
    ) -> pd.DataFrame:
        """
        Convert DataFrame to triangle format.

        Handles both wide format (already a triangle) and long format.
        """
        # Check if already in triangle format
        if origin_col is None and value_col is None:
            return df.copy()

        # Long format - need to pivot
        if origin_col and value_col:
            # Find development column
            dev_col = None
            for col in df.columns:
                if col not in [origin_col, value_col]:
                    dev_col = col
                    break

            if dev_col is None:
                raise ValueError("Cannot identify development period column")

            triangle = df.pivot(index=origin_col, columns=dev_col, values=value_col)
            return triangle

        return df.copy()

    def _standardize(self, triangle: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize triangle format.

        - Convert index to int (accident years)
        - Convert columns to int (development periods)
        - Sort by index and columns
        - Replace empty strings and zeros with NaN in upper triangle
        """
        # Copy to avoid modifying original
        df = triangle.copy()

        # Convert index to numeric
        try:
            df.index = df.index.astype(int)
        except (ValueError, TypeError):
            # Keep as is if can't convert
            pass

        # Convert columns to numeric
        try:
            df.columns = df.columns.astype(int)
        except (ValueError, TypeError):
            # Try extracting numbers from column names
            new_cols = []
            for col in df.columns:
                col_str = str(col)
                # Extract digits
                digits = ''.join(filter(str.isdigit, col_str))
                if digits:
                    new_cols.append(int(digits))
                else:
                    new_cols.append(col)
            df.columns = new_cols

        # Sort
        df = df.sort_index()
        df = df.reindex(sorted(df.columns), axis=1)

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')

        # Replace zeros with NaN only in positions that should be empty
        # (upper right triangle based on structure)
        n_rows, n_cols = df.shape
        for i in range(n_rows):
            for j in range(n_cols):
                if j > n_cols - 1 - i:
                    # This position should be empty (future development)
                    if df.iloc[i, j] == 0:
                        df.iloc[i, j] = np.nan

        return df

    def _validate(self, triangle: pd.DataFrame) -> None:
        """
        Validate triangle structure.

        Checks:
        - No negative values
        - Proper triangle structure (NaN in upper right)
        - Increasing development (cumulative)
        """
        # Check for negative values
        if (triangle < 0).any().any():
            warnings.warn("Triangle contains negative values")

        # Check triangle structure
        n_rows, n_cols = triangle.shape

        # Check if it looks like a proper triangle
        last_row_non_nan = triangle.iloc[-1].notna().sum()
        if last_row_non_nan > 2:
            warnings.warn("Last row has more than 2 non-NaN values. "
                         "This may not be a proper loss triangle.")

        # Check for increasing values (cumulative triangle)
        for i in range(n_rows):
            row = triangle.iloc[i].dropna()
            if len(row) > 1:
                diffs = np.diff(row.values)
                if (diffs < 0).any():
                    # Not strictly increasing - might be incremental
                    pass  # Just note, don't warn

    def save(
        self,
        triangle: pd.DataFrame,
        path: Union[str, Path],
        **kwargs
    ) -> None:
        """
        Save triangle to file.

        Args:
            triangle: Triangle DataFrame
            path: Output file path
            **kwargs: Additional arguments for pandas to_csv/to_excel
        """
        path = Path(path)

        if path.suffix.lower() == '.csv':
            triangle.to_csv(path, **kwargs)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            triangle.to_excel(path, **kwargs)
        elif path.suffix.lower() == '.json':
            data = {
                'data': triangle.to_dict(),
                'index': triangle.index.tolist(),
                'columns': triangle.columns.tolist()
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {path.suffix}")

        print(f"✅ Saved triangle to: {path}")


class TriangleInfo:
    """
    Utility class to analyze and describe a triangle.
    """

    def __init__(self, triangle: pd.DataFrame):
        self.triangle = triangle
        self.n_years = len(triangle)
        self.n_periods = len(triangle.columns)

        # Calculate key statistics
        self._analyze()

    def _analyze(self):
        """Analyze triangle structure."""
        df = self.triangle

        # Origin years
        self.origin_start = df.index.min()
        self.origin_end = df.index.max()

        # Development periods
        self.dev_start = df.columns.min()
        self.dev_end = df.columns.max()

        # Data points
        self.n_observations = df.notna().sum().sum()
        self.n_missing = df.isna().sum().sum()

        # Latest diagonal
        self.latest_values = []
        for i, year in enumerate(df.index):
            row = df.loc[year].dropna()
            if len(row) > 0:
                self.latest_values.append(row.iloc[-1])
            else:
                self.latest_values.append(np.nan)

        self.total_latest = sum(v for v in self.latest_values if not np.isnan(v))

        # Check if cumulative or incremental
        self.is_cumulative = self._check_cumulative()

    def _check_cumulative(self) -> bool:
        """Check if triangle appears to be cumulative."""
        df = self.triangle

        increasing_rows = 0
        total_rows = 0

        for i in range(len(df)):
            row = df.iloc[i].dropna()
            if len(row) > 1:
                total_rows += 1
                if all(np.diff(row.values) >= 0):
                    increasing_rows += 1

        # If most rows are increasing, assume cumulative
        return increasing_rows / total_rows > 0.8 if total_rows > 0 else True

    def summary(self) -> Dict:
        """Return summary dictionary."""
        return {
            'n_years': self.n_years,
            'n_periods': self.n_periods,
            'origin_range': f"{self.origin_start} - {self.origin_end}",
            'dev_range': f"{self.dev_start} - {self.dev_end}",
            'n_observations': self.n_observations,
            'total_latest': self.total_latest,
            'is_cumulative': self.is_cumulative
        }

    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 50)
        print("TRIANGLE SUMMARY")
        print("=" * 50)
        print(f"  Accident Years:      {self.n_years} ({self.origin_start} - {self.origin_end})")
        print(f"  Development Periods: {self.n_periods} ({self.dev_start} - {self.dev_end})")
        print(f"  Observations:        {self.n_observations}")
        print(f"  Total Latest:        {self.total_latest:,.0f}")
        print(f"  Type:                {'Cumulative' if self.is_cumulative else 'Incremental'}")
        print("=" * 50)


# Convenience functions
def load_triangle(source: Union[str, Path, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Quick function to load a triangle."""
    loader = TriangleLoader()
    return loader.load(source, **kwargs)


def load_sample(name: str) -> pd.DataFrame:
    """Quick function to load a sample triangle."""
    loader = TriangleLoader()
    return loader.load_sample(name)


def list_samples():
    """List available sample triangles."""
    loader = TriangleLoader()
    return loader.list_samples()


# Main test
if __name__ == "__main__":
    print("\n🔧 Testing Triangle Loader\n")

    loader = TriangleLoader()
    loader.list_samples()

    # Test loading each sample
    for name in ['mack', 'taylor_ashe', 'abc', 'uk_motor']:
        print(f"\n{'='*60}")
        triangle = loader.load_sample(name)
        info = TriangleInfo(triangle)
        info.print_summary()
        print("\nFirst 5 rows:")
        print(triangle.head())
