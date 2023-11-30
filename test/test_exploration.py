import unittest
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

# Import the ExplorationPlots class
from fplibrary.exploration import ExplorationPlots, MissingValues, Correlation


class TestExplorationPlots(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'Feature1': np.random.rand(100),
            'Feature2': np.random.choice(['A', 'B', 'C'], 100),
            'Feature3': np.random.randint(1, 100, 100),
            'Target': np.random.randn(100)
        })
        self.exploration_plots = ExplorationPlots(self.df)

    def _assert_plot_generated(self, plot_func, *args, **kwargs):
        # Use the Agg backend to avoid displaying plots during the tests
        plt.switch_backend('Agg')

        # Create an in-memory buffer to save the plot
        buf = io.BytesIO()

        # Redirect the plot to the buffer
        self.assertWarns(UserWarning, plot_func, *args, **kwargs)

        # Save the plot to the buffer
        plt.savefig(buf, format='png')

        # Switch back to the default backend
        plt.switch_backend('module://ipykernel.pylab.backend_inline')

        # Assert that the buffer is not empty
        self.assertGreater(len(buf.getvalue()), 0)

    def test_create_boxplot(self):
        self._assert_plot_generated(self.exploration_plots.create_boxplot, 'Target', ['Feature1', 'Feature2'])

    def test_create_scatter_plot(self):
        self._assert_plot_generated(self.exploration_plots.create_scatter_plot, [('Feature1', 'Feature3')])

    def test_create_histogram(self):
        self._assert_plot_generated(self.exploration_plots.create_histogram, ['Feature3'])


class TestMissingValues(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'Feature1': [1, 2, np.nan, 4, 5],
            'Feature2': [6, 7, 8, 9, 10],
            'Feature3': [11, np.nan, 13, 14, 15]
        })
        self.empty_df = pd.DataFrame()  # Empty DataFrame
        self.missing_values = MissingValues(self.df)
        self.empty_missing_values = MissingValues(self.empty_df)

    def test_missing_values_summary(self):
        # Test for non-empty DataFrame
        summary = self.missing_values.missing_values_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(summary.shape, (2, 2))

    def test_missing_values_summary_empty_df(self):
        # Test for empty DataFrame
        empty_summary = self.empty_missing_values.missing_values_summary()
        self.assertIsInstance(empty_summary, pd.DataFrame)
        self.assertTrue(empty_summary.empty)


if __name__ == '__main__':
    unittest.main()
