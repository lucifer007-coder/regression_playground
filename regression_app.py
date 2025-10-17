import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import time
import json
import math

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Regression Playground",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better responsiveness
st.markdown("""
<style>
    /* Color scheme: Orange for positive, Blue for negative */
    .positive-value { color: #FF6B35; font-weight: bold; }
    .negative-value { color: #004E89; font-weight: bold; }
    .neutral-value { color: #666; }

    /* Interactive controls styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #004E89 0%, #FF6B35 100%);
    }

    /* Metric cards with visual emphasis */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #FF6B35;
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* Progressive disclosure sections */
    .learning-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }

    /* Animated transitions */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Visual weight representation */
    .weight-strong { font-size: 1.2em; font-weight: bold; }
    .weight-medium { font-size: 1.0em; font-weight: normal; }
    .weight-weak { font-size: 0.9em; font-weight: 300; opacity: 0.7; }

    /* Error and warning styling */
    .error-box {
        background: #ffe6e6;
        border: 1px solid #ff9999;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }

    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class LearningState:
    """Track user's learning progress and provide adaptive guidance."""
    experiments_count: int = 0
    parameters_changed: List[str] = None
    patterns_discovered: List[str] = None
    current_hypothesis: str = ""
    reflection_notes: List[str] = None
    last_r_squared: float = 0.0
    best_r_squared: float = 0.0

    def __post_init__(self):
        if self.parameters_changed is None:
            self.parameters_changed = []
        if self.patterns_discovered is None:
            self.patterns_discovered = []
        if self.reflection_notes is None:
            self.reflection_notes = []

@dataclass
class ModelResults:
    """Container for model results and statistics."""
    coefficients: np.ndarray
    intercept: float
    predictions: np.ndarray
    residuals: np.ndarray
    r_squared: float
    adj_r_squared: float
    rmse: float
    mae: float
    aic: float
    bic: float
    std_errors: Optional[np.ndarray] = None
    t_stats: Optional[np.ndarray] = None
    p_values: Optional[np.ndarray] = None
    cost_history: Optional[List[float]] = None
    converged: bool = True
    decision_boundary: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
    training_time: float = 0.0
    final_cost: float = 0.0

class DataValidator:
    """Validate and clean data for regression analysis."""

    @staticmethod
    def validate_data(X: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
        """Validate input data and return status with message."""
        try:
            # Check for NaN or infinite values
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                return False, "Data contains NaN values. Please clean your data."

            if np.any(np.isinf(X)) or np.any(np.isinf(y)):
                return False, "Data contains infinite values. Please clean your data."

            # Check dimensions
            if X.shape[0] != y.shape[0]:
                return False, "X and y must have the same number of samples."

            if X.shape[0] < 3:
                return False, "Need at least 3 data points for regression."

            # Check for constant features
            if X.shape[1] > 1:
                constant_features = []
                for i in range(X.shape[1]):
                    if np.std(X[:, i]) < 1e-10:
                        constant_features.append(i)

                if constant_features:
                    return False, f"Features {constant_features} are constant. Remove them."

            # Check for perfect multicollinearity
            if X.shape[1] > 1:
                try:
                    correlation_matrix = np.corrcoef(X.T)
                    if np.any(np.abs(correlation_matrix - np.eye(X.shape[1])) > 0.999):
                        return False, "Perfect multicollinearity detected. Remove redundant features."
                except:
                    pass  # Skip if correlation calculation fails

            return True, "Data validation passed."

        except Exception as e:
            return False, f"Data validation error: {str(e)}"

    @staticmethod
    def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clean data by removing NaN and infinite values."""
        # Find valid indices
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) |
                      np.isinf(X).any(axis=1) | np.isinf(y))

        return X[valid_mask], y[valid_mask]

class DatasetGenerator:
    """Generate educational datasets with known patterns."""

    @staticmethod
    def generate_linear(n_samples: int = 100, noise: float = 0.1, slope: float = 2.0,
                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simple linear relationship."""
        np.random.seed(seed)
        X = np.random.uniform(-2, 2, n_samples).reshape(-1, 1)
        y = slope * X.flatten() + 1 + noise * np.random.randn(n_samples)
        return X, y

    @staticmethod
    def generate_polynomial(n_samples: int = 100, degree: int = 2, noise: float = 0.1,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate polynomial relationship."""
        np.random.seed(seed)
        X = np.random.uniform(-2, 2, n_samples).reshape(-1, 1)
        y = X.flatten()**degree + 0.5 * X.flatten() + noise * np.random.randn(n_samples)
        return X, y

    @staticmethod
    def generate_multivariate(n_samples: int = 100, n_features: int = 3, noise: float = 0.1,
                            seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate multivariate linear relationship."""
        np.random.seed(seed)
        X = np.random.randn(n_samples, n_features)
        true_coefs = np.array([1.5, -2.0, 0.5, 1.0, -0.8][:n_features])
        y = X @ true_coefs + noise * np.random.randn(n_samples)
        return X, y

    @staticmethod
    def generate_nonlinear(n_samples: int = 100, noise: float = 0.1,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate non-linear relationship."""
        np.random.seed(seed)
        X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
        y = np.sin(X.flatten()) + 0.3 * X.flatten() + noise * np.random.randn(n_samples)
        return X, y

    @staticmethod
    def generate_outliers(n_samples: int = 100, n_outliers: int = 5, noise: float = 0.1,
                         seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate linear data with outliers."""
        np.random.seed(seed)
        X = np.random.uniform(-2, 2, n_samples).reshape(-1, 1)
        y = 2 * X.flatten() + 1 + noise * np.random.randn(n_samples)

        # Add outliers
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        y[outlier_indices] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(3, 6, n_outliers)

        return X, y

class VisualFeedbackSystem:
    """Provide real-time visual feedback for learning."""

    @staticmethod
    def create_parameter_impact_viz(old_results: ModelResults, new_results: ModelResults,
                                  parameter_name: str) -> go.Figure:
        """Show immediate impact of parameter changes."""
        fig = go.Figure()

        metrics = ['R² Score', 'RMSE', 'MAE']
        old_values = [old_results.r_squared, old_results.rmse, old_results.mae]
        new_values = [new_results.r_squared, new_results.rmse, new_results.mae]

        # Calculate percentage changes
        changes = []
        for old_val, new_val in zip(old_values, new_values):
            if old_val != 0:
                change = ((new_val - old_val) / abs(old_val)) * 100
            else:
                change = 0
            changes.append(change)

        # Before/after comparison
        fig.add_trace(go.Bar(
            x=metrics,
            y=old_values,
            name='Before',
            marker_color='#004E89',
            opacity=0.7,
            text=[f'{val:.3f}' for val in old_values],
            textposition='auto'
        ))

        fig.add_trace(go.Bar(
            x=metrics,
            y=new_values,
            name='After',
            marker_color='#FF6B35',
            opacity=0.7,
            text=[f'{val:.3f}<br>({change:+.1f}%)' for val, change in zip(new_values, changes)],
            textposition='auto'
        ))

        fig.update_layout(
            title=f"Impact of Changing {parameter_name}",
            barmode='group',
            height=350,
            yaxis_title="Value",
            showlegend=True
        )

        return fig

    @staticmethod
    def create_coefficient_network_viz(coefficients: np.ndarray, feature_names: List[str],
                                     intercept: float) -> go.Figure:
        """Visualize coefficients as network connections."""
        fig = go.Figure()

        n_features = len(coefficients)
        if n_features == 0:
            return fig

        # Create network-style visualization
        input_x = [0] * n_features
        input_y = list(range(n_features))

        # Output layer
        output_x = [3]
        output_y = [n_features // 2]

        # Normalize coefficient magnitudes for visualization
        max_coef = max(np.max(np.abs(coefficients)), 0.1)  # Avoid division by zero

        # Draw connections with thickness proportional to coefficient magnitude
        for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
            line_width = max(1, min(abs(coef) / max_coef * 10, 15))  # Scale line width
            line_color = '#FF6B35' if coef > 0 else '#004E89'
            opacity = min(0.9, max(0.3, abs(coef) / max_coef))

            fig.add_trace(go.Scatter(
                x=[0, 3],
                y=[i, n_features // 2],
                mode='lines',
                line=dict(width=line_width, color=line_color),
                opacity=opacity,
                name=f'{name}: {coef:.3f}',
                hovertemplate=f'{name}<br>Coefficient: {coef:.4f}<br>Weight: {"Strong" if abs(coef) > max_coef*0.5 else "Medium" if abs(coef) > max_coef*0.1 else "Weak"}<extra></extra>',
                showlegend=False
            ))

        # Add nodes
        fig.add_trace(go.Scatter(
            x=input_x,
            y=input_y,
            mode='markers+text',
            marker=dict(size=25, color='lightblue', line=dict(width=2, color='darkblue')),
            text=feature_names,
            textposition='middle left',
            name='Features',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=output_x,
            y=output_y,
            mode='markers+text',
            marker=dict(size=35, color='lightcoral', line=dict(width=2, color='darkred')),
            text=[f'Output<br>+{intercept:.2f}'],
            textposition='middle right',
            name='Prediction',
            showlegend=False
        ))

        fig.update_layout(
            title="Model Structure: Feature Weights Visualization",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 3.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=max(300, n_features * 50),
            plot_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    @staticmethod
    def create_residual_analysis(residuals: np.ndarray, predictions: np.ndarray) -> go.Figure:
        """Create comprehensive residual analysis plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Fitted', 'Q-Q Plot', 'Histogram of Residuals', 'Residuals vs Index'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Residuals vs Fitted
        fig.add_trace(
            go.Scatter(x=predictions, y=residuals, mode='markers',
                      marker=dict(color='#004E89', opacity=0.6),
                      name='Residuals'),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        # Q-Q Plot
        try:
            from scipy.stats import probplot
            qq_data = probplot(residuals, dist="norm")
            fig.add_trace(
                go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers',
                          marker=dict(color='#FF6B35', opacity=0.6),
                          name='Q-Q'),
                row=1, col=2
            )
            # Add reference line
            fig.add_trace(
                go.Scatter(x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                          mode='lines', line=dict(color='red', dash='dash'),
                          name='Reference Line'),
                row=1, col=2
            )
        except:
            # Fallback if scipy is not available
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers',
                          marker=dict(color='#FF6B35', opacity=0.6),
                          name='Q-Q'),
                row=1, col=2
            )

        # Histogram of Residuals
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=20, marker_color='#004E89',
                        opacity=0.7, name='Residual Distribution'),
            row=2, col=1
        )

        # Residuals vs Index
        fig.add_trace(
            go.Scatter(x=list(range(len(residuals))), y=residuals, mode='markers',
                      marker=dict(color='#FF6B35', opacity=0.6),
                      name='Residuals vs Index'),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

        fig.update_layout(height=600, showlegend=False, title_text="Residual Analysis")
        return fig

class BaseRegressor(ABC):
    """Abstract base class for all regression models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> ModelResults:
        """Fit the model and return results."""
        pass

    def _calculate_statistics(self, X: np.ndarray, y: np.ndarray,
                            predictions: np.ndarray, coefficients: np.ndarray,
                            intercept: float) -> Dict[str, float]:
        """Calculate common regression statistics with improved numerical stability."""
        n, p = X.shape
        residuals = y - predictions

        # R-squared with numerical stability
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot < 1e-10:  # Constant target variable
            r_squared = 0.0
        else:
            r_squared = max(0.0, 1 - (ss_res / ss_tot))  # Ensure non-negative

        # Adjusted R-squared
        if n > p + 1:
            adj_r_squared = max(0.0, 1 - (1 - r_squared) * (n - 1) / (n - p - 1))
        else:
            adj_r_squared = r_squared

        # RMSE and MAE
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))

        # AIC and BIC with numerical stability
        mse = np.mean(residuals ** 2)
        if mse > 1e-10:
            log_likelihood = -n/2 * np.log(2 * np.pi * mse) - ss_res / (2 * mse)
            aic = 2 * (p + 1) - 2 * log_likelihood
            bic = np.log(n) * (p + 1) - 2 * log_likelihood
        else:
            aic = -np.inf  # Perfect fit
            bic = -np.inf

        # Feature importance (normalized absolute coefficients)
        coef_sum = np.sum(np.abs(coefficients))
        if coef_sum > 1e-10:
            feature_importance = np.abs(coefficients) / coef_sum
        else:
            feature_importance = np.ones_like(coefficients) / len(coefficients)

        return {
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'rmse': rmse,
            'mae': mae,
            'aic': aic,
            'bic': bic,
            'feature_importance': feature_importance
        }

class GradientDescentRegressor(BaseRegressor):
    """Enhanced Gradient Descent with improved numerical stability."""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, cost_function: str = 'mse'):
        self.learning_rate = max(1e-6, min(10.0, learning_rate))  # Clamp learning rate
        self.max_iterations = max(1, min(10000, max_iterations))  # Clamp iterations
        self.tolerance = max(1e-12, tolerance)
        self.cost_function = cost_function.lower()

    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray,
                     delta: float = 1.0) -> float:
        """Compute cost based on selected function with numerical stability."""
        residuals = y_true - y_pred

        if self.cost_function == 'mse':
            return np.mean(residuals ** 2)
        elif self.cost_function == 'mae':
            return np.mean(np.abs(residuals))
        elif self.cost_function == 'huber':
            abs_residuals = np.abs(residuals)
            return np.mean(np.where(abs_residuals <= delta,
                                  0.5 * residuals ** 2,
                                  delta * abs_residuals - 0.5 * delta ** 2))
        else:
            return np.mean(residuals ** 2)  # Default to MSE

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray,
                          predictions: np.ndarray, delta: float = 1.0) -> Tuple[np.ndarray, float]:
        """Compute gradients with numerical stability."""
        m = len(y)
        residuals = y - predictions

        if self.cost_function == 'mse':
            dw = -(2/m) * X.T @ residuals
            db = -(2/m) * np.sum(residuals)
        elif self.cost_function == 'mae':
            signs = np.sign(residuals)
            signs[residuals == 0] = 0  # Handle exact zeros
            dw = -(1/m) * X.T @ signs
            db = -(1/m) * np.sum(signs)
        elif self.cost_function == 'huber':
            abs_residuals = np.abs(residuals)
            mask = abs_residuals <= delta
            huber_residuals = np.where(mask, residuals, delta * np.sign(residuals))
            dw = -(1/m) * X.T @ huber_residuals
            db = -(1/m) * np.sum(huber_residuals)
        else:
            dw = -(2/m) * X.T @ residuals
            db = -(2/m) * np.sum(residuals)

        # Clip gradients to prevent explosion
        dw = np.clip(dw, -100, 100)
        db = np.clip(db, -100, 100)

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray, delta: float = 1.0,
            progress_callback=None) -> ModelResults:
        """Fit model using gradient descent with enhanced stability."""
        start_time = time.time()

        # Validate data
        is_valid, message = DataValidator.validate_data(X, y)
        if not is_valid:
            st.error(f"Data validation failed: {message}")
            # Return dummy results
            return ModelResults(
                coefficients=np.zeros(X.shape[1]),
                intercept=0.0,
                predictions=np.zeros_like(y),
                residuals=y.copy(),
                r_squared=0.0,
                adj_r_squared=0.0,
                rmse=np.std(y),
                mae=np.mean(np.abs(y - np.mean(y))),
                aic=np.inf,
                bic=np.inf,
                converged=False,
                training_time=time.time() - start_time
            )

        m, n = X.shape

        # Initialize parameters with better initialization
        coefficients = np.random.normal(0, 0.01, n)
        intercept = np.mean(y)  # Better initialization

        cost_history = []
        prev_cost = float('inf')
        converged = False
        patience = 50  # Early stopping patience
        no_improve_count = 0

        # Adaptive learning rate
        initial_lr = self.learning_rate

        for i in range(self.max_iterations):
            # Forward pass
            predictions = X @ coefficients + intercept
            cost = self._compute_cost(y, predictions, delta)

            # Check for numerical issues
            if np.isnan(cost) or np.isinf(cost):
                st.warning("Training diverged due to numerical instability. Try reducing learning rate.")
                break

            cost_history.append(cost)

            # Progress callback for real-time updates (less frequent to avoid lag)
            if progress_callback and i % 20 == 0:
                try:
                    progress_callback(i, cost, coefficients.copy(), intercept)
                except:
                    pass  # Don't let callback errors stop training

            # Check for divergence
            if cost > 1e10:
                st.warning("Cost is growing too large. Training stopped.")
                break

            # Adaptive learning rate based on cost improvement
            if i > 0 and cost > cost_history[-2]:
                self.learning_rate *= 0.95  # Reduce learning rate if cost increases
                no_improve_count += 1
            else:
                no_improve_count = 0

            # Early stopping
            if no_improve_count > patience:
                converged = True
                break

            # Check convergence
            if i > 0 and abs(prev_cost - cost) < self.tolerance:
                converged = True
                break

            # Compute gradients
            dw, db = self._compute_gradients(X, y, predictions, delta)

            # Update parameters with momentum (simple version)
            coefficients -= self.learning_rate * dw
            intercept -= self.learning_rate * db

            prev_cost = cost

        # Reset learning rate
        self.learning_rate = initial_lr

        final_predictions = X @ coefficients + intercept
        residuals = y - final_predictions

        stats_dict = self._calculate_statistics(X, y, final_predictions, coefficients, intercept)

        training_time = time.time() - start_time

        return ModelResults(
            coefficients=coefficients,
            intercept=intercept,
            predictions=final_predictions,
            residuals=residuals,
            cost_history=cost_history,
            converged=converged,
            training_time=training_time,
            final_cost=cost_history[-1] if cost_history else float('inf'),
            **stats_dict
        )

class OLSRegressor(BaseRegressor):
    """Ordinary Least Squares with improved numerical stability."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> ModelResults:
        """Fit OLS model using normal equations with regularization fallback."""
        start_time = time.time()

        # Validate data
        is_valid, message = DataValidator.validate_data(X, y)
        if not is_valid:
            st.error(f"Data validation failed: {message}")
            # Return dummy results
            return ModelResults(
                coefficients=np.zeros(X.shape[1]),
                intercept=np.mean(y) if len(y) > 0 else 0.0,
                predictions=np.full_like(y, np.mean(y)) if len(y) > 0 else np.zeros_like(y),
                residuals=y - np.mean(y) if len(y) > 0 else y.copy(),
                r_squared=0.0,
                adj_r_squared=0.0,
                rmse=np.std(y) if len(y) > 0 else 0.0,
                mae=np.mean(np.abs(y - np.mean(y))) if len(y) > 0 else 0.0,
                aic=np.inf,
                bic=np.inf,
                converged=False,
                training_time=time.time() - start_time
            )

        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        try:
            # Try normal equations first
            XtX = X_with_intercept.T @ X_with_intercept
            Xty = X_with_intercept.T @ y

            # Check condition number
            cond_num = np.linalg.cond(XtX)

            if cond_num > 1e12:  # Matrix is ill-conditioned
                st.warning(f"Matrix is ill-conditioned (condition number: {cond_num:.2e}). Using regularized solution.")
                # Add small regularization
                regularization = 1e-6 * np.eye(XtX.shape[0])
                theta = np.linalg.solve(XtX + regularization, Xty)
            else:
                theta = np.linalg.solve(XtX, Xty)

            intercept = theta[0]
            coefficients = theta[1:]

        except np.linalg.LinAlgError:
            # Use pseudo-inverse for rank-deficient matrices
            st.warning("Using pseudo-inverse due to singular matrix.")
            try:
                theta = np.linalg.pinv(X_with_intercept) @ y
                intercept = theta[0]
                coefficients = theta[1:]
            except:
                # Last resort: return mean prediction
                st.error("Could not solve linear system. Returning mean prediction.")
                intercept = np.mean(y)
                coefficients = np.zeros(X.shape[1])

        predictions = X @ coefficients + intercept
        residuals = y - predictions

        # Calculate standard errors and t-statistics
        std_errors = None
        t_stats = None
        p_values = None

        try:
            n, p = X.shape
            if n > p + 1:  # Need more observations than parameters
                mse = np.sum(residuals ** 2) / (n - p - 1)
                if mse > 1e-10:
                    try:
                        var_covar_matrix = mse * np.linalg.inv(XtX)
                        std_errors = np.sqrt(np.diag(var_covar_matrix))

                        # Avoid division by zero
                        non_zero_se = std_errors > 1e-10
                        t_stats = np.zeros_like(theta)
                        t_stats[non_zero_se] = theta[non_zero_se] / std_errors[non_zero_se]

                        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
                    except:
                        pass  # Keep as None if calculation fails
        except:
            pass  # Keep as None if calculation fails

        stats_dict = self._calculate_statistics(X, y, predictions, coefficients, intercept)

        training_time = time.time() - start_time

        return ModelResults(
            coefficients=coefficients,
            intercept=intercept,
            predictions=predictions,
            residuals=residuals,
            std_errors=std_errors,
            t_stats=t_stats,
            p_values=p_values,
            training_time=training_time,
            **stats_dict
        )

def create_interactive_playground():
    """Create the main interactive playground interface."""

    # Initialize session state for learning tracking
    if 'learning_state' not in st.session_state:
        st.session_state.learning_state = LearningState()

    if 'previous_results' not in st.session_state:
        st.session_state.previous_results = None

    # Header with visual emphasis
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(90deg, #004E89, #FF6B35); color: white; border-radius: 10px; margin-bottom: 2rem;'>
        <h1>🎯 Regression Playground</h1>
        <p style='font-size: 1.2em; margin: 0;'>Visual-First Learning for Linear Regression</p>
        <p style='margin: 0; opacity: 0.9;'>Experiment • Discover • Learn</p>
    </div>
    """, unsafe_allow_html=True)

    # Progressive disclosure: Start with data selection
    with st.container():
        st.markdown("### 📊 Step 1: Choose Your Data")

        data_source = st.radio(
            "Select data source:",
            ["📚 Educational Datasets", "📁 Upload Your Own"],
            horizontal=True,
            help="Start with educational datasets to learn patterns, then try your own data!"
        )

        if data_source == "📚 Educational Datasets":
            dataset_type = st.selectbox(
                "Choose a learning scenario:",
                [
                    "Linear Relationship (Easy)",
                    "Polynomial Relationship (Medium)",
                    "Multiple Features (Medium)",
                    "Non-linear Challenge (Hard)",
                    "Data with Outliers (Advanced)"
                ],
                help="Each dataset teaches different concepts!"
            )

            # Dataset parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                n_samples = st.slider("Number of samples", 50, 500, 100)
            with col2:
                noise_level = st.slider("Noise level", 0.0, 1.0, 0.1, 0.05)
            with col3:
                random_seed = st.number_input("Random seed", 0, 1000, 42)

            # Generate educational dataset
            try:
                if "Linear" in dataset_type:
                    slope = st.slider("True slope", -5.0, 5.0, 2.0, 0.1)
                    X, y = DatasetGenerator.generate_linear(n_samples, noise_level, slope, random_seed)
                    feature_names = ["X"]
                    target_name = "Y"
                    st.info("🎯 **Learning Goal**: Understand how slope and intercept affect the line!")

                elif "Polynomial" in dataset_type:
                    degree = st.slider("Polynomial degree", 2, 4, 2)
                    X, y = DatasetGenerator.generate_polynomial(n_samples, degree, noise_level, random_seed)
                    feature_names = ["X"]
                    target_name = "Y"
                    st.info("🎯 **Learning Goal**: See why linear models struggle with curves!")

                elif "Multiple" in dataset_type:
                    n_features = st.slider("Number of features", 2, 5, 3)
                    X, y = DatasetGenerator.generate_multivariate(n_samples, n_features, noise_level, random_seed)
                    feature_names = [f"Feature_{i+1}" for i in range(n_features)]
                    target_name = "Y"
                    st.info("🎯 **Learning Goal**: Learn how multiple features combine!")

                elif "Outliers" in dataset_type:
                    n_outliers = st.slider("Number of outliers", 0, 20, 5)
                    X, y = DatasetGenerator.generate_outliers(n_samples, n_outliers, noise_level, random_seed)
                    feature_names = ["X"]
                    target_name = "Y"
                    st.info("🎯 **Learning Goal**: See how outliers affect different models!")

                else:  # Non-linear
                    X, y = DatasetGenerator.generate_nonlinear(n_samples, noise_level, random_seed)
                    feature_names = ["X"]
                    target_name = "Y"
                    st.info("🎯 **Learning Goal**: Discover the limits of linear regression!")

            except Exception as e:
                st.error(f"Error generating dataset: {e}")
                return

        else:
            # File upload with enhanced guidance
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=['csv'],
                help="Upload a CSV file with numeric columns for regression analysis"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)

                    # Show data preview
                    st.write("**Data Preview:**")
                    st.dataframe(df.head())

                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                    if len(numeric_cols) < 2:
                        st.error("Need at least 2 numeric columns for regression!")
                        return

                    target_name = st.selectbox("Choose target variable (Y):", numeric_cols)
                    feature_names = st.multiselect(
                        "Choose feature variables (X):",
                        [col for col in numeric_cols if col != target_name],
                        default=[col for col in numeric_cols if col != target_name][:min(3, len(numeric_cols)-1)]
                    )

                    if not feature_names:
                        st.warning("Please select at least one feature!")
                        return

                    # Handle missing values
                    if df[feature_names + [target_name]].isnull().any().any():
                        st.warning("Missing values detected. They will be removed.")
                        df_clean = df[feature_names + [target_name]].dropna()
                        if len(df_clean) < 3:
                            st.error("Not enough data after removing missing values!")
                            return
                        X = df_clean[feature_names].values
                        y = df_clean[target_name].values
                    else:
                        X = df[feature_names].values
                        y = df[target_name].values

                    # Additional data validation
                    X, y = DataValidator.clean_data(X, y)

                    if len(X) < 3:
                        st.error("Not enough valid data points for regression!")
                        return

                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    return
            else:
                return

    # Data validation check
    is_valid, validation_message = DataValidator.validate_data(X, y)
    if not is_valid:
        st.error(validation_message)
        return

    # Visual data preview with enhanced analysis
    st.markdown("### 👀 Data Visualization & Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Create enhanced scatter plot
        if X.shape[1] == 1:
            fig = go.Figure()

            # Color points by target value
            fig.add_trace(go.Scatter(
                x=X.flatten(),
                y=y,
                mode='markers',
                marker=dict(
                    color=y,
                    colorscale=[[0, '#004E89'], [1, '#FF6B35']],
                    size=8,
                    opacity=0.7,
                    colorbar=dict(title="Target Value"),
                    line=dict(width=1, color='white')
                ),
                name='Data Points',
                hovertemplate=f'{feature_names[0]}: %{{x:.3f}}<br>{target_name}: %{{y:.3f}}<extra></extra>'
            ))

            fig.update_layout(
                title="Data Distribution (Color = Target Value)",
                xaxis_title=feature_names[0],
                yaxis_title=target_name,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        elif X.shape[1] == 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=y,
                mode='markers',
                marker=dict(
                    color=y,
                    colorscale=[[0, '#004E89'], [1, '#FF6B35']],
                    size=5,
                    opacity=0.7,
                    colorbar=dict(title=target_name)
                ),
                name='Data Points',
                hovertemplate=f'{feature_names[0]}: %{{x:.3f}}<br>{feature_names[1]}: %{{y:.3f}}<br>{target_name}: %{{z:.3f}}<extra></extra>'
            ))
            fig.update_layout(
                title="3D Data Visualization",
                scene=dict(
                    xaxis_title=feature_names[0],
                    yaxis_title=feature_names[1],
                    zaxis_title=target_name
                ),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # For higher dimensions, show pairwise correlations
            st.info("📊 For high-dimensional data, showing correlation matrix and feature distributions.")

            # Feature distributions
            fig = make_subplots(
                rows=1, cols=min(3, len(feature_names)),
                subplot_titles=feature_names[:3]
            )

            for i, feature in enumerate(feature_names[:3]):
                fig.add_trace(
                    go.Histogram(x=X[:, i], name=feature, opacity=0.7),
                    row=1, col=i+1
                )

            fig.update_layout(height=300, showlegend=False, title_text="Feature Distributions")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Enhanced data statistics
        st.markdown("#### 📈 Data Statistics")

        n_samples, n_features = X.shape

        # Basic statistics
        st.markdown(f"""
        <div class="metric-card">
            <strong>Samples:</strong> {n_samples}<br>
            <strong>Features:</strong> {n_features}<br>
            <strong>Target Range:</strong> {y.min():.3f} to {y.max():.3f}<br>
            <strong>Target Mean:</strong> {y.mean():.3f}<br>
            <strong>Target Std:</strong> {y.std():.3f}
        </div>
        """, unsafe_allow_html=True)

        # Data quality indicators
        outlier_threshold = 3
        z_scores = np.abs(stats.zscore(y))
        n_outliers = np.sum(z_scores > outlier_threshold)

        if n_outliers > 0:
            st.warning(f"⚠️ {n_outliers} potential outliers detected (|z-score| > {outlier_threshold})")
        else:
            st.success("✅ No obvious outliers detected")

        # Correlation matrix for multiple features
        if n_features > 1:
            corr_data = np.column_stack([X, y])
            corr_matrix = np.corrcoef(corr_data.T)

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=feature_names + [target_name],
                y=feature_names + [target_name],
                colorscale=[[0, '#004E89'], [0.5, 'white'], [1, '#FF6B35']],
                zmid=0,
                text=np.round(corr_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            fig_corr.update_layout(
                title="Feature Correlations",
                height=300,
                xaxis={'side': 'bottom'}
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # Model selection with guided learning
    st.markdown("### 🤖 Step 2: Choose Your Model & Parameters")

    # Contextual model recommendations
    if X.shape[1] == 1:
        st.info("💡 **Recommendation**: Start with Gradient Descent to see the learning process!")
    elif n_outliers > 0:
        st.info("💡 **Recommendation**: Try Huber loss in Gradient Descent to handle outliers!")
    else:
        st.info("💡 **Recommendation**: Compare OLS with regularized models to see the differences!")

    model_tabs = st.tabs(["🎯 Gradient Descent", "📊 Ordinary Least Squares", "🔧 Regularized Models"])

    with model_tabs[0]:
        st.markdown("#### Real-time Learning Visualization")

        col1, col2 = st.columns([1, 1])

        with col1:
            learning_rate = st.slider(
                "Learning Rate (α)",
                min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f",
                help="🎯 Higher = faster learning, but might overshoot! Try 0.001 for stability, 0.1 for speed."
            )

            max_iterations = st.slider(
                "Max Iterations",
                min_value=50, max_value=2000, value=500, step=50,
                help="🔄 More iterations = more learning time. Watch the cost curve!"
            )

        with col2:
            cost_function = st.selectbox(
                "Cost Function",
                ["MSE", "MAE", "Huber"],
                help="🎯 MSE: sensitive to outliers, MAE: robust, Huber: balanced"
            )

            if cost_function == "Huber":
                delta = st.slider("Huber Delta", 0.1, 5.0, 1.0, 0.1,
                                help="🎚️ Threshold between MSE and MAE behavior")
            else:
                delta = 1.0

            show_realtime = st.checkbox(
                "Show Real-time Training",
                value=True,
                help="🎬 Watch the model learn step by step!"
            )

        if st.button("🚀 Start Training!", type="primary"):
            # Update learning state
            st.session_state.learning_state.experiments_count += 1

            # Real-time training visualization
            if show_realtime and X.shape[1] == 1:
                progress_bar = st.progress(0)
                status_text = st.empty()
                realtime_plot = st.empty()
                cost_plot = st.empty()

                cost_history_realtime = []

                # Training progress callback
                def progress_callback(iteration, cost, coefficients, intercept):
                    progress = min(1.0, iteration / max_iterations)
                    progress_bar.progress(progress)
                    status_text.text(f"Iteration {iteration}: Cost = {cost:.6f}")

                    cost_history_realtime.append(cost)

                    # Update plots every 25 iterations to balance responsiveness and performance
                    if iteration % 25 == 0:
                        # Prediction plot
                        fig = go.Figure()

                        # Data points
                        fig.add_trace(go.Scatter(
                            x=X.flatten(),
                            y=y,
                            mode='markers',
                            marker=dict(color='#004E89', opacity=0.6, size=8),
                            name='Data'
                        ))

                        # Current prediction line
                        x_line = np.linspace(X.min(), X.max(), 100)
                        y_line = coefficients[0] * x_line + intercept
                        fig.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode='lines',
                            line=dict(color='#FF6B35', width=3),
                            name=f'Prediction (iter {iteration})'
                        ))

                        fig.update_layout(
                            title=f"Learning Progress - Iteration {iteration}",
                            height=300,
                            showlegend=True
                        )
                        realtime_plot.plotly_chart(fig, use_container_width=True)

                        # Cost history plot
                        if len(cost_history_realtime) > 1:
                            fig_cost = go.Figure()
                            fig_cost.add_trace(go.Scatter(
                                x=list(range(len(cost_history_realtime))),
                                y=cost_history_realtime,
                                mode='lines',
                                line=dict(color='#FF6B35', width=2),
                                name='Cost'
                            ))
                            fig_cost.update_layout(
                                title="Cost Function Progress",
                                xaxis_title="Iteration",
                                yaxis_title="Cost",
                                height=200
                            )
                            cost_plot.plotly_chart(fig_cost, use_container_width=True)

                # Train model with callback
                model = GradientDescentRegressor(
                    learning_rate=learning_rate,
                    max_iterations=max_iterations,
                    cost_function=cost_function.lower()
                )

                with st.spinner("Training model..."):
                    results = model.fit(X, y, delta=delta, progress_callback=progress_callback)

                progress_bar.progress(1.0)
                if results.converged:
                    status_text.success(f"✅ Training Complete! Final cost: {results.final_cost:.6f}")
                else:
                    status_text.warning(f"⚠️ Training stopped early. Final cost: {results.final_cost:.6f}")

            else:
                # Train without real-time updates
                model = GradientDescentRegressor(
                    learning_rate=learning_rate,
                    max_iterations=max_iterations,
                    cost_function=cost_function.lower()
                )

                with st.spinner("Training model..."):
                    results = model.fit(X, y, delta=delta)

                if results.converged:
                    st.success(f"✅ Training completed in {results.training_time:.2f} seconds!")
                else:
                    st.warning("⚠️ Training did not converge. Try reducing learning rate or increasing iterations.")

            # Store results for comparison
            if st.session_state.previous_results is not None:
                # Show parameter impact
                try:
                    impact_fig = VisualFeedbackSystem.create_parameter_impact_viz(
                        st.session_state.previous_results, results, "Learning Parameters"
                    )
                    st.plotly_chart(impact_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create comparison plot: {e}")

            st.session_state.previous_results = results
            st.session_state.learning_state.last_r_squared = results.r_squared
            if results.r_squared > st.session_state.learning_state.best_r_squared:
                st.session_state.learning_state.best_r_squared = results.r_squared

            # Display results
            display_results(results, X, y, feature_names, target_name, "Gradient Descent")

    with model_tabs[1]:
        st.markdown("#### Analytical Solution")
        st.info("🧮 **How it works**: Solves the normal equations directly - no iterations needed!")

        # Show the mathematical formula
        st.latex(r"\hat{\beta} = (X^T X)^{-1} X^T y")

        if st.button("⚡ Compute OLS Solution", type="primary"):
            st.session_state.learning_state.experiments_count += 1

            with st.spinner("Computing OLS solution..."):
                model = OLSRegressor()
                results = model.fit(X, y)

            if results.training_time < 0.01:
                st.success(f"✅ Solution computed instantly!")
            else:
                st.success(f"✅ Solution computed in {results.training_time:.3f} seconds!")

            # Store results
            st.session_state.previous_results = results
            st.session_state.learning_state.last_r_squared = results.r_squared
            if results.r_squared > st.session_state.learning_state.best_r_squared:
                st.session_state.learning_state.best_r_squared = results.r_squared

            display_results(results, X, y, feature_names, target_name, "OLS")

    with model_tabs[2]:
        st.markdown("#### Regularization Techniques")

        col1, col2 = st.columns(2)

        with col1:
            reg_type = st.selectbox(
                "Regularization Type",
                ["Ridge (L2)", "Lasso (L1)", "Elastic Net (L1+L2)"],
                help="🔧 Regularization prevents overfitting by constraining coefficients"
            )

            alpha = st.slider(
                "Regularization Strength (α)",
                min_value=0.001, max_value=10.0, value=1.0, step=0.001, format="%.3f",
                help="🎚️ Higher values = more regularization = simpler model"
            )

        with col2:
            if "Elastic" in reg_type:
                l1_ratio = st.slider(
                    "L1 Ratio",
                    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                    help="⚖️ Balance between Ridge (0) and Lasso (1)"
                )
            else:
                l1_ratio = 0.5

            max_iter = st.slider(
                "Max Iterations",
                min_value=100, max_value=5000, value=2000, step=100,
                help="🔄 Maximum iterations for convergence"
            )

        if st.button("🔧 Apply Regularization", type="primary"):
            st.session_state.learning_state.experiments_count += 1

            try:
                with st.spinner(f"Training {reg_type} model..."):
                    start_time = time.time()

                    if "Ridge" in reg_type:
                        model = Ridge(alpha=alpha, max_iter=max_iter)
                    elif "Lasso" in reg_type:
                        model = Lasso(alpha=alpha, max_iter=max_iter, tol=1e-4)
                    else:  # Elastic Net
                        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=1e-4)

                    model.fit(X, y)
                    training_time = time.time() - start_time

                    # Check convergence
                    converged = True
                    if hasattr(model, 'n_iter_'):
                        if isinstance(model.n_iter_, np.ndarray):
                            converged = np.all(model.n_iter_ < max_iter)
                        else:
                            converged = model.n_iter_ < max_iter

                    if not converged:
                        st.warning("⚠️ Model may not have converged. Try increasing max iterations.")

                # Convert sklearn results to our format
                predictions = model.predict(X)
                residuals = y - predictions

                # Calculate additional statistics
                stats_dict = BaseRegressor()._calculate_statistics(
                    None, X, y, predictions, model.coef_, model.intercept_
                )

                results = ModelResults(
                    coefficients=model.coef_,
                    intercept=model.intercept_,
                    predictions=predictions,
                    residuals=residuals,
                    converged=converged,
                    training_time=training_time,
                    **stats_dict
                )

                st.success(f"✅ {reg_type} model trained successfully in {training_time:.3f} seconds!")

                # Store results
                st.session_state.previous_results = results
                st.session_state.learning_state.last_r_squared = results.r_squared
                if results.r_squared > st.session_state.learning_state.best_r_squared:
                    st.session_state.learning_state.best_r_squared = results.r_squared

                display_results(results, X, y, feature_names, target_name, reg_type)

            except Exception as e:
                st.error(f"Error training {reg_type} model: {e}")
                st.info("💡 Try reducing the regularization strength or increasing max iterations.")

def display_results(results: ModelResults, X: np.ndarray, y: np.ndarray,
                   feature_names: List[str], target_name: str, model_type: str):
    """Display results with enhanced visualizations and learning insights."""

    st.markdown("### 🎯 Results & Analysis")

    # Performance metrics with visual emphasis and better formatting
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        r2_color = "positive-value" if results.r_squared > 0.7 else "negative-value" if results.r_squared < 0.3 else "neutral-value"
        r2_emoji = "🎉" if results.r_squared > 0.8 else "👍" if results.r_squared > 0.6 else "🤔" if results.r_squared > 0.3 else "😞"
        st.markdown(f"""
        <div class="metric-card">
            <h4>{r2_emoji} R² Score</h4>
            <p class="{r2_color}" style="font-size: 2em;">{results.r_squared:.4f}</p>
            <small>{"Excellent!" if results.r_squared > 0.8 else "Good" if results.r_squared > 0.6 else "Fair" if results.r_squared > 0.3 else "Poor"}</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📏 RMSE</h4>
            <p style="font-size: 2em; color: #666;">{results.rmse:.4f}</p>
            <small>Root Mean Square Error</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📐 MAE</h4>
            <p style="font-size: 2em; color: #666;">{results.mae:.4f}</p>
            <small>Mean Absolute Error</small>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        convergence_status = "✅ Converged" if results.converged else "⚠️ Not Converged"
        convergence_color = "#28a745" if results.converged else "#ffc107"
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚙️ Status</h4>
            <p style="font-size: 1.2em; color: {convergence_color};">{convergence_status}</p>
            <small>Training: {results.training_time:.3f}s</small>
        </div>
        """, unsafe_allow_html=True)

    # Additional metrics row
    if hasattr(results, 'aic') and results.aic != 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Adjusted R²", f"{results.adj_r_squared:.4f}")
        with col2:
            st.metric("AIC", f"{results.aic:.2f}" if not np.isinf(results.aic) else "Perfect Fit")
        with col3:
            st.metric("BIC", f"{results.bic:.2f}" if not np.isinf(results.bic) else "Perfect Fit")
        with col4:
            if hasattr(results, 'final_cost'):
                st.metric("Final Cost", f"{results.final_cost:.6f}")

    # Coefficient visualization as network
    if len(results.coefficients) > 0:
        st.markdown("#### 🕸️ Model Structure Visualization")
        try:
            network_fig = VisualFeedbackSystem.create_coefficient_network_viz(
                results.coefficients, feature_names, results.intercept
            )
            st.plotly_chart(network_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create network visualization: {e}")

    # Main prediction visualization
    st.markdown("#### 📈 Predictions vs Reality")

    try:
        if X.shape[1] == 1:
            # Enhanced 2D plot with residuals
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Predictions vs Actual', 'Residual Plot'),
                column_widths=[0.6, 0.4]
            )

            # Left plot: Data and prediction line
            residual_colors = results.residuals
            fig.add_trace(go.Scatter(
                x=X.flatten(),
                y=y,
                mode='markers',
                marker=dict(
                    color=residual_colors,
                    colorscale=[[0, '#004E89'], [0.5, 'white'], [1, '#FF6B35']],
                    size=8,
                    opacity=0.7,
                    colorbar=dict(title="Residuals", x=0.45)
                ),
                name='Actual Data',
                hovertemplate=f'{feature_names[0]}: %{{x:.3f}}<br>{target_name}: %{{y:.3f}}<br>Residual: %{{marker.color:.3f}}<extra></extra>'
            ), row=1, col=1)

            # Prediction line
            x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_line = x_line.flatten() * results.coefficients[0] + results.intercept

            fig.add_trace(go.Scatter(
                x=x_line.flatten(),
                y=y_line,
                mode='lines',
                line=dict(color='red', width=3),
                name='Prediction Line'
            ), row=1, col=1)

            # Right plot: Residuals vs fitted
            fig.add_trace(go.Scatter(
                x=results.predictions,
                y=results.residuals,
                mode='markers',
                marker=dict(color='#FF6B35', opacity=0.6, size=6),
                name='Residuals',
                hovertemplate='Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>'
            ), row=1, col=2)

            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

            fig.update_xaxes(title_text=feature_names[0], row=1, col=1)
            fig.update_yaxes(title_text=target_name, row=1, col=1)
            fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
            fig.update_yaxes(title_text="Residuals", row=1, col=2)

            fig.update_layout(
                title=f"{model_type} Regression Results",
                height=500,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        elif X.shape[1] == 2:
            # 3D surface plot for 2 features
            fig = go.Figure()

            # Data points
            fig.add_trace(go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=y,
                mode='markers',
                marker=dict(
                    color=results.residuals,
                    colorscale=[[0, '#004E89'], [0.5, 'white'], [1, '#FF6B35']],
                    size=5,
                    opacity=0.8,
                    colorbar=dict(title="Residuals")
                ),
                name='Actual Data'
            ))

            # Prediction surface
            x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
            y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
            xx, yy = np.meshgrid(x_range, y_range)
            zz = results.coefficients[0] * xx + results.coefficients[1] * yy + results.intercept

            fig.add_trace(go.Surface(
                x=xx, y=yy, z=zz,
                opacity=0.6,
                colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
                name='Prediction Surface',
                showscale=False
            ))

            fig.update_layout(
                title=f"{model_type} 3D Regression Results",
                scene=dict(
                    xaxis_title=feature_names[0],
                    yaxis_title=feature_names[1],
                    zaxis_title=target_name
                ),
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            # For higher dimensions, show actual vs predicted
            fig = go.Figure()

            # Perfect prediction line
            min_val = min(y.min(), results.predictions.min())
            max_val = max(y.max(), results.predictions.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction'
            ))

            # Actual vs predicted points
            fig.add_trace(go.Scatter(
                x=y,
                y=results.predictions,
                mode='markers',
                marker=dict(
                    color=np.abs(results.residuals),
                    colorscale=[[0, '#004E89'], [1, '#FF6B35']],
                    size=8,
                    opacity=0.7,
                    colorbar=dict(title="Absolute Residuals")
                ),
                name='Predictions',
                hovertemplate='Actual: %{x:.3f}<br>Predicted: %{y:.3f}<br>Residual: %{marker.color:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"{model_type} - Actual vs Predicted",
                xaxis_title=f"Actual {target_name}",
                yaxis_title=f"Predicted {target_name}",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating prediction visualization: {e}")

    # Cost history for gradient descent
    if hasattr(results, 'cost_history') and results.cost_history and len(results.cost_history) > 1:
        st.markdown("#### 📉 Learning Curve")

        try:
            fig_cost = go.Figure()
            iterations = list(range(len(results.cost_history)))

            fig_cost.add_trace(go.Scatter(
                x=iterations,
                y=results.cost_history,
                mode='lines',
                line=dict(color='#FF6B35', width=2),
                name='Cost',
                hovertemplate='Iteration: %{x}<br>Cost: %{y:.6f}<extra></extra>'
            ))

            # Add annotations for key points
            if len(results.cost_history) > 10:
                min_cost_idx = np.argmin(results.cost_history)
                fig_cost.add_annotation(
                    x=min_cost_idx,
                    y=results.cost_history[min_cost_idx],
                    text=f"Minimum: {results.cost_history[min_cost_idx]:.6f}",
                    showarrow=True,
                    arrowhead=2,
                    bgcolor="white",
                    bordercolor="black"
                )

                # Show convergence behavior
                if len(results.cost_history) > 50:
                    final_50_std = np.std(results.cost_history[-50:])
                    if final_50_std < results.cost_history[0] * 0.001:
                        fig_cost.add_annotation(
                            x=len(results.cost_history) * 0.8,
                            y=max(results.cost_history) * 0.8,
                            text="✅ Converged",
                            showarrow=False,
                            bgcolor="lightgreen",
                            bordercolor="green"
                        )

            fig_cost.update_layout(
                title="Cost Function During Training",
                xaxis_title="Iteration",
                yaxis_title="Cost",
                height=350,
                yaxis_type="log" if max(results.cost_history) / min(results.cost_history) > 100 else "linear"
            )

            st.plotly_chart(fig_cost, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not create learning curve: {e}")

    # Comprehensive residual analysis
    if len(results.residuals) > 10:  # Only show for sufficient data
        st.markdown("#### 🔍 Residual Analysis")

        try:
            residual_fig = VisualFeedbackSystem.create_residual_analysis(
                results.residuals, results.predictions
            )
            st.plotly_chart(residual_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create residual analysis: {e}")

    # Interactive learning insights
    st.markdown("#### 🧠 Learning Insights")

    insights_tabs = st.tabs(["🎯 What Happened?", "🔍 Coefficient Analysis", "🚀 Try Next"])

    with insights_tabs[0]:
        # Generate contextual explanations
        if results.r_squared > 0.9:
            st.success("🎉 **Outstanding fit!** Your model captures almost all the variation in the data.")
        elif results.r_squared > 0.8:
            st.success("🎉 **Excellent fit!** Your model captures the underlying pattern very well.")
        elif results.r_squared > 0.6:
            st.info("👍 **Good fit!** The model explains most of the variation in your data.")
        elif results.r_squared > 0.3:
            st.warning("🤔 **Moderate fit.** Consider feature engineering or trying a different model.")
        else:
            st.error("📈 **Poor fit.** The linear model might not be suitable for this data.")

        # Model-specific insights
        if model_type == "Gradient Descent":
            if not results.converged:
                st.warning("⚠️ **Training didn't converge!** Try reducing the learning rate or increasing iterations.")
                st.info("💡 **Tip**: Start with learning rate 0.001 and gradually increase if training is too slow.")
            else:
                st.success("✅ **Training converged successfully!** The algorithm found a good solution.")
                if hasattr(results, 'cost_history') and len(results.cost_history) > 100:
                    final_improvement = (results.cost_history[0] - results.cost_history[-1]) / results.cost_history[0] * 100
                    st.info(f"📈 **Cost improved by {final_improvement:.1f}%** during training.")

        elif "Ridge" in model_type or "Lasso" in model_type or "Elastic" in model_type:
            # Count zero coefficients for Lasso
            if "Lasso" in model_type:
                zero_coefs = np.sum(np.abs(results.coefficients) < 1e-6)
                if zero_coefs > 0:
                    st.info(f"🎯 **Feature Selection**: Lasso set {zero_coefs} coefficients to zero, effectively removing those features.")
                else:
                    st.info("🎯 **No features removed**: Try increasing regularization strength to see feature selection in action.")

        # Feature-specific insights
        if len(results.coefficients) == 1:
            coef = results.coefficients[0]
            if abs(coef) > 2:
                strength = "strong"
            elif abs(coef) > 0.5:
                strength = "moderate"
            else:
                strength = "weak"

            direction = "positive" if coef > 0 else "negative"
            st.info(f"📈 **{strength.title()} {direction} relationship**: As {feature_names[0]} increases by 1 unit, {target_name} {'increases' if coef > 0 else 'decreases'} by {abs(coef):.3f} units on average.")

        # Data quality insights
        residual_std = np.std(results.residuals)
        target_std = np.std(y)
        noise_ratio = residual_std / target_std if target_std > 0 else 0

        if noise_ratio < 0.1:
            st.success("🎯 **Low noise**: Your model captures the signal very well!")
        elif noise_ratio < 0.3:
            st.info("🎯 **Moderate noise**: Some unexplained variation remains.")
        else:
            st.warning("🎯 **High noise**: Consider feature engineering or collecting more relevant features.")

    with insights_tabs[1]:
        # Detailed coefficient analysis with enhanced formatting
        coef_data = []
        for i, (name, coef) in enumerate(zip(feature_names, results.coefficients)):
            importance = results.feature_importance[i] if hasattr(results, 'feature_importance') and results.feature_importance is not None else abs(coef) / (np.sum(np.abs(results.coefficients)) + 1e-8)

            # Determine effect strength
            if abs(coef) > 2:
                magnitude = "Very Strong"
            elif abs(coef) > 1:
                magnitude = "Strong"
            elif abs(coef) > 0.1:
                magnitude = "Medium"
            else:
                magnitude = "Weak"

            coef_data.append({
                'Feature': name,
                'Coefficient': f"{coef:.4f}",
                'Importance': f"{importance:.3f}",
                'Effect': 'Positive ↗️' if coef > 0 else 'Negative ↘️',
                'Magnitude': magnitude
            })

        coef_df = pd.DataFrame(coef_data)

        # Display as a nice table
        st.dataframe(
            coef_df,
            use_container_width=True,
            hide_index=True
        )

        # Intercept explanation with context
        intercept_interpretation = f"🎯 **Intercept**: {results.intercept:.4f}"
        if abs(results.intercept) > 0.001:
            intercept_interpretation += f" - This is the predicted {target_name} when all features are zero."
        else:
            intercept_interpretation += " - Nearly zero intercept suggests the relationship passes through the origin."

        st.info(intercept_interpretation)

        # Statistical significance for OLS
        if hasattr(results, 'p_values') and results.p_values is not None:
            st.markdown("#### 📊 Statistical Significance")
            sig_data = []
            for i, (name, p_val) in enumerate(zip(['Intercept'] + feature_names, results.p_values)):
                if p_val < 0.001:
                    significance = "*** (p < 0.001)"
                elif p_val < 0.01:
                    significance = "** (p < 0.01)"
                elif p_val < 0.05:
                    significance = "* (p < 0.05)"
                else:
                    significance = "Not significant"

                sig_data.append({
                    'Parameter': name,
                    'p-value': f"{p_val:.4f}",
                    'Significance': significance
                })

            sig_df = pd.DataFrame(sig_data)
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
            st.caption("*** p<0.001, ** p<0.01, * p<0.05")

    with insights_tabs[2]:
        # Enhanced suggestions for further exploration
        st.markdown("#### 🚀 Experiment Ideas:")

        suggestions = []

        # Model-specific suggestions
        if model_type == "Gradient Descent":
            suggestions.extend([
                "🎚️ **Learning Rate**: Try 0.001 (stable), 0.01 (balanced), 0.1 (fast), 1.0 (risky)",
                "🔄 **Cost Functions**: Compare MSE (standard) vs MAE (robust) vs Huber (balanced)",
                "⏱️ **Iterations**: Watch how the cost curve changes with more/fewer iterations"
            ])

            if results.r_squared < 0.5:
                suggestions.append("📐 **Try polynomial features**: Your data might have non-linear patterns")

        elif model_type == "OLS":
            suggestions.extend([
                "🔧 **Compare with regularization**: See how Ridge/Lasso change the coefficients",
                "📊 **Check assumptions**: Look at residual plots for patterns",
                "🎯 **Feature engineering**: Try polynomial or interaction terms"
            ])

        # Performance-based suggestions
        if results.r_squared < 0.7:
            suggestions.extend([
                "📐 **Polynomial features**: Add X², X³ terms to capture curves",
                "🔧 **Regularization**: Prevent overfitting with Ridge/Lasso",
                "🧹 **Outlier detection**: Check for and handle unusual data points"
            ])

        if len(feature_names) > 1:
            suggestions.extend([
                "🎯 **Feature selection**: Remove less important features",
                "⚖️ **Ridge vs Lasso**: Compare L2 vs L1 regularization effects",
                "🔍 **Interaction terms**: Try X₁ × X₂ combinations"
            ])

        # Data-specific suggestions
        if len(y) < 100:
            suggestions.append("📊 **More data**: Small datasets benefit from regularization")

        if np.std(results.residuals) / np.std(y) > 0.3:
            suggestions.append("🎯 **Feature engineering**: High residual variance suggests missing features")

        for i, suggestion in enumerate(suggestions, 1):
            st.markdown(f"{i}. {suggestion}")

        # Hypothesis formation with better guidance
        st.markdown("#### 🤔 Form a Hypothesis:")

        # Provide example hypotheses based on current results
        example_hypotheses = []
        if model_type == "Gradient Descent":
            example_hypotheses.extend([
                "If I increase the learning rate, the model will converge faster but might be less stable",
                "Changing from MSE to MAE will make the model more robust to outliers",
                "Adding more iterations will improve the final cost"
            ])

        if results.r_squared < 0.6:
            example_hypotheses.extend([
                "Adding polynomial features will improve the R² score",
                "The relationship might not be linear - I should try a different model",
                "There might be outliers affecting the fit"
            ])

        if example_hypotheses:
            st.markdown("**Example hypotheses:**")
            for hyp in example_hypotheses[:3]:  # Show max 3 examples
                st.markdown(f"• *{hyp}*")

        hypothesis = st.text_area(
            "What do you think will happen if you change the parameters?",
            placeholder="e.g., 'If I increase the learning rate, the model will converge faster but might be less stable...'",
            height=100
        )

        if hypothesis:
            st.session_state.learning_state.current_hypothesis = hypothesis
            st.success("💡 Hypothesis recorded! Now test it by changing parameters.")

            # Add to reflection notes
            if hypothesis not in st.session_state.learning_state.reflection_notes:
                st.session_state.learning_state.reflection_notes.append(f"Experiment {st.session_state.learning_state.experiments_count}: {hypothesis}")

def main():
    """Main application entry point with error handling."""
    try:
        create_interactive_playground()

        # Enhanced learning progress tracking in sidebar
        if hasattr(st.session_state, 'learning_state'):
            learning_state = st.session_state.learning_state

            with st.sidebar:
                st.markdown("### 📚 Your Learning Journey")

                # Progress metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Experiments", learning_state.experiments_count)
                with col2:
                    st.metric("Best R²", f"{learning_state.best_r_squared:.3f}")

                # Progress bar
                progress = min(1.0, learning_state.experiments_count / 10)  # 10 experiments = 100%
                st.progress(progress)
                st.caption(f"Progress: {progress*100:.0f}% (Target: 10 experiments)")

                # Current hypothesis
                if learning_state.current_hypothesis:
                    st.markdown("#### 🤔 Current Hypothesis")
                    st.info(learning_state.current_hypothesis)

                # Learning tips based on progress
                st.markdown("#### 💡 Learning Tips")
                if learning_state.experiments_count == 0:
                    st.info("🚀 **Getting Started**: Try the Linear Relationship dataset first!")
                elif learning_state.experiments_count < 3:
                    st.info("🎚️ **Explore Parameters**: Try different learning rates to see their effects!")
                elif learning_state.experiments_count < 5:
                    st.info("🔄 **Compare Models**: Try both Gradient Descent and OLS on the same data!")
                elif learning_state.experiments_count < 8:
                    st.info("🔧 **Advanced Techniques**: Experiment with regularization methods!")
                else:
                    st.success("🎉 **Expert Level**: You're becoming a regression master! Try your own data!")

                # Reflection notes
                if learning_state.reflection_notes:
                    st.markdown("#### 📝 Your Reflections")
                    for note in learning_state.reflection_notes[-3:]:  # Show last 3
                        st.caption(note)

                # Reset button
                if st.button("🔄 Reset Progress", help="Clear all learning progress"):
                    st.session_state.learning_state = LearningState()
                    st.session_state.previous_results = None
                    st.rerun()

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please refresh the page and try again. If the problem persists, check your data format.")

        # Show error details in expander for debugging
        with st.expander("🔧 Error Details (for debugging)"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()