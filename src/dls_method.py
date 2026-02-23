"""
Duckworth-Lewis-Stern (DLS) Method Implementation
Parametric model: Z(u, w) = Z0(w) * [1 - exp(-b(w) * u)]
where u = overs remaining, w = wickets fallen
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "results" / "models"


class DLSModel:
    """
    Parametric DLS resource model.

    The DLS method models the proportion of resources remaining as:
        Z(u, w) = Z0(w) * [1 - exp(-b(w) * u)]

    where:
        u = overs remaining
        w = wickets lost (0-9)
        Z0(w) = asymptotic maximum runs available with w wickets lost
        b(w) = exponential decay parameter for w wickets lost

    Resource percentage = Z(u, w) / Z(U, 0)  (where U = total overs)
    """

    def __init__(self, overs_limit: int = 50):
        self.overs_limit = overs_limit
        # G50: average score in uninterrupted innings (approx 245 for modern ODIs)
        self.G50 = 245.0 if overs_limit == 50 else 155.0

        # Parameters per wicket state (0-9 wickets lost)
        self.Z0 = np.zeros(10)
        self.b = np.zeros(10)
        self.fitted = False

        # Default professional DLS parameters (approximate)
        self._set_default_params()

    def _set_default_params(self):
        """Set default DLS parameters as starting point / fallback."""
        if self.overs_limit == 50:
            # Approximate standard DLS Z0 values (runs available)
            self.Z0 = np.array([
                295.0,  # 0 wickets lost
                275.0,  # 1 wicket lost
                255.0,  # 2 wickets lost
                230.0,  # 3 wickets lost
                200.0,  # 4 wickets lost
                165.0,  # 5 wickets lost
                125.0,  # 6 wickets lost
                85.0,   # 7 wickets lost
                50.0,   # 8 wickets lost
                20.0,   # 9 wickets lost
            ])
            self.b = np.array([
                0.035, 0.040, 0.045, 0.052, 0.060,
                0.072, 0.090, 0.120, 0.180, 0.350,
            ])
        else:
            # T20 parameters (proportionally adjusted)
            self.Z0 = np.array([
                190.0, 175.0, 160.0, 145.0, 125.0,
                100.0, 75.0, 50.0, 30.0, 12.0,
            ])
            self.b = np.array([
                0.070, 0.080, 0.090, 0.105, 0.125,
                0.150, 0.190, 0.260, 0.400, 0.750,
            ])

    @staticmethod
    def _resource_function(u, Z0, b):
        """Z(u) = Z0 * [1 - exp(-b * u)]"""
        return Z0 * (1.0 - np.exp(-b * u))

    def fit(self, snapshots_df: pd.DataFrame):
        """
        Fit Z0 and b parameters from actual match data.

        Uses first innings data: at each (overs_remaining, wickets_fallen),
        the "runs still to come" = final_total - current_score.
        """
        df = snapshots_df.copy()
        df["runs_remaining"] = df["final_total"] - df["current_score"]
        df = df[df["runs_remaining"] >= 0]

        # Fit for each wicket state
        for w in range(10):
            subset = df[df["wickets_fallen"] == w]

            if len(subset) < 10:
                logger.warning(f"Wickets={w}: only {len(subset)} samples, using defaults.")
                continue

            u = subset["overs_remaining"].values.astype(float)
            z = subset["runs_remaining"].values.astype(float)

            # Remove zero-overs-remaining rows (can't fit)
            mask = u > 0
            u, z = u[mask], z[mask]

            if len(u) < 5:
                continue

            try:
                popt, pcov = curve_fit(
                    self._resource_function,
                    u,
                    z,
                    p0=[self.Z0[w], self.b[w]],
                    bounds=([1.0, 0.001], [600.0, 2.0]),
                    maxfev=10000,
                )
                self.Z0[w] = popt[0]
                self.b[w] = popt[1]
                logger.info(
                    f"Wickets={w}: Z0={popt[0]:.2f}, b={popt[1]:.4f} "
                    f"(fitted from {len(u)} points)"
                )
            except RuntimeError as e:
                logger.warning(f"Wickets={w}: curve_fit failed ({e}), using defaults.")

        self.fitted = True

        # Recalculate G50 based on fitted params
        self.G50 = self._resource_function(self.overs_limit, self.Z0[0], self.b[0])
        logger.info(f"Fitted G50 = {self.G50:.1f}")

    def resource_remaining(self, overs_remaining: float, wickets_fallen: int) -> float:
        """
        Calculate resource percentage remaining.
        Returns value between 0 and 100.
        """
        w = min(max(int(wickets_fallen), 0), 9)
        u = max(float(overs_remaining), 0.0)

        z_current = self._resource_function(u, self.Z0[w], self.b[w])
        z_full = self._resource_function(self.overs_limit, self.Z0[0], self.b[0])

        if z_full == 0:
            return 0.0

        return min(100.0, max(0.0, (z_current / z_full) * 100.0))

    def resource_used(self, overs_completed: float, wickets_fallen: int) -> float:
        """Calculate resource percentage used so far."""
        overs_remaining = self.overs_limit - overs_completed
        return 100.0 - self.resource_remaining(overs_remaining, wickets_fallen)

    def predict_final_score(
        self,
        current_score: float,
        overs_completed: float,
        overs_remaining: float,
        wickets_fallen: int,
    ) -> float:
        """
        Predict the final first innings score using DLS method.
        Extrapolates from current state assuming average resource usage.
        """
        resource_rem = self.resource_remaining(overs_remaining, wickets_fallen) / 100.0
        resource_used = 1.0 - resource_rem

        if resource_used <= 0:
            return self.G50  # No information yet

        # Projected final = current_score / resource_used_fraction
        predicted = current_score / resource_used

        # Sanity bounds
        predicted = max(predicted, current_score)  # Can't score less than current
        predicted = min(predicted, current_score + self.G50 * resource_rem)

        return round(predicted, 2)

    def par_score(
        self,
        target: float,
        overs_remaining: float,
        wickets_fallen: int,
    ) -> float:
        """
        Calculate par score for second innings under DLS.
        Given a target and current resources, what should the chasing team's target be?
        """
        resource_rem = self.resource_remaining(overs_remaining, wickets_fallen) / 100.0
        total_resource = self.resource_remaining(self.overs_limit, 0) / 100.0

        if total_resource == 0:
            return target

        resource_available = 1.0 - resource_rem  # resources used by chasing team
        par = target * (resource_available / total_resource)
        return round(par, 2)

    def get_resource_table(self) -> pd.DataFrame:
        """Generate standard DLS resource table (overs remaining x wickets lost)."""
        overs_range = np.arange(0, self.overs_limit + 1)
        wickets_range = range(10)

        table = pd.DataFrame(index=overs_range, columns=[f"w={w}" for w in wickets_range])

        for u in overs_range:
            for w in wickets_range:
                table.loc[u, f"w={w}"] = round(
                    self.resource_remaining(u, w), 1
                )

        table.index.name = "overs_remaining"
        return table

    def save(self, filepath: str = None):
        """Save fitted model to disk."""
        if filepath is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = MODELS_DIR / f"dls_model_{self.overs_limit}.pkl"

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "Z0": self.Z0,
                    "b": self.b,
                    "G50": self.G50,
                    "overs_limit": self.overs_limit,
                    "fitted": self.fitted,
                },
                f,
            )
        logger.info(f"DLS model saved to {filepath}")

    def load(self, filepath: str = None):
        """Load fitted model from disk."""
        if filepath is None:
            filepath = MODELS_DIR / f"dls_model_{self.overs_limit}.pkl"

        with open(filepath, "rb") as f:
            params = pickle.load(f)

        self.Z0 = params["Z0"]
        self.b = params["b"]
        self.G50 = params["G50"]
        self.overs_limit = params["overs_limit"]
        self.fitted = params["fitted"]
        logger.info(f"DLS model loaded from {filepath}")
