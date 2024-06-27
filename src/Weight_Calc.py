import numpy as np
from scipy.optimize import minimize
from typing import Literal


class Weight_Calc:
    def __init__(
        self,
        smartScheme: Literal["EW", "RP", "MDR", "GMV", "MSR", "SpecReturn"],
        predictive_mean: np.ndarray,
        predictive_covMat: np.ndarray[np.ndarray],
        rf: float = 0.05,
        reuired_return: float = None,
        boundary: tuple[float, float] = (-1, 1),
        tol: float = 1e-10,
    ):
        self.smartScheme = smartScheme
        self.mu = predictive_mean
        self.covMat = predictive_covMat
        self.rf = rf
        self.required_return = reuired_return
        self.lbound = boundary[0]
        self.ubound = boundary[1]
        self.tol = tol

    def RP(self, w):
        w = np.array(w)
        covMat = self.covMat
        vol = np.sqrt(w.T @ covMat @ w)
        marginal_contribution = w * (covMat @ w / vol)
        r = vol / w.shape - marginal_contribution
        rp = sum(r**2)
        return rp

    def DR(self, w):
        w = np.array(w)
        covMat = self.covMat
        vol = np.sqrt(w.T @ covMat @ w)
        di = sum(w * np.sqrt(np.diag(covMat))) / vol
        return -di  # to minimize take negative

    def MV(self, w):
        w = np.array(w)
        covMat = self.covMat
        var = w.T @ covMat @ w
        return var

    def SR(self, w):
        w = np.array(w)
        rf = self.rf
        mu = self.mu
        covMat = self.covMat
        sharpe = (w.T @ mu - rf) / np.sqrt(w.T @ covMat @ w)
        return -sharpe  # to minimize take negative

    def weight_constraint(self, w):
        return 1 - sum(w)

    def return_constraint(self, w):
        return self.required_return - w.T @ self.mu

    def retrieve_beta(self):
        K = self.covMat.shape[0]
        x_0 = np.ones(K) / K  # initial weighting for SmartBeta Optimizer
        bounds = ((self.lbound, self.ubound),) * K

        if self.smartScheme == "EW":
            Beta = np.ones(K) / K
        else:
            if self.smartScheme == "RP":
                objective = self.RP
            elif self.smartScheme == "MDR":
                objective = self.DR
            elif self.smartScheme == "GMV" or self.smartScheme == "SpecReturn":
                objective = self.MV
            elif self.smartScheme == "MSR":
                objective = self.SR

            if self.smartScheme == "SpecReturn":
                const = [{"type": "eq", "fun": self.weight_constraint}, {"type": "eq", "fun": self.return_constraint}]
            else:
                const = {"type": "eq", "fun": self.weight_constraint}
            res = minimize(objective, x_0, method="SLSQP", bounds=bounds, tol=self.tol, constraints=const)
            Beta = res.x
        return Beta
