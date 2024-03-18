import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm

class KalmanFilter():
    def __init__(self, dt, x, P, Q, R, F, H):
        # Initialise our KF variables, state vector, and model parameters
        self.dt = dt
        self.x = x
        self.P = P
        self.Q = Q
        self.R = R
        self.F = F
        self.H = H
        
        self.e = 0
        self.d = 0
        
        # Fading memory parameter
        self.alpha = 1

        # Arrays to store results
        self.xs = []
        self.Ps = []
        self.predictions = []
        
    def predict(self):
        # Predict our state using our model of the system
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.transpose() + self.Q

    def update(self, z):
        # R estimation from 
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8273755
        #self.R = self.alpha * self.R + (1 - self.alpha) * (self.e**2 + self.H @ self.P @ self.H.transpose())
        #self.R = self.R[0][0]
        #print(self.R)
        
        # Innovation and innovation covariance
        self.d = z - self.H @ self.x
        self.S = self.H @ self.P @ self.H.transpose() + self.R
        
        # Kalman Gain
        self.K = self.alpha * self.P @ self.H.transpose() @ np.linalg.inv(self.S)
        
        # Posteriori state and covariance estimate
        self.x = self.x + self.K @ self.d
        self.P = self.P - self.K @ self.H @ self.P
        
        # Residual
        self.e = z - self.H @ self.x
        
        # Q estimation from 
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8273755
        #self.Q = self.alpha * self.Q + (1 - self.alpha) * (self.K * self.d**2 * self.K.transpose())
        
        # Compute the likelihood that the filter is performing optimally
        #self.L = (1 / (2 * np.pi * self.S)**0.5 @ np.exp(-0.5 * self.d.transpose() * np.linalg.inv(self.S) * self.d))[0][0]
        self.L = np.exp(multivariate_normal.logpdf(z, np.dot(self.H, self.x), self.S))

        return self.x

    def run(self):
        for i in range(self.data.shape[0]):
            self.predict()
            self.predictions.append(self.x)
            print(self.x)
            self.update(self.data[i])
            self.xs.append(self.x)
            print(self.x)
            self.Ps.append(self.P)

        self.predictions = np.asarray(self.predictions)
        self.xs = np.asarray(self.xs)
        self.Ps = np.asarray(self.Ps)

    def NEES(self, xs, est_xs, Ps):
        est_err = xs - est_xs
        err = []
        for x, p in zip(est_err, Ps):
            err.append(x.T @ np.linalg.inv(p) @ x)
        return err
    
    def residuals(self, xs, est_xs):
        return xs - est_xs

    @classmethod
    def constant_acceleration(cls, dt, q, R, x_0, P_0):
        # This is a constant acceleration filter
        x = x_0
        P = P_0
        Q = np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6], 
                    [dt**4 / 8, dt**3 / 3, dt**2 / 2], 
                    [dt**3 / 6, dt**2 / 2, dt]]) * q**2
        F = np.array([[1, dt, dt**2 / 2], 
                    [0, 1, dt], 
                    [0, 0, 1]])
        H = np.array([[1, 0, 0]])

        return cls(dt, x, P, Q, R, F, H)
    
    @classmethod
    def constant_velocity_3(cls, dt, q, R, x_0, P_0):
        # This is a constant velocity filter with 3 dimensions
        x = x_0
        P = P_0
        Q = np.array([[dt**3 / 3, dt**2 / 2, 0], 
                    [dt**2 / 2, dt, 0], 
                    [0, 0, 0]]) * q**2
        F = np.array([[1, dt, 0], 
                    [0, 1, 0], 
                    [0, 0, 0]])
        H = np.array([[1, 0, 0]])

        return cls(dt, x, P, Q, R, F, H)
    
    @classmethod
    def constant_velocity_2(cls, dt, q, R, x_0, P_0):
        # This is a constant velocity filter with 2 dimensions
        x = x_0
        P = P_0
        Q = np.array([[dt**3 / 3, dt**2 / 2], 
                    [dt**2 / 2, dt]]) * q**2
        F = np.array([[1, dt], 
                    [0, 1]])
        H = np.array([[1, 0]])

        
        kf = cls(dt, x, P, Q, R, F, H)

        kf.xs.append(x_0)
        kf.Ps.append(P_0)

        return kf