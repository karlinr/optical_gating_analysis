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
            self.update(self.data[i])
            self.xs.append(self.x)
            self.Ps.append(self.P)

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


class InteractingMultipleModelFilter():
    # IMM adapted from filterpy implementation
    
    
    def __init__(self, models, mu, M):
        """
        _summary_

        Args:
            models (list): Instances of Kalman filter class in a list or tuple of length n
            mu (np.array): Initial filter probabilities of length n
            M (np.array): Filter transition matrix of length n x n
        """        
        # Add our models
        # Instances of the KalmanFilter class
        self.models = models
        
        # Initialise state probabilities
        self.mu = mu
        self.M = M
        
        self.omega = np.zeros((len(self.models), len(self.models)))

        self.compute_mixing_probabilities()
        self.compute_state_estimate()
        
    def predict(self):       
        xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.models, self.omega.T)):
            x = np.zeros(self.x.shape)
            for kf, wj in zip(self.models, w):
                x += kf.x * wj
            xs.append(x)

            P = np.zeros(self.P.shape)
            for kf, wj in zip(self.models, w):
                y = kf.x - x
                P += wj * (np.outer(y, y) + kf.P)
            Ps.append(P)

        # compute each filter's prior using the mixed initial conditions
        for i, f in enumerate(self.models):
            # propagate using the mixed state estimate and covariance
            f.x = xs[i].copy()
            f.P = Ps[i].copy()
            f.predict()

        # compute mixed IMM state and covariance and save posterior estimate
        self.compute_state_estimate()
                    
    def update(self, z):
        self.d = z - self.x[0]
        
        for i, model in enumerate(self.models):
            model.update(z)
            self.mu[i] = model.L * self.cbar[i]
        self.mu /= np.sum(self.mu)
        
        self.compute_mixing_probabilities()
        self.compute_state_estimate()
            
    def compute_mixing_probabilities(self):
        self.cbar = self.mu @ self.M
        for i in range(len(self.models)):
            for j in range(len(self.models)):
                self.omega[i, j] = (self.M[i, j] * self.mu[i]) / self.cbar[j]
                
    def compute_state_estimate(self):
        self.x = np.zeros(self.models[0].x.shape)
        self.P = np.zeros(self.models[0].P.shape)
        for i, model in enumerate(self.models):
            self.x += model.x * self.mu[i]

        for i, model in enumerate(self.models):
            y = model.x - self.x
            self.P += self.mu[i] * np.outer(y, y) + model.P

    def run(self):
        self.xs = []
        self.Ps = []
        self.mus = []
        for i in range(self.data.shape[0]):
            self.predict()
            self.update(self.data[i])
            self.xs.append(self.x)
            self.Ps.append(self.P)
            self.mus.append(self.mu)

        self.xs = np.asarray(self.xs)
        self.Ps = np.asarray(self.Ps)
        self.mus = np.asarray(self.mus)

    def NEES(self, xs, est_xs, Ps):
        est_err = xs - est_xs
        err = []
        for x, p in zip(est_err, Ps):
            err.append(x.T @ np.linalg.inv(p) @ x)
        return err
    
    def residuals(self, xs, est_xs):
        return xs - est_xs

class VariableStructureMultipleModel:
    def __init__(self) -> None:
        pass

class UnscentedKalmanFilter:
    def __init__(self, x, P, dt, process_var, measurement_var):
        # State vector
        self.x = x
        self.P = P

        # other params
        self.dt = dt

        # Process and measurement variance
        self.Q = np.array([[self.dt**4 / 4, self.dt**3 / 2],
                           [self.dt**3 / 2, self.dt**2]])* process_var

        self.Q = np.array([[self.dt**4 / 4, 0],
                           [0, self.dt**2]])* process_var

        self.Q = np.array([[self.dt * process_var, 0, 0],
                           [0, self.dt**2, self.dt**2 / 2],
                           [0, self.dt**2 / 2, self.dt**3 / 3]])

        pvar1 = 0.0001**2
        pvar2 = 0.0015**2
        self.Q = np.array([[self.dt**2 * pvar1, 0, 0],
                           [0, self.dt**2 * pvar2, (self.dt**2 / 2) * pvar2],
                           [0, (self.dt**2 / 3) * pvar2, (self.dt**3 / 3) * pvar2]])

        #self.Q = np.array([[self.dt**2 * pvar1, 0, 0],
        #                   [0, self.dt**2 * pvar2, 0],
        #                   [0, 0, self.dt**2 * pvar3]])

        self.R = np.array([measurement_var])

        # UKF params
        self.L = len(self.x.transpose())
        self.alpha = 1e-2
        self.kappa = 3 - self.L
        self.beta = 2
        self.lambda_ = (self.alpha**2) * (self.L + self.kappa) - self.L

        # Get our sigma point weights
        self.weights_mean = np.empty(2 * self.L + 1)
        self.weights_cova = np.empty(2 * self.L + 1)

        self.weights_mean[0] = self.lambda_ / (self.L + self.lambda_)
        self.weights_cova[0] = (self.lambda_ / (self.L + self.lambda_)) + (1 - self.alpha**2 + self.beta)

        for i in range(0, 2 * self.L):
            self.weights_mean[i + 1] = 1 / (2 * (self.L + self.lambda_))
            self.weights_cova[i + 1] = 1 / (2 * (self.L + self.lambda_))

    def F(self, x):
        return [x[0], x[1], x[1] * self.dt + x[2]]

    def H(self, x):
        return x[0] * np.sin(x[2])

    def get_sigma_points(self, mean, cova):
        # Get sigma points for a given mean and covariance

        # Set up array and set central sigma point
        sigma_points = np.empty((2 * self.L + 1, self.L))
        sigma_points[0] = mean

        # Get our remaining sigma points
        mat = sqrtm((self.L + self.lambda_) * cova)
        #mat = np.linalg.cholesky((self.L + self.lambda_) * cova)
        for i in range(0, self.L):
            sigma_points[i + 1] = mean + mat[i]
        for i in range(self.L, 2 * self.L):
            sigma_points[i + 1] = mean - mat[i - self.L]

        return sigma_points

    def get_sigma_points_mean_cova(self, sigma_points):
        # Get mean and coviariance for a given set of sigma points

        # Get the mean
        mean = np.dot(self.weights_mean, sigma_points)

        # Get the covariance
        n, m = sigma_points.shape
        covariance = np.zeros((m, m))
        for i in range(n):
            covariance += self.weights_cova[i] * np.outer(sigma_points[i] - mean, sigma_points[i] - mean)

        return mean, covariance

    def predict(self):
        # Get our sigma points
        self.sigma_points = self.get_sigma_points(self.x, self.P)

        #self.plot_sigma_points(self.sigma_points, *self.get_sigma_points_mean_cova(self.sigma_points), False)


        # Propagate sigma points through process function
        self.sigma_points_pred = np.empty_like(self.sigma_points)
        for i in range(len(self.sigma_points_pred)):
            self.sigma_points_pred[i] = self.F(self.sigma_points[i])

        # Unscented transform to get mean and covariance
        self.x_pred, self.P_pred = self.get_sigma_points_mean_cova(self.sigma_points_pred)
        self.P_pred += self.Q

    def update(self, z):
        # Get our predictions in measurement space
        sigma_points_meas = np.empty((len(self.sigma_points_pred), 1))
        for i in range(len(self.sigma_points_pred)):
            sigma_points_meas[i] = self.H(self.sigma_points_pred[i])
        sigma_points_mean_meas, sigma_points_cova_meas = self.get_sigma_points_mean_cova(sigma_points_meas)

        # Get residuals of measurement vs prediction
        y = z - sigma_points_mean_meas

        # Get uncertainty on predicted measurement
        P_z, = sigma_points_cova_meas
        P_z += self.R
        P_z = np.array([P_z])

        Pxz = np.zeros((3, 1))
        for i in range(self.L):
            Pxz += self.weights_cova[i] * np.outer(self.sigma_points_pred[i] - self.x_pred, sigma_points_meas[i] - sigma_points_mean_meas)
        K = np.dot(Pxz, np.linalg.inv(P_z))
        self.x = self.x_pred + np.dot(K, y)
        self.P = self.P_pred - np.dot(K, P_z).dot(K.T)

        return self.H(self.x)