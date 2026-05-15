import numpy as np
from .base_kalman import BaseKalman

class SDEKalman(BaseKalman):
    def __init__(self, dt=1/120.0): 
        state_dim = 8  # [cx, cy, w, h, vx, vy, vw, vh]
        observation_dim = 4 # [cx, cy, w, h]
        
        F = np.eye(state_dim, state_dim)
        for i in range(state_dim // 2):
            F[i, i + state_dim // 2] = dt

        H = np.eye(observation_dim, state_dim)
        super().__init__(state_dim=state_dim, observation_dim=observation_dim, F=F, H=H)
        
        self.dt = dt
        self.epsilon = 1e-5 
        
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = (1. / 160) * (1.0 / dt) 

    def initialize(self, observation):
        mean_pos = observation
        mean_vel = np.zeros_like(observation)
        self.kf.x = np.r_[mean_pos, mean_vel]

        w, h = observation[2], observation[3]
        std = [
            2 * self._std_weight_position * w,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * w,  
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * w,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * w, 
            10 * self._std_weight_velocity * h, 
        ]       
        self.kf.P = np.diag(np.square(std))

    def predict(self, is_activated=True):
        if not is_activated:
            self.kf.x[7] = 0.0

        w, h = self.kf.x[2], self.kf.x[3]
        std_pos = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,  
            self._std_weight_position * h
        ]
        std_vel = [
            self._std_weight_velocity * w,
            self._std_weight_velocity * h,
            self._std_weight_velocity * w,  
            self._std_weight_velocity * h
        ]
        
        Q_discrete = np.diag(np.square(np.r_[std_pos, std_vel]))
        self.kf.Q = Q_discrete 

        self.kf.predict() 
        self.kf.P += np.eye(8) * self.epsilon 

    def update(self, z):
        if z is None:
            return
        
        w, h = self.kf.x[2], self.kf.x[3]
        std = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,  
            self._std_weight_position * h
        ]
        self.kf.R = np.diag(np.square(std))
        self.kf.update(z)