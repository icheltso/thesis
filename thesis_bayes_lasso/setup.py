# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:46:07 2023

@author: ichel
"""

"Prepares data and variables for runs"


class Setup:
    def __init__(self, sigma, loss_type, A1, A2, n, lam, y, y2, initx, initv, initz, M, gamma, R):
        "A and As are functions"
        self.sigma = sigma
        self.A1 = A1
        self.A2 = A2
        self.n = n
        self.lam = lam
        self.y = y
        self.y2 = y2
        self.initx = initx
        self.initv = initv
        self.initz = initz
        self.M = M
        self.gamma = gamma
        self.R = R
        self.loss_type = loss_type
        
    def get_data(self):
        datadict = {
            "sigma":    self.sigma,
            "n":        self.n,
            "lam":      self.lam,
            "y":        self.y,            
            "y2":       self.y2,
            "initx":    self.initx,
            "initv":    self.initv,
            "initz":    self.initz,
            "M":        self.M,
            "gamma":    self.gamma,
            "R":        self.R,
            "loss_type":self.loss_type
            } 
        return datadict
    
    def A(self, x):
        return self.A1(x)
    
    def As(self, x):
        return self.A2(x)

