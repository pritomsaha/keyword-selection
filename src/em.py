import numpy as np
class EM(object):
    def __init__(self):
        """
        Initialize empty document list.
        """
        self.C_wd = None  # C(w,d)
        self.P_zw = None # P(z = 0 | w)
        self.P_wd = None  # P(w | theta_d)
        self.P_wb = None  # P(w | theta_b)
        self.P_d = None  # P(theta_d)
        self.P_b = None  # P(theta_b)
        self.epsilon = None
        self.max_iter = None 
        self.likelihoods = []
    
    def initialize(self):
        self.P_wd = np.random.random_sample(self.P_wb.shape)
        self.P_wd /= np.sum(self.P_wd)
        
        
    def expectation_step(self):
        """ The E-step updates P(z = 0 | w)
        """
        
        # ############################
        
        self.P_zw = self.P_d*self.P_wd
        self.P_zw /= self.P_zw + self.P_b*self.P_wb
        
        # ############################
            

    def maximization_step(self):
        """ The M-step updates P(w | theta_d)
        """
        
        # update P(w | z)
        
        # ############################
        
        self.P_wd = self.C_wd * self.P_zw
        self.P_wd /= np.sum(self.P_wd)
        
        
        # ############################

    def calculate_likelihood(self):
        llh = self.C_wd*np.log(self.P_d*self.P_wd + self.P_b*self.P_wb)
        llh = np.sum(llh)
        self.likelihoods.append(llh)
        
        
    def train_once(self):
        self.initialize()
        
        # Run the EM algorithm
        current_likelihood = -np.inf

        for iteration in range(self.max_iter):
#             print("Iteration #" + str(iteration + 1) + "...")
            
            # ############################
            self.expectation_step()
            self.maximization_step()
            self.calculate_likelihood()
            if abs(self.likelihoods[-1] - current_likelihood) < self.epsilon:
                break
            current_likelihood = self.likelihoods[-1]
            # ############################
            
        return current_likelihood
        
    def run_model(self, C_wd, P_wb, sigma = 0.5, n_start = 5, max_iter = 10, epsilon = 0.001):
        self.P_wb = P_wb
        self.C_wd = C_wd
        self.P_b = sigma
        self.P_d = 1 - sigma
        self.max_iter = max_iter
        self.epsilon = epsilon
        
        max_llh = -np.inf
        best_P_wd = None
        for _ in range(n_start):
            llh = self.train_once()
            if llh > max_llh:
                max_llh = llh
                best_P_wd = self.P_wd.copy()
        return max_llh, best_P_wd
        