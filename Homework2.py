"""
-----------------------------------------------------------
Homework Set 2 | Computational Fluid Dynammics | EME 451
-----------------------------------------------------------
CHAMBON Yanis
MERCERON Julien
"""

from math import *
import numpy as np
import matplotlib.pyplot as plt


class LinearAdvection1D():
    """
    Solving the 1D Linear Advection and plot the solution for question 2.1 and 2.2
    """
    
    def __init__(self, N: int, nu: float, q: float, mode: int):
        self.N = N
        self.nu = nu
        self.q = q
        self.mode = mode
        self.Tobs = 3/2
        self.T = np.linspace(0, 3/2, N)
        self.X = np.linspace(0, 1, N) 
                
    
    def loop(self, u0: np.ndarray, u1: np.ndarray, N: int):
        """
        Calculates the different iterations of uj

        Args:
            u0 (np.ndarray): uj at t=0
            u1 (np.ndarray): uj at t=1
            N (int): number of iterations that we want 
        """
        for i in range(N):            
            for j in range(self.N):
                if (j - 1) < 0:
                    u1[j] = u0[j] - self.nu/2 * u0[j+1] + self.q/2 * (u0[j+1] - 2 * u0[j])
                if (j+1) >= N:
                    u1[j] = u0[j] - self.nu/2 * (-u0[j-1]) + self.q/2 * (-2 * u0[j] + u0[j-1])
                else:
                    u1[j] = u0[j] - self.nu/2 * (u0[j+1] - u0[j-1]) + self.q/2 * (u0[j+1] - 2 * u0[j] + u0[j-1])
            u0 = u1 
            

    def qScheme(self) -> np.ndarray:
        """
        Solving the 1D Linear Advection with the q-Scheme

        Returns:
            np.ndarray: solution of the 1D Linear Advection
        """
        xmax = len(self.X)     
        
        if self.mode == 1:              # Stands for Question 2.1
             
            u0 = np.sin(2*np.pi*self.X)
            u1 = np.zeros(u0.shape)

            if int(self.N/32) == 1:     # Verification for the time step
                self.loop(u0, u1, self.N)
                        
            else:
                self.loop(u0, u1, 32)
                u1[round(xmax/2):xmax] = u1[:round(xmax/2)]
                            
        elif self.mode == 2:            # Stands for question 2.2
            
            u0 = np.zeros(self.N)
            u1 = np.zeros(u0.shape)
            
            for i in range(self.N):     # Initialization of u0
                if int(self.N/4) <= i <= int((3*self.N)/4):
                    u0[i] = 1
                else:
                    u0[i] = 0
                    
            modulo = int(self.N/32)
            
            if modulo == 1:             # Verification for the time step
                self.loop(u0, u1, self.N)
                        
            else:
                self.loop(u0, u1, 32)
                u1[round(xmax/modulo):xmax] = u1[:round(xmax/modulo)]
    
        return u1
        
    
    def PlotLinearAdvection(self):
        """
        Plotting the function
        """
        U = self.qScheme()
        
        fig, ax = plt.subplots()
              
        if self.mode == 1:          # Stands for question 2.1
        
            ax.plot(self.T, U, color="#d600ff", label=f"nu={self.nu}, q={self.q}")
            ax.set_xlabel(r"$\Delta t$")
            ax.set_ylabel("$u_j$")
            ax.set_title(f"Propagation of a smooth function with $N={self.N}$")
            ax.legend()
            ax.grid()
            plt.show()
        
        elif self.mode == 2:        # Stands for question 2.2
            
            ax.plot(self.T, U, color="#b590cf", label=f"nu={self.nu}, q={self.q}")
            ax.set_xlabel(r"$\Delta t$")
            ax.set_ylabel("$u_j$")
            ax.set_title(f"Propagation of a discontinuous function with a linear scheme with $N={self.N}$")
            ax.legend()
            ax.grid()
            plt.show()


class HeatEquation1D():
    """
    Solving the 1D Heat Equation
    """
    
    def __init__(self, N: int, mu: int):
        self.Tobs = 4
        self.N = N
        self.mu = mu
        self.X = np.linspace(-1, 1, self.N)
        self.T = np.linspace(0, 4, self.N+2)
        

    def ElecticBlanketProblem(self) -> np.ndarray:
        
        u0, u1 = np.zeros(self.N), np.zeros(self.N)
        U = np.zeros((self.N, self.N))      # Matrix in which will be all the iterations of the function (uj)

        for i in range(self.N):
            if -1 <= self.X[i] <= 0:
                u0[i] = 1. + self.X[i]
            elif 0 <= self.X[i] <= 1:
                u0[i] = 1. - self.X[i]
                
        U[0:] = u0
                                
        for i in range(self.Tobs):
                for j in range(self.N):
                    if (j - 1) < 0:
                        u1[j] = u0[j] + self.mu * (u0[j+1] - 2 * u0[j])
                    if (j + 1) >= self.N:
                        u1[j] = u0[j] + self.mu * (-2 * u0[j] + u0[j-1])
                    else:
                        u1[j] = u0[j] + self.mu * (u0[j+1] - 2 * u0[j] + u0[j-1])
                
                U[i:] = u1                        
                u0 = u1
                
        u1 = np.insert(u1, 0, 0)
        u1 = np.insert(u1, self.N+1, 0)
        
        return u1
    
    
    def PlotHeatEquation(self):
        """
        Plotting the function
        """
        
        U = self.ElecticBlanketProblem()
        
        fig, ax = plt.subplots()
        
        ax.plot(self.T, U, label=f"mu={self.mu}")
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel("$u_j$")
        ax.set_title(f"Resolution of the 1D Heat Equation")
        ax.legend()
        ax.grid()
        
        
        
def ExactSolutionHE():
    return False


def SubplotDesign(N: int, nu: float, mode: int):

    for N_ in N:

        for nu_ in nu:
            
            q = [abs(nu_), nu_**2, 1]
            
            plt.suptitle(f"Propagation of a smooth function with $N={N_}$")
            
            for i in range(len(q)):
                
                plt.subplot(3, 1, i+1)
                
                LA = LinearAdvection1D(N_, nu_, q[i], mode)
                U = LA.qScheme()
                
                if mode == 1:
                    color = "#ff8243"
                elif mode == 2:
                    color = "#9370db"
                
                plt.plot(LA.T, U, color=color, label=rf"For $\nu={nu_}$, $q={q[i]}$")
                plt.xlabel(r"$\Delta t$")
                plt.ylabel("$u_j$")
                plt.legend(loc="upper right")
                plt.grid()
                
            plt.show()

        
        
if __name__ == "__main__":
    
    # Declaring variables    
    N = [32, 64]; M = [4, 8, 16, 32]; nu = [1, 0.75, 0.5, 0.25]; mu = [i/10 for i in range(1, 7)]
        
    # -------------------- QUESTION 2.1 -------------------- # 
    
    # --> To plot every curves on by one
    # for N_ in N:
    #     for nu_ in nu:
    #         q = [abs(nu_), nu_**2, 1]
    #         for q_ in q: 
    #             LA = LinearAdvection1D(N_, nu_, q_, 1)
    #             LA.PlotLinearAdvection()
    
    # --> To plot the curves three by three
    SubplotDesign(N, nu, 1)    
    
    # -------------------- QUESTION 2.2 -------------------- # 
    
    # --> To plot every curves on by one
    # for N_ in N:
    #     for nu_ in nu:
    #         q = [abs(nu_), nu_**2, 1]
    #         for q_ in q: 
    #             LA = LinearAdvection1D(N_, nu_, q_, 2)
    #             LA.PlotLinearAdvection()
    
    # --> To plot the curves three by three
    SubplotDesign(N, nu, 2)
    
    # -------------------- QUESTION 3 -------------------- # 
    
    for M_ in M:
        for mu_ in mu:
            U = HeatEquation1D(M_, mu_)
            U0 = U.ElecticBlanketProblem()
            plt.plot(U.T, U0, label=f"mu={mu_}")

        plt.xlabel(r"$\Delta t$")
        plt.ylabel("$u_j$")
        plt.title(f"Resolution of the 1D Heat Equation with $N={M_}$")
        plt.legend()
        plt.grid()
        plt.show()
        
    for mu_ in mu:
        for M_ in M:
            U = HeatEquation1D(M_, mu_)
            U0 = U.ElecticBlanketProblem()
            plt.plot(U.T, U0, label=f"N={M_}")

        plt.xlabel(r"$\Delta t$")
        plt.ylabel("$u_j$")
        plt.title(rf"Resolution of the 1D Heat Equation with $\mu={mu_}$")
        plt.legend()
        plt.grid()
        plt.show()