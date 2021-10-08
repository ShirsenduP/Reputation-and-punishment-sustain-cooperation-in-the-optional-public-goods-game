'''
Reputation and Punishment sustain cooperation in the Optional Public Goods Game
Shirsendu Podder, Simone Righi, Francesca Pancotto

 # @ Author: Shirsendu Podder
 # @ Created: 2021-05-10 
 # @ Description: Example script to run a single simulation from the paper
'''

"""
| Configuration(N: int, n: int, r: float, sigma: float, t: int, composition: dict, norm: str, gamma: float, beta: float, u: float, strategy_group: str)
|
|  Provides the functionality and error checking for all parameterizations of simulations.
|
|  Args:
|      N (int): The number of players in the population.
|      n (int): The number of players per Public Goods Game (PGG).
|      t (int): The length of the simulation.
|      r (float): The growth factor to the group contribution in the PGG.
|      sigma (float): The growth factor for a loner's utility.
|      composition (dict): A dictionary of strategy and proportion key-value pairs. See _Strategy documentation.
|      norm (str): The specific social norm within the population. See _Norm documentation.
|      gamma (float): The cost required to punish someone.
|      beta (float): The penalty one pays if one is punished.
|      u (float): The probability of update to a random strategy in the strategy group.
"""


from opgar import Configuration, Population, _Strategy

if __name__ == "__main__":

    model_name = "purenopunish"
    social_norm = None

    strategies = _Strategy.strategy_groups[model_name]
    config = Configuration(
        N=1000, 
        n=5, 
        t=200000, 
        r=3, 
        sigma=1,
        composition={}.fromkeys(strategies, 1/len(strategies)), 
        norm=social_norm,
        gamma=1,
        beta=2,
        u=0.01,
        m=0.95,
        omega=10/11, 
        epsilon=0.1,
        strategy_group=model_name)
        
    print(config)
    
    P = Population(config=config)
    P.simulate(use_group_selection=True, t_step=25000, job_id=0000)
