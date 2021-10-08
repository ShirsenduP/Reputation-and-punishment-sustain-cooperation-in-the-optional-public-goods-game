import copy
import json
import logging
from collections import Counter
from datetime import datetime
from itertools import product
from random import seed
from time import time

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from .utils import deprecated, Utils, DummyTqdmFile

class Configuration:
    """Provides the functionality and error checking for all parameterizations of simulations.

        Args:
            N (int): The number of players in the population.
            n (int): The number of players per Public Goods Game (PGG).
            t (int): The length of the simulation.
            r (float): The growth factor to the group contribution in the PGG.
            sigma (float): The growth factor for a loner's utility.
            composition (dict): A dictionary of strategy and proportion key-value pairs. See _Strategy documentation.
            norm (str): The specific social norm within the population. See _Norm documentation.
            gamma (float): The cost required to punish someone.
            beta (float): The penalty one pays if one is punished.
            u (float): The probability of update to a random strategy in the strategy group using the Rand and Nowak evolutionary mechanism.
            m (float): The degree of evolutionary mixing in group selection
            epsilon (float): The rate of mutation under group selection, this is unused with Rand and Nowak evolutionary mechanism.
            strategy_group (str): The name of the model being simulated, see `_Strategy.strategy_groups`.
            omega (float): Probability of multiple rounds of the OPGG in a single time-step. 

        Returns:
            opgar.Configuration object which is input to opgar.Population object.

        Examples:
            >>> from opgar import Configuration
            >>> config = Configuration.test_case_1()
            >>> config = Configuration(
                    N=1000, 
                    n=5, 
                    t=200000, 
                    r=3, 
                    sigma=1,
                    composition={}.fromkeys(strategies, 1/len(strategies)), 
                    norm="Defector",
                    gamma=1,
                    beta=2,
                    u=0.01,
                    m=0.95,
                    omega=10/11, 
                    epsilon=0.1,
                    strategy_group="all"
                )
    """

    __slots__ = [
        "social_norm",
        "composition",
        "t",
        "r",
        "sigma",
        "n",
        "N",
        "gamma",
        "beta",
        "u",
        "m",
        "epsilon",
        "omega",
        "_meta_data",
    ]

    def __init__(
        self,
        N: int,
        n: int,
        r: float,
        sigma: float,
        t: int,
        composition: dict,
        norm: str,
        gamma: float,
        beta: float,
        u: float,
        m: float,
        epsilon: float,
        strategy_group: str, 
        omega: float = 0,
    ):
        self._meta_data = {}

        # ----------------------------------------------------------------------
        # SOCIAL NORM (Anti-defector, Anti-loner, Anti-Neither, None)
        # ----------------------------------------------------------------------

        if norm not in _Norm.social_norm_names:
            raise ValueError(f"Your choice of norm ('{norm}') is invalid.")
        self.social_norm = norm 
        self._meta_data["strategy group"] = strategy_group
        
        # An absence of a social norm can only occur when there are no impure strategies in the population.
        if norm == None:
            for strategy in composition.keys():
                if strategy not in _Strategy.strategy_groups["pure"]:
                    raise ValueError(f"Strategy {strategy} cannot exist without a social norm.")

        # ----------------------------------------------------------------------
        # STRATEGY COMPOSITION
        # ----------------------------------------------------------------------
        if type(composition) == dict:
            if round(sum(composition.values()), 6) != 1:
                raise ValueError(
                    f"Your choice of strategy proportions must sum to 1 (not '{sum(composition.values())}')."
                )
            for strategy in composition.keys():
                if strategy not in _Strategy.all_strategies:
                    raise ValueError(
                        f"Your choice of strategy ('{strategy}') is invalid."
                    )
                if composition[strategy] < 0 or composition[strategy] > 1:
                    raise ValueError(
                        f"Your choice of strategy proportion ('{strategy}': {composition[strategy]}) is invalid."
                    )
        elif type(composition) == str:
            if composition not in _Strategy.strategy_groups:
                raise ValueError(
                    f"Your choice of strategy set ('{composition}') is not valid. See _Strategy.strategy_groups"
                )
        self.composition = pd.Series(composition)
        self._meta_data["norm"] = norm


        # ----------------------------------------------------------------------
        # LENGTH OF SIMULATION
        # ----------------------------------------------------------------------
        if t < 1:
            raise ValueError(f"Your choice of t ('{t}') is invalid.")
        self.t = t

        # ----------------------------------------------------------------------
        # GROUP SYNERGY FACTOR
        # ----------------------------------------------------------------------
        self.r = r

        # ----------------------------------------------------------------------
        # LONER PAYOFFS
        # ----------------------------------------------------------------------
        if strategy_group == "noloner" or strategy_group == "prosocialnoloner":
            self.sigma = None
        else:
            if sigma < 0 or sigma is None:
                raise ValueError(f"Your choice of sigma ('{sigma}') is invalid.")
            self.sigma = sigma

        # ----------------------------------------------------------------------
        # POPULATION SIZE
        # ----------------------------------------------------------------------

        if N < 0:
            raise ValueError(f"Your choice of population size ('{N}') is invalid.")
        self.N = N

        # ----------------------------------------------------------------------
        # OPTIONAL PGG SIZE
        # ----------------------------------------------------------------------

        if n < 0 or n > N:
            raise ValueError(f"Your choice of n ('{n}') is invalid.")
        if not N % n == 0:
            raise ValueError(
                f"PGG size ('{n}') must be a divisor of the population size ('{N}')."
            )
        self.n = n

        # ----------------------------------------------------------------------
        # PUNISHMENT COST & PENALTY
        # ----------------------------------------------------------------------
        if "P" in "".join(composition.keys()):
            if gamma > beta:
                raise ValueError(
                    f"The cost to punish ('{gamma}') should not be greater than the penalty "
                    f"incurred by being punished ('{beta}') ."
                )
            if gamma < 0 or beta < 0:
                raise ValueError(
                    f"The cost to punish ('{gamma}') or the  penalty incurred by being punished "
                    f"('{beta}') cannot be negative."
                )
        self.gamma = gamma
        self.beta = beta

        # ----------------------------------------------------------------------
        # EVOLUTIONARY UPDATE PROBABILITY
        # ----------------------------------------------------------------------
        self.u = u

        # ----------------------------------------------------------------------
        # PROBABILITY OF FURTHER GAMES IN SAME PERIOD
        # ----------------------------------------------------------------------
        if omega < 0 or omega >= 1:
            raise ValueError("Probability of further interactions omega ('{omega}') must be within [0,1).")
        self.omega = omega

        # ----------------------------------------------------------------------
        # GROUP-WIDE vs POPULATION-WIDE EVOLUTION 
        # ----------------------------------------------------------------------
        # If m=0, people always imitate people from outside of the group
        # if m=1, people always imitate people from the same group
        if m < 0 or m > 1:
            raise ValueError("Probability of mutation vs evolution m ('{m}') must be in [0,1].")
        self.m = m
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Probability of mutation epsilon ('{epsilon}') must be in [0,1].")
        self.epsilon = epsilon


    def to_dict(self):
        """
        Return the parameters of a configuration object in dictionary.
        """
        config_dict = {slot: getattr(self, slot) for slot in self.__slots__}
        # Custom strategy group
        try:
            config_dict["composition"] = (
                self._meta_data["strategy group"],
                1 / len(_Strategy.strategy_groups[self._meta_data["strategy group"]]),
            )
        except:
            config_dict["composition"] = (self._meta_data["strategy group"])
        return config_dict


class _Strategy:
    """
    The _Strategy class provides all the relevant behavioural strategy functions for agents deciding their actions. Agents call the _Strategy._choose_action and _Strategy._choose_punishment methods to assign actions and punishments according to the function look-up tables this class provides.

    Class variables:
        - `_Strategy.behavioural_strategy_names` is a list of the names of the strategies ignoring punishment variants (ie. the strategy roots).
        - `_Strategy.behavioural_strategy` is a dictionary where the keys are the strategy roots, and the values are additional dictionaries containing the lambda function prescribing the action, and a description of the strategy
        - `_Strategy.punishment_strategies_names` and `_Strategy.punishment_strategies` are the same as above except describing the punishment variants and ignoring the strategy roots. 
        - _Strategy.all_strategies is a comprehensive list of all strategies available under this model
        - _Strategy.strategy_groups is a dictionary containing lists of strategies grouped by particular models. 

            - all - Model with Reputation and Punishment
            - pure - Baseline model with only punishment
            - purenopunish - Baseline model with no reputation or punishment
            - nopunish - Model with only reputation
            - impure - All conditional strategies
            - prosocial - AllC, AllD, AllL and AllC that punishes defectors.
            - prosocialnoloner - AllC, AllD, and AllC that punishes defectors.
            - noloner - All strategies that do not involve being a loner.
            - noAllC - All strategies without any unconditional cooperators.

        - _Strategy.strategy_name_mapping is a dictionary mapping the roman numeral version of strategy names (from I - XI) to the form used in the paper.    
    """

    behavioural_strategy_names = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    behavioural_strategy = {
        "I": {
            "function": lambda avg: 1,
            "description": "Unconditional Cooperation",
        },
        "II": {
            "function": lambda avg: 0,
            "description": "Unconditional Defection",
        },
        "III": {
            "function": lambda avg: None,
            "description": "Never participate",
        },
        "IV": {
            "function": lambda avg: 1 if avg > -1 else 0,
            "description": "Cooperate if anyone in the group is not bad otherwise defect",
        },
        "V": {
            "function": lambda avg: 1 if avg > 0 else 0,
            "description": "Cooperate if group has mostly good people otherwise defect",
        },
        "VI": {
            "function": lambda avg: 1 if avg > -1 else None,
            "description": "Cooperate if anyone in the group is not bad otherwise don't participate",
        },
        "VII": {
            "function": lambda avg: 1 if avg > 0 else None,
            "description": "Cooperate if group has mostly good people otherwise don't participate",
        },
        "VIII": {
            "function": lambda avg: 0 if avg > -1 else None,
            "description": "Defect if group has any not-bad people otherwise don't participate",
        },
        "IX": {
            "function": lambda avg: 0 if avg > 0 else None,
            "description": "Defect if group is mostly good otherwise don't participate",
        },
        "X": {
            "function": lambda avg: None if avg > -1 else 0,
            "description": "Don't participate if group has anyone not-bad otherwise defect",
        },
        "XI": {
            "function": lambda avg: None if avg > 0 else 0,
            "description": "Don't participate if group is mostly good otherwise defect",
        },
    }

    punishment_strategies_names = ["".join(d) for d in product("NP", repeat=3)]
    punishment_strategies = {
        "NNN": {"function": lambda action: False, "description": "Do not punish"},
        "NNP": {
            "function": lambda action: True if action in [None] else False,
            "description": "Punish loners",
        },
        "NPN": {
            "function": lambda action: True if action in [0] else False,
            "description": "Punish defectors",
        },
        "NPP": {
            "function": lambda action: True if action in [0, None] else False,
            "description": "Punish defectors and loners",
        },
        "PNN": {
            "function": lambda action: True if action in [1] else False,
            "description": "Punish cooperators",
        },
        "PNP": {
            "function": lambda action: True if action in [1, None] else False,
            "description": "Punish cooperators and loners",
        },
        "PPN": {
            "function": lambda action: True if action in [1, 0] else False,
            "description": "Punish cooperators and defectors",
        },
        "PPP": {"function": lambda action: True, "description": "Punish everyone"},
    }

    all_strategies = [
        "_".join([i, j])
        for i, j in product(behavioural_strategy_names, punishment_strategies_names)
    ]
    strategy_groups = {
        "all": all_strategies,
        "pure": all_strategies[:24],
        "purenopunish": all_strategies[:24:8],
        "nopunish": [strategy for strategy in all_strategies if "_NNN" in strategy],
        "impure": all_strategies[24:],
        "prosocial": ["I_NNN", "I_NPN", "II_NNN", "III_NNN"],
        "prosocialnoloner": ["I_NNN", "I_NPN", "II_NNN"],
        "noloner": [strategy for strategy in all_strategies if strategy.split("_")[0] in ["I", "II", "IV", "V"]],
        "noAllC": all_strategies[8:],
    }

    strategy_name_mapping = {
        "I": "I",
        "II": "II",
        "III": "III",
        "IV": "I^{-1,D}",
        "V": "I^{0,D}",
        "VI": "I^{-1,L}",
        "VII": "I^{0,L}",
        "VIII": "II^{-1,L}",
        "IX": "II^{0,L}",
        "X": "III^{-1,D}",
        "XI": "III^{0,D}",
    }

    @staticmethod
    def _choose_action(ID, avg_rep):
        """
        Return a function implementing the agent's behavioural strategy depending on the strategy from _Strategy.names.
        Each agent stores a reference to their function in _Strategy.choices to call as needed.

        Args:
            ID (str): A string representation the 11 behavioural strategies 'I' - 'XI' within the model
            avg_rep (float): The average reputation of the PGG group, used as an argument to the strategy behavioural
            function.
        Returns:
            _Strategy.choices[ID] (func): A function requiring a single argument 'average reputation'. See
            agent._choose_action for more information.
        """
        return _Strategy.behavioural_strategy[ID]["function"](avg_rep)

    @staticmethod
    def _choose_punishment(ID, action):
        """
        Return a function implementing the agent's punishment strategy depending on a receiving agent's last action out
        of 1, 0, or None representing a contribution, defection or withdrawal from the public goods game.

        Args:
            ID (str): A string representation of the punishment strategy 'NNN', 'NNP', ..., 'PPP'. These denote
            punishment in the order of Cooperator, Defector, Loner such that 'PPN' entails punishment towards
            cooperators and defectors but not loners.
            action ([1, 0, None]): An action specifying whether the receiving agent's last action was cooperation,
            defection or withdrawal.
        Returns:
            punishment (bool): A True/False value to state whether the receiving agent will or will not receive
            punishment from this agent.
        """
        return _Strategy.punishment_strategies[ID]["function"](action)


class _Norm:
    """
    A population-wide object specifying the rules of reputation assignment after a round of the Public Goods Game. The population updates agents' reputations following any round of the OPGG with _Norm.assign_reputation taking an action (one of 1, 0, None representing C, D, L) and returning 1, 0, -1 representing a good, intermediate and bad reputation. This reputation map is selected from `_Norm.social_norm_rules` and stored in `_Norm.reputation_matrix`.

    Class variables
        - Use `_Norm.social_norm_names` fpr a list of the social norm names
        - Use `_Norm.social_norm_rules` for the dictionary of actions: reputation key/value pairs
        - Use `_Norm.social_norm_descriptions` for an explanation of each norm

    Args:
        norm_name (str) = Name of the social norm of the population, can be one of ["Loner", "Defector", "Neither", \
        None].
    """

    social_norm_rules = {
        "Loner": {1: 1, 0: 0, None: -1},
        "Defector": {1: 1, 0: -1, None: 0},
        "Neither": {1: 1, 0: 0, None: 0},
        "Both": {1: 1, 0: -1, None: -1},
        None: None,
    }

    social_norm_descriptions = {
        "Loner": {
            "Name": "Loner",
            "Summary": "This is the social norm that is biased against people who do not contribute.",
            "Rule": "Assign 1 (good) if player contributed, 0 (okay) if he defected, -1 (bad) if he did not "
            "participate.",
        },
        "Defector": {
            "Name": "Defector",
            "Summary": "This is the social norm that is biased against people who participate but do not contribute.",
            "Rule": "Assign 1 (good) if player contributed, 0 (okay) if he didn't participate, -1 (bad) if he "
            "participated but did not contribute.",
        },
        "Neither": {
            "Name": "Neither",
            "Summary": "This is the social norm that is indifferent towards Loners or Defectors.",
            "Rule": "Assign 1 (good) if player contributed, 0 (okay) if he did not contributed or if he did not "
            "participate.",
        },
        "Both": {
            "Name": "Both",
            "Summary": "This social norm hates both Loners and Defectors.",
            "Rule": "Assign 1 (good) if player contributed, -1 (bad) if he defected or abstained.",
        },
        None: {
            "Name": "None",
            "Summary": "This is the social norm that ignores reputation.",
            "Rule": "No actions are assigned reputations. This is only valid in populations with only strategies "
            "I, II, and III. Errors will be raised to prevent this.",
        },
    }
    social_norm_names = [*social_norm_descriptions.keys()]

    def __init__(self, norm_name):
        self.norm_name = norm_name
        self.reputation_matrix = _Norm.social_norm_rules[norm_name]

    def _assign_reputation(self, contribution):
        """
        Given a contribution, assign new reputations according to the social norm type
        Args:
            contribution (int or None): An agent's contribution (1) or defection (0) or loner (None)

        Returns:
            A new reputation, either good (1), okay (0), bad (-1)
        """
        return self.reputation_matrix[contribution]


class _Agent:
    __slots__ = ["ID", "strategy", "utility", "reputation", "tracker"]

    def __init__(self, ID, strategy):
        self.ID = ID
        self.strategy = {
            key: val
            for key, val in zip(["behavioural", "punishment"], strategy.split("_"))
        }
        self.strategy["ID"] = strategy
        self.utility = 1
        self.reputation = 1
        self.tracker = None
        logging.info(
            f"Agent {self.ID} created with r={self.reputation} & s={self.strategy}"
        )

    def _choose_action(self, average_reputation):
        """
        _Agent chooses it's action in a public good game given the average reputation of the other players. Choice
        depends on the player's strategy.
        Args:
            average_reputation (float): The average reputation in [-1, 1] of the other players in the group

        Returns:
            Action (str): Contributes 1 or 0 if playing, if not participating, then return None
        """
        return _Strategy._choose_action(
            self.strategy["behavioural"], average_reputation
        )

    def choose_punishment(self, opponent_action):
        """
        _Agent chooses it's punishment for his opponent depending on his own punishment strategy and the opponent's
        previous action.
        Args:
            opponent_action ([1, 0, None]): Opponent's last action of cooperation/defection/withdrawal from the game.
        Returns:
            punishment (bool): Returns True/False depending on whether the agent punishes his opponent or not.
        """
        return _Strategy._choose_punishment(
            self.strategy["punishment"], opponent_action
        )

    def __str__(self):
        return (
            f"A(ID={self.ID}, "
            f"s={self.strategy['behavioural']}_{self.strategy['punishment']}, "
            f"u={round(self.utility, 4)}, "
            f"r={self.reputation})"
        )


class Population:
    """Simulate a population of agents playing public goods games.

    Generate a population of agents to play the Optional Public Good Game (OPGG).

    Examples:
        Define a Configuration object containing all the parameters for a single simulation. Create a Population object
        using the parameter dictionary. Run the simulation. Results will be automatically exported.

        >>> import opgar
        >>> C = Configuration(...) # See Configuration class docstring.
        >>> P = Population(C)
        >>> P.simulate()
    """

    def __init__(self, config):
        """
        Create a Population object ready to start a simulation.

        Args:
            config (opgar.Configuration): Object of simulation parameters.
        """

        # Parameters
        self.config = config
        self.strategies = set(config.composition.keys())

        # Generate agents with strategy distribution
        self.agents = self._generate_population(config.N, config.composition)
        self.agents_by_strategy = self._generate_population_by_strategy(
            self.agents, self.strategies
        )
        self.social_norm_type = config.social_norm
        self.social_norm = _Norm(config.social_norm)

        # Granular record of actions
        self.track_strategy_actions = False

    def simulate(self, t_step=None, job_id="", rng_seed=None, disable_bar=False, disable_export=False, use_group_selection=True, record_actions_by_strategy=False):
        """
        Simulate multiple rounds of public goods games

        Args:
            t_step (int): Export data every t_step periods. For efficient memory usage in very long simulations. Default is None.
            job_id (str): Prepend all exported data files with this job_id.
            rng_seed (int): Set the seed for numpy RNG. (WARNING: Not functional yet)
            disable_bar (bool): Remove the tqdm progress bar if True. Default is False.
            disable_export (bool): Do not export datafiles if True. Default is False.

        Returns:
            Tuple of dataframes (avg_payoffs, composition, transitions, punishment_tracker, action_tracker, reputation_tracker)

        Notes:
            Use t_step in conjunction with disable_export=False for long simulations to reduce load on system memory. Every t_step time-steps a new dataframe will be exported. 
            
        """

        logging.info("New Simulation starting")

        if record_actions_by_strategy:
            self.track_strategy_actions = True

        if rng_seed is not None:
            with open(f"j{job_id}_seed.txt", "w") as f:
                f.write(str(rng_seed))

        using_batch = True if self.config.t > 250000 else False
        t_start = 0
        t_end = self.config.t
        if t_step is None:
            t_step = self.config.t
        batches = list(range(t_start, t_end, t_step)) + [t_end]

        transition_matrix = self._generate_transition_matrix(self.strategies)

        for (batch_start, batch_end) in zip(batches[:-1], batches[1:]):
            period_results = {}.fromkeys(range(batch_start, batch_end))
            punishment_tracker = {}.fromkeys(range(batch_start, batch_end))
            cooperative_action_tracker = [None] * (batch_end - batch_start)
            reputation_tracker = [None] * (batch_end - batch_start)
            strategy_actions_tracker = {}.fromkeys(range(batch_start, batch_end))

            for t in trange(batch_start, batch_end, desc=f"T=[{batch_start:,}-{batch_end:,}]", disable=disable_bar):
                logging.info(f"T={t} starting")
         
                groups_of_player_IDs = self._get_groups()
                all_action_tracker = {}.fromkeys(["_".join(s) for s in product(self.strategies, ["1", "0", "None"])], 0)

                # First game
                self._play_public_good_game(groups_of_player_IDs, all_action_tracker)
                if self.social_norm_type:
                    self._update_reputations()
                punishment_tracker[t] = self._punish(groups_of_player_IDs, None)
                
                # Following games occur with probability omega
                while np.random.random() < self.config.omega:
                    self._play_public_good_game(groups_of_player_IDs, all_action_tracker)
                    if self.social_norm_type:
                        self._update_reputations()
                    punishment_tracker[t] = self._punish(groups_of_player_IDs, punishment_tracker[t])

                # Neaten results
                period_results[t] = self._get_period_result()
                punishment_tracker[t] = self._neaten_punishment_results(punishment_tracker[t])
                strategy_actions_tracker[t] = all_action_tracker

                # Evolution
                if use_group_selection:
                    self._evolve_group_selection(groups_of_player_IDs, transition_matrix)
                    self._mutate()
                else:
                    self._evolve_randnowak(transition_matrix)
                
                # Gather extra information and reset
                period_results[t]["Fitness"] = self._get_population_fitness(period_results[t])
                cooperative_action_tracker[t % t_step] = self._record_cooperative_actions()
                reputation_tracker[t % t_step] = self._record_reputations()
                self._reset_population()

            # ----------------------------------------------------------------------
            # POST-PROCESSING OF EACH BATCH
            # ----------------------------------------------------------------------
            processing_start = time()

            # Average Payoffs
            avg_payoffs = pd.concat([period_results[t]["Average Payoffs"] for t in range(batch_start, batch_end)], axis=1).transpose()
            avg_payoffs.index = np.arange(batch_start, batch_end)
            avg_payoffs = avg_payoffs.astype("float16")

            # Population State
            composition = pd.concat([period_results[t]["Composition"] for t in range(batch_start, batch_end)], axis=1).transpose()
            composition.index = np.arange(batch_start, batch_end)
            composition = composition.astype("float16")

            # Actions
            action_tracker = pd.DataFrame(cooperative_action_tracker, columns=["Cooperative", "Non-Cooperative", "Loner"], index=range(batch_start, batch_end))
            action_tracker = action_tracker.astype("float16")
            strategy_actions_tracker = pd.DataFrame.from_dict(strategy_actions_tracker, orient="index", dtype="int16")

            # Strategy transitions
            transitions = [(outerKey, innerKey, innerVal) for outerKey, outerVal in transition_matrix.items() for innerKey, innerVal in outerVal.items() if innerVal != 0]
            transitions = pd.DataFrame(transitions, columns=["Source", "Destination", "#"])
            # Reputations
            reputation_tracker = pd.DataFrame(reputation_tracker, columns=["Good", "Medium", "Bad"], index=range(batch_start, batch_end))
            reputation_tracker.astype("float16")
            
            # Punishments
            punishment_tracker = pd.DataFrame.from_dict(punishment_tracker, orient="index", dtype="float16")
            if not disable_export:
                punishment_tracker.to_csv(f"j{job_id}_punishments_{batch_start}.csv")

                if t_step is not None:
                    batch_code = f"_{batch_start}"
                else:
                    batch_code = ""
                action_tracker.to_csv(f"j{job_id}_actions{batch_code}.csv")
                avg_payoffs.to_csv(f"j{job_id}_payoffs{batch_code}.csv")
                composition.to_csv(f"j{job_id}_composition{batch_code}.csv")
                reputation_tracker.to_csv(f"j{job_id}_reputations{batch_code}.csv")
                transitions.to_csv(f"j{job_id}_transitions{batch_code}.csv")

                if self.track_strategy_actions:
                    strategy_actions_tracker.to_csv(f"j{job_id}_granular_actions{batch_code}.csv")



            processing_end = time()
            print(f"---> Export Batch Data ---> {processing_start-processing_end} seconds elapsed")

        if not disable_export:
            with open(f"j{job_id}_config.json", "w") as f:
                json.dump(self.config.to_dict(), f)

        if batch_end == self.config.t or disable_export is True:
            return avg_payoffs, composition, transitions, punishment_tracker, action_tracker, reputation_tracker
        else:
            logging.warning("Batch processing used. Results are automatically exported as .csv/.json files.")
            return

    def _get_groups(self, N=None, n=None):
        """
        Return a list of lists representing the player IDs for each group

        Args:
            N (int, optional): Size of the Population. Defaults to None.
            n (int, optional): Size of a single group. Defaults to None.

        Returns:
            groups_of_player_IDs (list): List of agent ID numbers (ints).
        """
        if N is None or n is None:
            N = self.config.N
            n = self.config.n

        player_IDs = np.arange(N)
        np.random.shuffle(player_IDs)
        groups_of_player_IDs = np.reshape(player_IDs, (int(N / n), n)) 
        return groups_of_player_IDs

    def _record_reputations(self):
        """
        Return a tuple of the proportion of good, OK, bad people. 
        """
        reps = Counter([agent.reputation for agent in self.agents])
        return (reps[rep]/self.config.N for rep in [1, 0, -1])

    def _record_cooperative_actions(self):
        """
        Return the proportion of cooperative, defective, loner actions at a single time-step.
        """
        actions_C, actions_D, actions_L = 0, 0, 0
        N = self.config.N

        for agent in self.agents:
            if agent.tracker == 1:
                actions_C += 1
            elif agent.tracker == 0:
                actions_D += 1
            else:
                actions_L += 1
        return (actions_C/N, actions_D/N, actions_L/N)

    def _get_population_fitness(self, period_results):
        """Calculate the fitness of the population where F = x1*payoff(x1) + x2*payoff(x2) + ... + xn*payoff(xn).

        Args:
            period_results (pandas.Series): Series containing summary statistics about a single time-step of the simulation.

        Returns:
            fitness (float): The population fitness at a single time-step.
        """ 

        average_payoffs = period_results["Average Payoffs"]
        composition = period_results["Composition"]
        fitness = (average_payoffs * composition).sum()
        return fitness

    def _punish(self, groups, temp_tracker):
        """
        Each player in each group has the opportunity to punish any and all of the other players in his group.

        Args:
            groups (list): List of lists specifying the player IDs in each group
            temp_tracker (dict): Temporary container of all the results from the present time-step.
        """

        if "nopunish" in self.config._meta_data["strategy group"]:
            return

        if temp_tracker is None:
            temp_tracker = {(src, dst): 0 for src, dst in product([1, 0, None],[1, 0, None])}

        logging.info("Punishment period starting")
        total_punishments = 0
        for group in groups:
            for playerID in group:
                # Each person in the group punishes the others 
                punishing_agent = self.agents[playerID]

                # Players "to be or not to be" punished
                rest_of_the_group = [self.agents[ID] for ID in group if ID != playerID]

                for recipient in rest_of_the_group:
                    # Decide whether opponent needs punishment
                    opponents_last_action = recipient.tracker
                    punishment_needed = punishing_agent.choose_punishment(opponents_last_action)

                    # Apply punishment, agent pays the punishment cost, opponent pays the punishment penalty
                    if punishment_needed:
                        punishing_agent.utility -= self.config.gamma
                        recipient.utility -= self.config.beta

                        temp_tracker[(punishing_agent.tracker, recipient.tracker)] += 1
                        total_punishments += 1
                        logging.debug(
                            f"A{punishing_agent.ID}(s={punishing_agent.strategy['ID']}) punished "
                            f"A{recipient.ID}(s={recipient.strategy['ID']}) "
                        )

        logging.info(
            f"Punishment period ending: {total_punishments} punishment(s) enacted"
        )

        return temp_tracker

    def _neaten_punishment_results(self, punishment_tracker):
        """
        Convert the punishment result dictionary from counts to a proportion of total number of punishments.

        Args:
            punishment_tracker (dict): Dictionary tracking the source and target of punishment. i.e. a Cooperator punished a defector exactly 7 times, and punished a loner 2 times.
        
        Return:
            punishment_tracker (dict): Updated tracker.
        """

        # Model does not use punishment
        if punishment_tracker is None:
            return

        total_punishments = sum(punishment_tracker.values())
        if total_punishments != 0:
            for direction, punishment_amount in punishment_tracker.items():
                punishment_tracker[direction] = punishment_amount / total_punishments
        return punishment_tracker

    def _update_reputations(self):
        """
        Iterate through the population and update their reputations based on the social norm and their previous action.
        """

        # If we don't care about reputation
        if self.social_norm is None:
            return

        logging.info(f"Reputations are updated ('{self.social_norm_type}')")

        for agent in self.agents:
            most_recent_action = agent.tracker
            current_reputation = agent.reputation
            new_reputation = self.social_norm._assign_reputation(most_recent_action)
            agent.reputation = new_reputation
            logging.debug(
                f"A{agent.ID}(old={current_reputation}, action={most_recent_action}, new={new_reputation})"
            )

    def _play_public_good_game(self, groups_of_player_IDs, strategy_action_tracker):
        """
        Simulate the public good game (PGG) once for each player in the population

        Args:
            groups_of_player_IDs (list): List of lists describing the groups in which the OPGG is to be played.
            strategy_action_tracker (dictionary): Temporary dictionary tracking the number of C/D/L actions within this time-step.
        """

        # Setup time-step results
        logging.info("Public Goods Game beginning")


        # ----------------------------------------------------------------------
        # Play PGG
        # ----------------------------------------------------------------------
        
        # group size
        n = len(groups_of_player_IDs[0])
        
        # Play PGG in groups of n
        for group in groups_of_player_IDs:
            # Each player calculates the average reputation of the group based on the *rest* of the group
            avg_reps = {}
            for playerID in group:
                avg_reps[playerID] = sum([self.agents[ID].reputation for ID in group if ID != playerID]) / (n - 1)

            # Players decide to contribute 1, contribute 0, or not participate (None)
            group_contribution = [self.agents[ID]._choose_action(average_reputation=avg_reps[ID]) for ID in group]

            # Possible cases
            #   1. (n-1) Loners -> SKIP PGG -> EVERYONE gets sigma
            #   2. Any non-zero amount of Cooperators/Defectors plays PGG as normal

            group_actions = Counter(group_contribution)
            if group_actions[None] >= n - 1:
                # OPGG skipped -> everyone gets the loner's payoff (sigma)
                for playerID, contribution in zip(group, group_contribution):
                    self.agents[playerID].tracker = None
                    self.agents[playerID].utility += self.config.sigma

                logging.info(
                    f"Group of agents ({group}) did not play the PGG, everyone receives {self.config.sigma}."
                )
                
                if self.track_strategy_actions:
                    for playerID in group:
                        strategy_action_tracker[self.agents[playerID].strategy["ID"]+"_"+"None"] += 1
            else:
                # Normal PGG
                try:
                    total_contribution = group_actions[1]  # number of contributors
                except KeyError:
                    total_contribution = 0

                total_participating = (
                    group_actions[1] + group_actions[0]
                )  # number of defectors
                payoff_per_player = (
                    total_contribution * self.config.r / total_participating
                ) 

                for playerID, contribution in zip(group, group_contribution):
                    self.agents[playerID].tracker = contribution
                    if contribution == 1:
                        self.agents[playerID].utility -= 1  
                        self.agents[playerID].utility += payoff_per_player
                    elif contribution == 0:
                        self.agents[playerID].utility += payoff_per_player
                    else:
                        self.agents[playerID].utility += self.config.sigma

                if self.track_strategy_actions:
                    for playerID, contribution in zip(group, group_contribution):    
                        strategy_action_tracker[self.agents[playerID].strategy["ID"]+"_"+str(contribution)] += 1

                logging.info(
                    f"Group of agents ({group}) played the PGG, average payoff was {round(payoff_per_player, 2)} each "
                    f"to {total_participating} agents"
                )

    def _get_period_result(self):
        """
        Get period results in a pandas Series.

        Returns:
            pandas.Series: Contains all information regarding the present time-step.
        """

        # Save all results for the time-step here (possibly multiple rounds of games)
        period_result = {
            "Payoffs": {}.fromkeys(self.strategies, 0),
            "Composition Count": {}.fromkeys(self.strategies, 0),
            "Composition": {}.fromkeys(self.strategies, 0),
            "Average Payoffs": {}.fromkeys(self.strategies, 0),
        }
        # Record total strategy payoffs, strategy composition
        for agent in self.agents:
            period_result["Payoffs"][agent.strategy["ID"]] += agent.utility
            period_result["Composition Count"][agent.strategy["ID"]] += 1

        for strategy in self.strategies:
            # If no players left in the strategy, payoff is 0
            player_count = period_result["Composition Count"][strategy]
            if player_count == 0:
                period_result["Average Payoffs"][strategy] = 0
                continue

            # Average strategy payoffs by number of agents using the strategy
            period_result["Average Payoffs"][strategy] = (
                period_result["Payoffs"][strategy] / player_count
            )

            # Get population composition as a proportion instead of relative size
            period_result["Composition"][strategy] = (
                period_result["Composition Count"][strategy] / self.config.N
            )

        # Flatten period_result from dict-of-dicts to single-level dict
        #   e.g. {"a1": {"a2": []}, "b1"={"b2":[]}} -> {('a1', 'a2'): [], ('b1', 'b2'): []}
        #   This makes it easier to convert it to a MultiIndex Series/DataFrame later when concatenating results over
        #   multiple time-steps.
        flat_dict = Utils.flatten_nested_dict(period_result)

        classed_series = pd.Series(flat_dict)
        classed_series.index.names = ["Statistic", "Strategy"]
        return classed_series

    @deprecated
    def _evolve_randnowak(self, transitions):
        """
        Update agent strategies according to Rand and Nowak 2011. 

        We use a frequency dependent Moran process with an exponential payoff
        function. In each round, agents interact at random. One agent is then
        randomly selected to change strategy. With probability u, a mutation
        occurs and the agent picks a new strategy at random. With probability
        1 − u, the agent adopts the strategy of another agent j, who is
        selected from the population with probability proportional to
        exp(utility(j)) where utility(j) is the payoff of agent j.

        Args:
            transitions (dict): Nested dictionary of counts of strategy transitions.
        """

        # Choose a single agent to evolve
        evolving_agent = np.random.choice(self.agents)
        evolving_agent_old_strategy = evolving_agent.strategy["ID"]

        if np.random.random() < self.config.u:
            possible_strategies = self.config._meta_data["strategy group"]
            possible_strategies = _Strategy.strategy_groups[possible_strategies]
            strategy_to_switch_to = np.random.choice(possible_strategies)
        else:
            # Identify the strategy to switch to by the exponential of their utilities
            strategies = [
                agent.strategy["ID"]
                for agent in self.agents
                if agent is not evolving_agent
            ]
            utilities = np.array(
                [
                    np.exp(agent.utility)
                    for agent in self.agents
                    if agent is not evolving_agent
                ]
            )
            normalised_utilities = utilities / utilities.sum()
            strategy_to_switch_to = np.random.choice(
                a=strategies, size=1, p=normalised_utilities
            )[0]

        # Update the evolving player's new strategy
        if evolving_agent_old_strategy == strategy_to_switch_to:
            pass
        else:
            self._change_agent_strategy(evolving_agent, strategy_to_switch_to)
            transitions[evolving_agent_old_strategy][strategy_to_switch_to] += 1

    def _evolve_group_selection(self, groups, transitions):
        """
        Evolution implementation using group-selection. 

        Players encounter another player from their own group with probability 1-m and from another randomly chosen 
        group with probability m. An individual i who encounters an individual j imitates j with probability 
        W_j / (W_j + W_i) where W_x is the payoff of individual x including the costs of giving or receiving punishment.

        Args:
            groups (list): Lists the groups in which the OPGG has been played.
            transitions (dict): Nested dictionary of counts of strategy transitions.
        """

        evolving_agent_id = np.random.choice(range(self.config.N))
        evolving_agent = self.agents[evolving_agent_id]
        evolving_agent_strategy = evolving_agent.strategy["ID"]
        evolving_agent_group = [group for group in groups if evolving_agent_id in group][0]
                
        if np.random.random() < 1 - self.config.m:
            # selection from another individual in the same group
            other_agent_id = np.random.choice(evolving_agent_group)
            while other_agent_id == evolving_agent_id:
                other_agent_id = np.random.choice(evolving_agent_group)
            other_agent = self.agents[other_agent_id]
            other_agent_strategy = other_agent.strategy["ID"]
        else:
            # selection from someone in another group
            other_group = groups[np.random.randint(len(groups))]
            while evolving_agent_id in other_group:
                other_group = groups[np.random.randint(len(groups))]
            other_agent_id = np.random.choice(other_group)
            other_agent = self.agents[other_agent_id]
            other_agent_strategy = other_agent.strategy["ID"]
        
        probability_of_update = np.exp(other_agent.utility) / (np.exp(evolving_agent.utility) + np.exp(other_agent.utility))
        if probability_of_update < 0: 
            logging.warn(f"Probability of update ('{probability_of_update}') is negative!")
        
        if np.random.random() < probability_of_update:
            self._change_agent_strategy(evolving_agent, other_agent_strategy)
            if evolving_agent_strategy != other_agent_strategy:
                transitions[evolving_agent_strategy][other_agent_strategy] += 1
        
    def _mutate(self):
        """
        With probability epsilon, introduce a mutant into the population.

        This should probably not be used in conjunction with self.evolve_randnowak since that incorporates mutation in itself.
        """
        if np.random.random() < self.config.epsilon:
            mutant = np.random.choice(self.agents)
            mutated_strategy = np.random.choice([*self.config.composition.keys()])
            while mutated_strategy == mutant.strategy["ID"]:
                mutated_strategy = np.random.choice([*self.config.composition.keys()])
            self._change_agent_strategy(mutant, mutated_strategy)

    def _change_agent_strategy(self, agent, new_strategy):
        """
        Change an agent's strategy updating the Population.

        Args:
            agent (opgar.Agent): An agent to mutate
            new_strategy (str): A new strategy.
        """

        # Update strategy in Agent object
        old_strategy = agent.strategy["ID"]
        agent.strategy = {
            key: val
            for key, val in zip(["behavioural", "punishment"], new_strategy.split("_"))
        }
        agent.strategy["ID"] = new_strategy

        # Update population census by strategy
        if new_strategy not in self.agents_by_strategy.keys():
            self.agents_by_strategy[new_strategy] = set()
        self.agents_by_strategy[old_strategy].remove(agent)
        self.agents_by_strategy[new_strategy].add(agent)

        logging.info(
            f"Agent {agent.ID} switched from {old_strategy} to {new_strategy}."
        )

    def _reset_population(self):
        """
        Reset all agents and the population between time-steps.

        Things that are reset:
            An agent's utility is reset to 1

        Things that do not reset:
            An agent's reputation
            Any population parameters
        """
        logging.info("Agents reset")
        for agent in self.agents:
            agent.utility = 1
            agent.reputation = 1

    @staticmethod
    def _log_series(name, series):
        """
        Given a pandas.Series, return a one-line string representation for logging.
        """
        return f"{name}: {', '.join([k[1] + '->' + str(round(v, 4)) for k, v in series.items()])}"

    @staticmethod
    def _generate_population(N, composition):
        """
        Create the population of agents with the required distribution of strategies.

        Given some proportion of strategies within the population as a dictionary of strategy: 'proportion of 1' pairs,
        convert the proportions to numbers of agents by multiplying by the size of the population. These may or may not
        be non-integers, so we use the Largest Remainder method to assign the last few agents.

        Args:
            N (int): Size of the network
            composition (dict): Dictionary of strategies and their proportions within the population

        Returns:
            List of agents with a strategy distribution equal to that of composition
        """
        # Partition N players into fractions of 1 as accurately as possible
        proportions = Population._distribute_over_N(composition, N)

        agents = []
        id_counter = 0
        for strategy, count in proportions.items():
            for _ in range(int(count)):
                agents.append(_Agent(ID=id_counter, strategy=strategy))
                id_counter += 1

        return agents

    @staticmethod
    def _generate_population_by_strategy(agentSet, strategies):
        """
        Create an alternative view of Population.agents by strategy.

        Args:
            agentSet (list): List of agent objects

        Returns:
            Dictionary of agents where the keys:value pairs are strategies and lists of the agents running them
        """
        agents_by_strategy = {}.fromkeys(strategies)

        # Using "agents_by_strategy = {}.fromkeys(strategies, set())" makes each value point to the same set object
        for key, value in agents_by_strategy.items():
            agents_by_strategy[key] = set()

        # Organise each agent by their strategy
        for agent in agentSet:
            agents_by_strategy[agent.strategy["ID"]].add(agent)

        return agents_by_strategy

    @staticmethod
    def _generate_transition_matrix(strategies):
        """Initialise a strategy transition matrix.

        Args:
            strategies (list): List of strategies in the population

        Returns:
            matrix (dict): Nested dicts in the form of an KxK matrix where K is the number of strategies.
        """
        template = {}.fromkeys(strategies, 0)
        matrix = {}.fromkeys(strategies)
        for strategy in strategies:
            matrix[strategy] = copy.deepcopy(template)
        return matrix

    @staticmethod
    def _distribute_over_N(dist, N):
        """
        Distribute N into k classes as specified by the distribution in dist.
        
        Args:
            dist (list): The distribution of the population over the strategies.
            N (int): The total number of players to assign to a strategy.
        """

        # Roughly distribute N agents to the k classes by the proportions in dist allowing only integers
        dist_over_N = dist * N
        dist_over_N = dist_over_N
        temp_dist = dist_over_N.astype(int)
        logging.debug(Population._log_series("Allocated: ", temp_dist))

        # How many remaining agents need to be assigned a class
        remaining = N - sum(temp_dist)
        logging.debug(f"Remaining: {remaining}")
        if remaining == 0:
            return temp_dist

        logging.info("Approximating New Population using Largest Remainder Method")
        while remaining > 0:
            # get remainders
            unallocated_dist = dist_over_N - temp_dist
            logging.debug(Population._log_series("Unallocated: ", unallocated_dist))

            # get largest remainder
            max_remainder = max(unallocated_dist)
            logging.debug(f"Max Remainder: {max_remainder}")

            # get list of indices with the max remainder
            idxs = unallocated_dist[unallocated_dist == max_remainder]
            logging.debug(f"Indices: {idxs.index}")

            # Choose strategy to increment (if multiple choose randomly)
            new_strat = np.random.choice(idxs.index)
            logging.debug(f"New Strategy: {new_strat}")

            # Allocate agent
            temp_dist[new_strat] += 1
            remaining -= 1
        return temp_dist

    def __str__(self):
        s = []
        for agent in self.agents:
            s += [
                f"A(ID={agent.ID}, s={agent.strategy['ID']}, u={round(agent.utility, 4)}, r={agent.reputation[-1]})"
            ]
        output = "\n".join(s)
        return output

