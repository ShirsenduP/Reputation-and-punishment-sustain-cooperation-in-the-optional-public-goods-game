# Reputation and (anti-social) Punishment Stabilize Cooperation in the Optional Public Good Game

## Introduction

This is the source code used to simulate the experiments detailed in Reputation and (anti-social) punishment stabilize cooperation in the optional public goods game. 

## Setup instructions
`opgar` is written in python3. 

1. `cd` into the directory called `opgar` to wherever you downloaded it (your working directory should contain the README file). The directory structure should look like this.

```bash
opgar/
├── README.md
├── opgar
│   ├── __init__.py
│   ├── opgar.py
│   └── utils.py
├── requirements.txt
└── setup.py
```

3. Create a virtual environment using `python -m venv env` where `env` can be anything you want.
4. Activate the virtual environment with `source env/bin/activate` (linux) or `env\Scripts\activate` (windows)
5. Run `pip install .` to install the `opgar` package and all dependencies in the virtual environment `env`.

## OPGAR

The **O**ptional **P**ublic **G**ood game **A**nd **R**eputation package is made up of two modules `opgar.py` and `utils.py`. The vast majority of functionality is found in the former of these two scripts. `opgar.py` consists of two "public" classes `Configuration` and `Population` that use functionality from "private" classes `_Strategy`, `_Norm`, `_Agent`.

The names of the strategies as defined in the paper are renamed for ease of use. Within this package, the strategies are referred to using Roman numerals, namely I, II, III, ..., XI. A mapping is provided in `_Strategy.strategy_name_mapping` which ignores the punishment variants. The utility function `_Utils.strategy_to_latex` can be used to convert strategies that include a punishment variant to the version as seen in the paper. 

Use `help(Configuration)` or `help(Population)` for documentation. 

### Configuration
 
This class is used to specify the parameters for a single simulation. The parameters are:
- `N` number of agents,
- `n` size of optional public good game group,
- `r` synergy factor,
- `sigma` loner's payoff,
- `t` maximum number of time-steps to simulate,
- `composition` starting population state (see the `_Strategy` class docstring),
- `norm` the social norm used by the agents (see the `_Norm` class docstring),
- `gamma` the cost to punish someone,
- `beta` the penalty paid if punished,
- `u` the probability of mutation in the model of Rand and Nowak 2011 evolutionary mechanism,
- `m` the degree of evolutionary mixing in group selection,
- `epsilon` the rate of mutation (only used in Rand and Nowak evolutionary mechanism),
- `strategy_group` the particular model being played (see the `_Strategy` class docstring),
- `omega` the likelihood of multiple rounds of the game within a single time-step.

**Example Usage:**
```python
from opgar import Configuration, _Strategy, Population

strategies = _Strategy.strategy_groups["all"]
config = Configuration(
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
```

### Population

This class will initialise and simulate an agent-based model using the defined parameters in the `Configuration` class. Begin the simulation with `Population.simulate`. Results are automatically exported upon completion into the current working directory. If the simulation takes a long time, storing large dataframes in memory can become taxing, so use the `t_step` with `disable_export=False` parameters to batch export results as `.csv` files every `t_step` time-steps. 

**Example Usage**

This is the most simple way to get running.
```python
config = Configuration(...) # as above
model = Population(config)
model.simulate()
```

This will export the configuration as a `.json` and the results as `.csv` files.

```python
config = Configuration(...) # as above
model = Population(config)
model.simulate(t_step=25000, disable_export=False)
```

This will not export any data, but will return all results in a tuple. The tuple has 6 pandas dataframes/series containing the following information as time-series by strategy/action:
1. Payoffs
2. Population composition
3. Transitions
4. Punishment tracker
5. Action tracker
6. Reputation tracker

This method is not recommended for lengthy simulations.

```python
config = Configuration(...) # as above
model = Population(config)
results = model.simulate(disable_export=True)
```

### The other classes (_Strategy, _Norm, _Agent)

These classes are mainly used by the `Population` class to run the agent based model:
- `_Strategy` contains lambda functions used by agents to program their actions. Generally, each agent has a strategy (`_Agent.strategy`): 
    ```python
    _Agent.strategy["ID"] = "I_NPN"
    _Agent.strategy["behavioural"] = "I"
    _Agent.strategy["punishment"] = "NPN"
    ```
    The agents use the behavioural and punishment IDs to look-up their decisions in `_Strategy.behavioural_strategy` or `_Strategy.punishment_strategies`.

- `_Norm` works in a similar manner. At initialisation, the reputation matrix of the population is created using the rules of the specific social norm. `Population.social_norm.assign_reputation` is used to assign reputations. 

Some shortcuts 

- `_Strategy.strategy_groups` is a dictionary of lists of strategies. These can be used in the `composition` parameter in the   `Configuration` constructor. Here, `composition` expects a dictionary where key/value pairs are strategies and their initial weighting in the population e.g. `composition={"I_NNN":1/4, "II_NNN":1/4, "III_NNN":1/4, "III_NPN":1/4}`. This is equivalent to 

    ```python
    strategies = _Strategy.strategy_groups["prosocial"]
    composition = {}.fromkeys(strategies, 1/len(strategies))
    ```

    The four models that are discussed in the paper are Baseline, Punishment, Reputation and Reputation + Punishment which can be achieved with the following initial compositions:
    ```python
    baseline = _Strategy.strategy_groups["purenopunish"]
    punishment = _Strategy.strategy_groups["pure"]
    reputation = _Strategy.strategy_groups["nopunish"]
    reputation_and_punishment = _Strategy.strategy_groups["all"]
    ```

- `_Norm.social_norm_rules` provides the action -> reputation mappings for each of the social norms. 
  - `"Defector"` - Anti-Defector norm
  - `"Loner"` - Anti-Loner norm
  - `"Neither"` - Anti-Neither norm
  - `"Both"` - Anti-Both norm
  - `None` - No social norm

    A description of these are found in `_Norm.social_norm_descriptions`. 

- A list of all possible strategies and their punishment variants is found in `_Strategy.all_strategies`.

If you use parts of our code, please acknowledge us by citing our paper. 

**Authors**

Shirsendu Podder - s.podder@live.co.uk\
Simone Righi\
Francesca Pancotto
