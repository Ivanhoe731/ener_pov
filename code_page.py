import streamlit as st

def code_page():
    # Add code for displaying the source code
    st.title("Source Code")


        # Display the source code of the model classes and functions
    st.subheader("Model Source Code")
    st.code(get_model_source_code(), language="python")

def get_model_source_code():
    # Paste the code directly here as a string
    source_code = """

## Loading neccessary libraries 

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
import scipy.stats as stats
import pandas as pd
import random
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import seaborn as sns


## Constant variables which will be updated in newer versions of the model to parameters of the Country class 
## All variables are by default unitless, but have assgined units for beeter illustration


COST_INSULATION = 0.45 # cost of insulating square foot, can be considered cents of euro
TRESHOLD_LEVEL = 0.25  # treshold level for arrears 
RESTORATION_COST = 1122 * COST_INSULATION # Restoration cost of one household, 1122 is average european house size in square feet
DWELLING_RESTORATION_LIMIT = 900 # upper limit of restoring one dwelling 
RESTORATION_DWELLING_REDUCTION = 0.5 # reduction of dwelling effectivity
RESTORATION_STEP = 7 # the step of simulation in which restoration policy executes
GRANULARITY = 'MONTH'  ## in further version of the model can be used to select granularity of studied steps
INABILITY_TRESHOLD = 0.1 # percentage of disposable income allocated to energy utility bills treshold, above which household is marked as in "inability" to keep dwelling adequately warm
ARREARS_TRESHOLD = 0.75 # similar to inability, but decisive treshold for arrears 
MPS_VALUES = [0.1, 0.13, 0.17, 0.20, 0.25] # Marginal Propensity to save by qunitiles from lowest to highest 
SHOCK_INDEX = 0  # price index to which shokc shall be applied, 0 is yellow fuel, 1 brown fuel  
SHOCK_MAGNITUDE = 0.5 # magnitude of price shock in percentage 
SHOCK_STEP = 5 # step in which price shock happens, if the schock is executed in the simulation 
ALLOWENCE_CHEQUE = 50 # amount of the allowence cheque assigned to households 
RESTORATION_BUDGET = 12000 # overall budget allocated to restoration policy 
ALLOWENCE_BUDGET = 50000 # overall budget allocated to allowence policy 
ALLOWENCE_FROM = 9 # step in which allowence policy is excercissed

class Household(Agent):
    def __init__(self, unique_id, model, disposable_income, dwelling, inability=False, restoration_recieved = False):
        super().__init__(unique_id, model)
        self.disposable_income = disposable_income # disposalble income of household
        self.dwelling = dwelling # energy intensity of household 
        self.inability = inability # boolean denoting weather household is in inability 
        self.savings = 0  # savings is initialized as 0 
        self.arrears = 0 # amount 
        self.x = random.uniform(0, 1) # random x coordinate to visualize agents 
        self.y = random.uniform(0, 1) # random y coordonate to visualize agents  
        self.restoration_recieved = restoration_recieved # boolean to denote weather agent recieved restoration help in restoration program 
        self.allowence_recieved = False # wheather household recieved allowence under allowence program
        self.allowence_cheques_sum = 0 # total amount of allowence cheques assigned to agent 
        self.mps = 0 # marginal propensity to save initialized to zero and later changed accordignly to income quintile 
        self.daai = 0 # disposable income after allowences 

    @property
    def energy_cost(self):
        return self.model.energy_price * (self.dwelling)
    
    # Assign marginal propensity to save 
    def update_mps(self):
        # Get all agent incomes
        incomes = [a.disposable_income for a in self.model.schedule.agents]

        # Calculate the quintile thresholds
        thresholds = np.quantile(incomes, [0.2, 0.4, 0.6, 0.8]) # artbitratly set income tresholds 

        # Determine which quintile the agent's income belongs to
        quintile = np.digitize(self.disposable_income, thresholds)

        # Set the MPS based on the quintile
        self.mps = MPS_VALUES[quintile]
        
    def step(self):
        # Calculate energy costs
        energy_costs = self.energy_cost

        self.update_mps()

        # Check if the agent is in arrears
        if energy_costs > self.disposable_income * ARREARS_TRESHOLD:  # now considering arrears if energy cost exceeds ARREARS_TRESHOLD of disposable income
            self.arrears += energy_costs
        else:
            self.savings += (self.disposable_income - energy_costs) * self.mps

        # If the agent has arrears
        if self.arrears > 0:
            if self.savings >= self.arrears:  # and if savings can cover it
                self.savings -= self.arrears
                self.arrears = 0 # clear the arrears amount
            else:  # if savings cannot cover it
                self.arrears = self.arrears - self.savings  # set arrears to remaining amount
                self.savings = 0

        self.inability = self.energy_cost > self.disposable_income * INABILITY_TRESHOLD  # agent is in inability state if energy costs exceed 10% of income

        # Update disposable income based on growth rate
        growth_rate = np.random.uniform(self.model.growth_boundaries[0], self.model.growth_boundaries[1])
        self.disposable_income *= (1 + growth_rate)


class Country(Model):
    
    # method to create such set of incomes which are close or equall to set gini coefficinet
    @classmethod
    def calculate_gini(cls, incomes):
        incomes = np.sort(incomes)
        n = len(incomes)
        index = np.arange(1, n + 1)
        return ((np.sum((2 * index - n  - 1) * incomes)) / (n * np.sum(incomes)))


    # method is using the target gini coefficient to assign incomes according to gamma distribution 
    # The process find the distribution of incomes itteratively 
    @classmethod
    def generate_income_distribution(cls, num_people, median_income, gini_target, lower_bound):
        alpha = (gini_target + 1) / (2 - gini_target)
        for _ in range(10000):
            incomes = stats.gamma.rvs(alpha, scale=median_income/alpha, size=num_people)
            incomes = np.clip(incomes, lower_bound, None)  # None means there is no upper bound
            gini_current = cls.calculate_gini(incomes)
            if np.isclose(gini_current, gini_target, atol=0.01):
                return incomes
            elif gini_current < gini_target:
                alpha *= 0.9
            else:
                alpha *= 1.1

        raise Exception(f"Failed to reach target Gini coefficient in 1000 iterations. Current Gini: {gini_current}")
    

    # Restoration program method sorts the agents by incomes and if the program is executed restores dwelling of agents sorted by income and dwelling intensity 
    def restoration_program(self):

        global RESTORATION_DWELLING_REDUCTION

        # Sort the agents by income in ascending order
        sorted_agents = sorted(self.schedule.agents, key=lambda agent: (agent.disposable_income, -agent.dwelling)) # agents sorted by ascending income and descending dwelling 

        for agent in sorted_agents:
            if self.restoration_budget >= self.restoration_cost:
                # Assign the restoration
                agent.restoration_recieved = True

                agent.dwelling *= (1- RESTORATION_DWELLING_REDUCTION)

                # Deduct the restoration cost from the budget
                self.restoration_budget -= self.restoration_cost
            else:
                # If the budget is not sufficient for another restoration, break the loop
                break

    # allowence program works in similar manner to restoration program but sorts the agents only by income        
    def allowence_program(self):

        # Get a list of all agents' incomes
        incomes = [agent.disposable_income for agent in self.schedule.agents]

        # Compute the 40th percentile of income, which will serve as the cutoff
        income_cutoff = np.percentile(incomes, 40)

        # Sort the agents by income in ascending order
        sorted_agents = sorted(self.schedule.agents, key=lambda agent: (agent.disposable_income))

        for agent in sorted_agents:
            # Check if the agent's income is below the cutoff
            if agent.disposable_income <= income_cutoff and agent.inability:
                if self.allowence_budget >= self.allowence_cheque:
                    # Assign the allowance
                    agent.allowence_recieved = True
                    agent.allowence_cheques_sum += self.allowence_cheque
                    agent.daai = agent.disposable_income - self.allowence_cheque 
                    agent.disposable_income += self.allowence_cheque

                    # Deduct the restoration cost from the budget
                    self.allowence_budget -= self.allowence_cheque
                else:
                    # If the budget is not sufficient for another restoration, break the loop
                    agent.daai = agent.disposable_income
                    break
            else:
                agent.daai = agent.disposable_income

    @staticmethod
    def assign_inability(agents, inability_start):

        agents.sort(key=lambda x: x.disposable_income)

        num_agents = len(agents)
        num_unable = int(inability_start * num_agents)
        inability_per_quintile = np.array([2, 1.6, 1.2, 0.8, 0]) # These values are set arbitrarily 
        inability_per_quintile *= num_unable / np.sum(inability_per_quintile)
        inability_per_quintile = inability_per_quintile.astype(int)

        assigned_unable = 0
        for i in range(5):  # For each quintile
            start = i * num_agents // 5
            end = (i + 1) * num_agents // 5 if i < 4 else num_agents
            quintile = agents[start:end]
            num_unable_quintile = min(inability_per_quintile[i], len(quintile))
            unable_agents = np.random.choice(quintile, num_unable_quintile, replace=False)
            for agent in unable_agents:
                agent.inability = True
            assigned_unable += num_unable_quintile

        # Assign remaining inability if any
        i = 0
        while assigned_unable < num_unable:
            start = i * num_agents // 5
            end = (i + 1) * num_agents // 5 if i < 4 else num_agents
            quintile = [agent for agent in agents[start:end] if not agent.inability]
            if quintile:
                agent = np.random.choice(quintile)
                agent.inability = True
                assigned_unable += 1
            else:
                i += 1

    @staticmethod
    def assign_arrears(agents, arrears_start):
 
        agents.sort(key=lambda x: x.disposable_income)

        num_agents = len(agents)
        num_arrears = int(arrears_start * num_agents)
        arrears_per_quintile = np.array([2, 1.6, 0, 0, 0])
        arrears_per_quintile *= num_arrears / np.sum(arrears_per_quintile)
        arrears_per_quintile = arrears_per_quintile.astype(int)

        assigned_arrears = 0
        for i in range(5):  # For each quintile
            start = i * num_agents // 5
            end = (i + 1) * num_agents // 5 if i < 4 else num_agents
            quintile = agents[start:end]
            num_arrears_quintile = min(arrears_per_quintile[i], len(quintile))
            arrears_agents = np.random.choice(quintile, num_arrears_quintile, replace=False)
            for agent in arrears_agents:
                agent.arrears = 2 * ARREARS_TRESHOLD * agent.disposable_income  # assign 0.6 * disposable_income
            assigned_arrears += num_arrears_quintile

        # Assign remaining arrears if any
        i = 0
        while assigned_arrears < num_arrears:
            start = i * num_agents // 5
            end = (i + 1) * num_agents // 5 if i < 4 else num_agents
            quintile = [agent for agent in agents[start:end] if not agent.arrears > 0]
            if quintile:
                agent = np.random.choice(quintile)
                agent.arrears = ARREARS_TRESHOLD * agent.disposable_income  # assign 0.6 * disposable_income
                assigned_arrears += 1
            else:
                i += 1


    @staticmethod
    def assign_dwelling(agents, energy_price):

        buffer = 0.02  # 2% buffer or margin
        all_incomes = [agent.disposable_income for agent in agents]
        quintile_thresholds = np.quantile(all_incomes, [0.2, 0.4])  # Calculate income thresholds for the bottom two quintiles
        for agent in agents:
            min_dwelling, max_dwelling = 500, 2000
            if agent.inability:
                min_dwelling = max(min_dwelling, int(np.ceil((agent.disposable_income * (INABILITY_TRESHOLD + buffer)) / energy_price)))
            else:
                max_dwelling = min(max_dwelling, int((agent.disposable_income * (INABILITY_TRESHOLD - buffer)) / energy_price))

            # Generate a list of feasible dwelling values
            feasible_dwelling = list(range(min_dwelling, max_dwelling + 1))

            # If the list is not empty, select a random dwelling from the list
            if feasible_dwelling:
                agent.dwelling = np.random.choice(feasible_dwelling)
            else:
                agent.dwelling = min_dwelling  # fallback to the min_dwelling if no other options

            # Check income quintile of the agent and if in inability state, increase dwelling size
            if agent.disposable_income < quintile_thresholds[1] and agent.inability:  # If in bottom two quintiles and inability is True
                agent.dwelling = min(max_dwelling, int(agent.dwelling * 1.25))  # Increase dwelling size by 25%, but not more than max_dwelling

    def __init__(self, N, median_income, min_disposal, gini_target, inability_target, arrears_target, growth_boundaries, prices, shares_p, growth_rate_lower_bound, growth_rate_upper_bound, restoration_ACTIVE = False, allowence_ACTIVE = False, price_shock = False):
        self.num_agents = N # total number of agents 
        self.median_income = median_income # median income used for gini distribution target
        self.min_disposal = min_disposal # minimal disposable income used for fini distribution target
        self.gini_target = gini_target # target of Gini index distribution 
        self.inability_target = inability_target # share of agents with inability in the first step od the simulation 
        self.growth_boundaries = growth_boundaries # how much does disposable income grows every step 
        self.prices = prices # prices of each fuel 
        self.shares_p = shares_p # shares of each fuel in the energy mix of the Country 
        self.growth_rate_lower_bound = growth_rate_lower_bound # lower bounds of growth rates of fuel prices 
        self.growth_rate_upper_bound = growth_rate_upper_bound # upper bound ...
        self.price_shock = price_shock # boolean of introducing price shock in the Country 
        self.restoration_ACTIVE = restoration_ACTIVE # Boolean activating restoration program 
        self.allowence_ACTIVE = allowence_ACTIVE # Boolean activating allowence program 
        self.restoration_cost = RESTORATION_COST # viz Variables  
        self.restoration_budget = RESTORATION_BUDGET # viz Variables 
        self.allowence_budget = ALLOWENCE_BUDGET # viz Variables 
        self.allowence_cheque = ALLOWENCE_CHEQUE # viz Variables 
        self.arrears_target = arrears_target # targeted share of arrears assigned to households 
        
        
        # Schedule randomly initializes agents 
        self.schedule = RandomActivation(self)

        # datacolelctor collects data for each step simulated 
        self.datacollector = DataCollector(
        agent_reporters={"Dwelling": "dwelling", 
                        "Income": "disposable_income",
                        "Inability": "inability",
                        "Savings":"savings",
                        "X":"x",
                        "Y":"y",
                        "MPS":"mps",
                        "DAAI":"daai",
                        "Allowence":"allowence_cheques_sum",
                        "Restoration Aid Recieved":"restoration_recieved",
                        "Arrears":"arrears",
                        "EnergyCost": "energy_cost"},
        model_reporters = {
                        "Inability": lambda model: sum(agent.inability for agent in model.schedule.agents) / model.num_agents,
                        "Arrears": lambda model: sum(agent.arrears for agent in model.schedule.agents) / model.num_agents,
                        "Allowences": lambda model: sum(agent.allowence_cheques_sum for agent in model.schedule.agents) / model.num_agents,
                        "Energy Price": lambda model: model.energy_price,
                        "Price Yellow Fuel": lambda model: model.prices[0],
                        "Price Brown Fuel": lambda model: model.prices[1],
                        "Average Dwelling Over Time": lambda model: sum(agent.dwelling for agent in model.schedule.agents) / model.num_agents,
                        "Restorations": lambda m: sum([agent.restoration_recieved for agent in m.schedule.agents])}
                        )

        # Generate disposable incomes and assign dwelling and technology
        incomes = self.generate_income_distribution(self.num_agents, self.median_income, self.gini_target, self.min_disposal)

        # initial energy price calculated as product of prices and share of each fuel in fuel mix
        self.initial_energy_price = np.dot(self.prices, self.shares_p)

 
        # Create agents
        agents = []
        for i in range(self.num_agents):
            a = Household(i, self, disposable_income=incomes[i], dwelling=0, inability=False)
            agents.append(a)

        # apply the constructed methods to agents 
        self.assign_inability(agents, self.inability_target)
        self.assign_dwelling(agents, self.energy_price)
        self.assign_arrears(agents, self.arrears_target)
        self.assign_mps(agents)

        # Add agents to the model
        for a in agents:
            self.schedule.add(a)

    @staticmethod
    def assign_mps(agents):
        agents.sort(key=lambda x: x.disposable_income)

        num_agents = len(agents)
        for i in range(5):  # For each quintile
            start = i * num_agents // 5
            end = (i + 1) * num_agents // 5 if i < 4 else num_agents
            quintile_agents = agents[start:end]
            for agent in quintile_agents:
                agent.mps = MPS_VALUES[i]

    @property
    def energy_price(self):
        return np.dot(self.prices, self.shares_p)

    def step(self):
        self.datacollector.collect(self)  # Collect data before updating the prices

        # Update prices based on unique growth rates
        growth_rates = np.random.uniform(self.growth_rate_lower_bound, self.growth_rate_upper_bound)

        self.prices = self.prices * (1 + growth_rates)

        if self.schedule.steps == SHOCK_STEP and self.price_shock == True:
            self.prices[SHOCK_INDEX] *= (1 + SHOCK_MAGNITUDE)

        # Execute the restoration program if it is enabled
        if self.schedule.steps == RESTORATION_STEP and self.restoration_ACTIVE:
            self.restoration_program()

        if self.schedule.steps >= ALLOWENCE_FROM and self.allowence_ACTIVE:
            self.allowence_program()

        self.schedule.step()

    """
    return source_code