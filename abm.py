import streamlit as st
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import pandas as pd
import numpy as np
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.space import SingleGrid
from mesa.visualization.modules import CanvasGrid
import math
from IPython.display import clear_output
import scipy.stats as stats # This is causing problem when deploying, change the code? 
import random
from numpy.random import choice
from numpy import exp
from ipywidgets import interact
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px


# COST_INSULATION = 0.45
# TRESHOLD_LEVEL = 0.25 
# RESTORATION_COST = 1122 * COST_INSULATION
# DWELLING_RESTORATION_LIMIT = 900 ## FUTURE FEATURE
# RESTORATION_DWELLING_REDUCTION = 0.5
# RESTORATION_STEP = 7
# GRANULARITY = 'MONTH'  ## ELABORATE ON THIS IN THE FURTHER VERSION
# # INABILITY_TRESHOLD = 0.1
# # ARREARS_TRESHOLD = 0.75
# # MPS_VALUES = [0.1, 0.13, 0.17, 0.20, 0.25] # Marginal Propensity to Save
# SHOCK_INDEX = 0
# SHOCK_MAGNITUDE = 0.5
# SHOCK_STEP = 5
# ALLOWENCE_CHEQUE = 50
# RESTORATION_BUDGET = 12000
# ALLOWENCE_BUDGET = 50000
# ALLOWENCE_FROM = 9


class Household(Agent):
    def __init__(self, unique_id, model, disposable_income, dwelling, inability=False, restoration_recieved = False):
        super().__init__(unique_id, model)
        self.disposable_income = disposable_income
        self.dwelling = dwelling
        self.inability = inability
        self.savings = 0  # savings is initialized as 0
        self.arrears = 0
        self.x = random.uniform(0, 1)
        self.y = random.uniform(0, 1)
        self.restoration_recieved = restoration_recieved
        self.allowence_recieved = False
        self.allowence_cheques_sum = 0
        self.mps = 0
        self.daai = 0

    @property
    def energy_cost(self):
        """Calculate and return the current energy cost for the agent."""
        return self.model.energy_price * (self.dwelling)
    
    # Add a new method to update MPS
    def update_mps(self):
        # Get all agent incomes
        incomes = [a.disposable_income for a in self.model.schedule.agents]

        # Calculate the quintile thresholds
        thresholds = np.quantile(incomes, [0.2, 0.4, 0.6, 0.8])

        # Determine which quintile the agent's income belongs to
        quintile = np.digitize(self.disposable_income, thresholds)

        # Set the MPS based on the quintile
        self.mps = Country.MPS_VALUES[quintile]
        
    def step(self):
        """Update the agent's savings in a step."""
        # Calculate energy costs
        energy_costs = self.energy_cost

        self.update_mps()

        # Check if the agent is in arrears
        if energy_costs > self.disposable_income * Country.ARREARS_TRESHOLD:  # now considering arrears if energy cost exceeds 30% of disposable income
            self.arrears += energy_costs
        else:
            self.savings += (self.disposable_income - energy_costs) * self.mps

        # If the agent has arrears
        if self.arrears > 0:
            if self.savings >= self.arrears:  # and if savings can cover it
                self.savings -= self.arrears
                self.arrears = 0
            else:  # if savings cannot cover it
                self.arrears = self.arrears - self.savings  # set arrears to remaining amount
                self.savings = 0

        self.inability = self.energy_cost > self.disposable_income * Country.INABILITY_TRESHOLD  # agent is in inability state if energy costs exceed 10% of income

        # Update disposable income based on growth rate
        growth_rate = np.random.uniform(self.model.growth_boundaries[0], self.model.growth_boundaries[1])
        self.disposable_income *= (1 + growth_rate)


class Country(Model):

    INABILITY_TRESHOLD = 0.1
    ARREARS_TRESHOLD = 0.75
    MPS_VALUES = [0.1, 0.13, 0.17, 0.20, 0.25] # Marginal Propensity to Save
    
    @classmethod
    def calculate_gini(cls, incomes):
        incomes = np.sort(incomes)
        n = len(incomes)
        index = np.arange(1, n + 1)
        return ((np.sum((2 * index - n  - 1) * incomes)) / (n * np.sum(incomes)))

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
    
    def restoration_program(self):

        # Sort the agents by income in ascending order
        sorted_agents = sorted(self.schedule.agents, key=lambda agent: (agent.disposable_income, -agent.dwelling))

        for agent in sorted_agents:
            if self.RESTORATION_BUDGET >= self.RESTORATION_COST:
                # Assign the restoration
                agent.restoration_recieved = True

                agent.dwelling *= (1- self.RESTORATION_DWELLING_REDUCTION)

                # Deduct the restoration cost from the budget
                self.RESTORATION_BUDGET -= RESTORATION_COST
            else:
                # If the budget is not sufficient for another restoration, break the loop
                break

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
                    agent.disposable_income += self.allowence_cheque
                    agent.daai = agent.disposable_income - self.allowence_cheque 

                    # Deduct the restoration cost from the budget
                    self.allowence_budget -= self.allowence_cheque
                else:
                    # If the budget is not sufficient for another restoration, break the loop
                    agent.daai = agent.disposable_income
                    break

    @staticmethod
    def assign_inability(agents, inability_start):
        """
        Assigns the inability to the agents.
        """
        agents.sort(key=lambda x: x.disposable_income)

        num_agents = len(agents)
        num_unable = int(inability_start * num_agents)
        inability_per_quintile = np.array([2, 1.6, 1.2, 0.8, 0])
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
        """
        Assigns the arrears to the agents.
        """
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
                agent.arrears = 2 * Country.ARREARS_TRESHOLD * agent.disposable_income  # assign 0.6 * disposable_income
            assigned_arrears += num_arrears_quintile

        # Assign remaining arrears if any
        i = 0
        while assigned_arrears < num_arrears:
            start = i * num_agents // 5
            end = (i + 1) * num_agents // 5 if i < 4 else num_agents
            quintile = [agent for agent in agents[start:end] if not agent.arrears > 0]
            if quintile:
                agent = np.random.choice(quintile)
                agent.arrears = Country.ARREARS_TRESHOLD * agent.disposable_income  # assign 0.6 * disposable_income
                assigned_arrears += 1
            else:
                i += 1


    @staticmethod
    def assign_dwelling(agents, energy_price):
        """
        Assigns the dwelling to the agents.
        """
        buffer = 0.02  # 2% buffer or margin
        all_incomes = [agent.disposable_income for agent in agents]
        quintile_thresholds = np.quantile(all_incomes, [0.2, 0.4])  # Calculate income thresholds for the bottom two quintiles
        for agent in agents:
            min_dwelling, max_dwelling = 500, 2000
            if agent.inability:
                min_dwelling = max(min_dwelling, int(np.ceil((agent.disposable_income * (Country.INABILITY_TRESHOLD + buffer)) / energy_price)))
            else:
                max_dwelling = min(max_dwelling, int((agent.disposable_income * (Country.INABILITY_TRESHOLD - buffer)) / energy_price))

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

    def __init__(self, N, median_income, min_disposal, gini_target, inability_target, arrears_target,
                growth_boundaries, prices, shares_p, growth_rate_lower_bound, growth_rate_upper_bound,
                restoration_ACTIVE = False, allowence_ACTIVE = False, price_shock = False, 
                COST_INSULATION = 0.45,
                TRESHOLD_LEVEL = 0.25,
                AVERAGE_HH_SIZE = 1122,
                DWELLING_RESTORATION_LIMIT = 900,
                RESTORATION_DWELLING_REDUCTION = 0.5,
                RESTORATION_STEP = 7,
                SHOCK_INDEX = 0,
                SHOCK_MAGNITUDE = 0.5,
                SHOCK_STEP = 5,
                ALLOWENCE_CHEQUE = 50,
                RESTORATION_BUDGET = 12000,
                ALLOWENCE_BUDGET = 50000,
                ALLOWENCE_FROM = 9):
        self.num_agents = N
        self.median_income = median_income
        self.min_disposal = min_disposal
        self.gini_target = gini_target
        self.inability_target = inability_target
        self.growth_boundaries = growth_boundaries
        self.prices = prices
        self.shares_p = shares_p
        self.growth_rate_lower_bound = growth_rate_lower_bound
        self.growth_rate_upper_bound = growth_rate_upper_bound
        self.price_shock = price_shock
        self.restoration_ACTIVE = restoration_ACTIVE
        self.allowence_ACTIVE = allowence_ACTIVE
        self.restoration_budget = RESTORATION_BUDGET
        self.allowence_budget = ALLOWENCE_BUDGET
        self.allowence_cheque = ALLOWENCE_CHEQUE
        self.arrears_target = arrears_target
        self.COST_INSULATION = COST_INSULATION
        self.TRESHOLD_LEVEL =  TRESHOLD_LEVEL 
        self.RESTORATION_COST = self.COST_INSULATION * AVERAGE_HH_SIZE
        self.DWELLING_RESTORATION_LIMIT = DWELLING_RESTORATION_LIMIT
        self.RESTORATION_DWELLING_REDUCTION = RESTORATION_DWELLING_REDUCTION
        self.RESTORATION_STEP = RESTORATION_STEP
        self.SHOCK_INDEX = SHOCK_INDEX
        self.SHOCK_MAGNITUDE = SHOCK_MAGNITUDE
        self.SHOCK_STEP = SHOCK_STEP
        self.ALLOWENCE_FROM = ALLOWENCE_FROM
        
        # Other initialization code...
        self.schedule = RandomActivation(self)
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
                        "Inability Over Time": lambda model: sum(agent.inability for agent in model.schedule.agents) / model.num_agents,
                        "Arrears Over Time": lambda model: sum(agent.arrears for agent in model.schedule.agents) / model.num_agents,
                        "Allowence Overt Time": lambda model: sum(agent.allowence_cheques_sum for agent in model.schedule.agents) / model.num_agents,
                        "Energy Price": lambda model: model.energy_price,
                        "Price Yellow Fuel": lambda model: model.prices[0],
                        "Price Brown Fuel": lambda model: model.prices[1],
                        "Average Dwelling Over Time": lambda model: sum(agent.dwelling for agent in model.schedule.agents) / model.num_agents,
                        "Restorations": lambda m: sum([agent.restoration_recieved for agent in m.schedule.agents])}
                        )

        # Generate disposable incomes and assign dwelling and technology
        incomes = self.generate_income_distribution(self.num_agents, self.median_income, self.gini_target, self.min_disposal)

        self.initial_energy_price = np.dot(self.prices, self.shares_p)

 
        # Create agents
        agents = []
        for i in range(self.num_agents):
            a = Household(i, self, disposable_income=incomes[i], dwelling=0, inability=False)
            agents.append(a)

        
        self.assign_inability(agents, self.inability_target)
        self.assign_dwelling(agents, self.energy_price)
        self.assign_arrears(agents, self.arrears_target)
        self.assign_mps(agents)

        # Add agents to the model
        for a in agents:
            self.schedule.add(a)

    @staticmethod
    def assign_mps(agents):
        """
        Assigns the MPS to the agents based on their income quintile.
        """
        agents.sort(key=lambda x: x.disposable_income)

        num_agents = len(agents)
        for i in range(5):  # For each quintile
            start = i * num_agents // 5
            end = (i + 1) * num_agents // 5 if i < 4 else num_agents
            quintile_agents = agents[start:end]
            for agent in quintile_agents:
                agent.mps = Country.MPS_VALUES[i]

    @property
    def energy_price(self):
        """Calculate and return the current energy price."""
        return np.dot(self.prices, self.shares_p)

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)  # Collect data before updating the prices

        # Update prices based on unique growth rates
        growth_rates = np.random.uniform(self.growth_rate_lower_bound, self.growth_rate_upper_bound)

        self.prices = self.prices * (1 + growth_rates)

        if self.schedule.steps == self.SHOCK_STEP and self.price_shock == True:
            self.prices[self.SHOCK_INDEX] *= (1 + self.SHOCK_MAGNITUDE)

        # Execute the restoration program if it is enabled
        if self.schedule.steps == self.RESTORATION_STEP and self.restoration_ACTIVE:
            self.restoration_program()

        if self.schedule.steps >= self.ALLOWENCE_FROM and self.allowence_ACTIVE:
            self.allowence_program()

        self.schedule.step()


def plot_model_data(model_df):
    # Model-level data
    n_cols = 2
    n_rows = math.ceil(len(model_df.columns) / n_cols)
    fig = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=model_df.columns)
    color_discrete_sequence=px.colors.qualitative.Plotly  # Default color sequence

    for i, col in enumerate(model_df.columns, start=1):
        fig.add_trace(go.Scatter(x=model_df.index, y=model_df[col], mode='lines', name=col, line=dict(color=color_discrete_sequence[(i-1) % len(color_discrete_sequence)])), row=((i-1)//n_cols)+1, col=((i-1)%n_cols)+1)

    fig.update_layout(height=400*n_rows, width=1200,showlegend=False)
    
    return fig

def plot_prices(model_df):
    # For Price1, Price2, and Energy Price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=model_df.index, y=model_df['Price Yellow Fuel'], mode='lines', name='Price Yellow Fuel', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=model_df.index, y=model_df['Price Brown Fuel'], mode='lines', name='Price Brown Fuel', line=dict(color='brown')))
    fig.add_trace(go.Scatter(x=model_df.index, y=model_df['Energy Price'], mode='lines', name='Energy Price', line=dict(color='green')))

    fig.update_layout(title='Energy Prices', xaxis_title='Time (steps)', yaxis_title='Price', showlegend=False)
    
    return fig


# Page: Model
def model_page():
    st.title("The Model")

    # Define the input parameters using Streamlit widgets in two columns

    set1, set2 = st.columns(2)
    st.markdown('---')

    st.header('Agents')

    # Set up Streamlit widgets
    st.caption("Number of agent used for simulation - small and large values are unrepresentative or computationally intensive respectivelly")
    N = st.slider('Number of agents', 100, 3000, 1000)
    median_income = st.slider('Median income', 500, 5000, 1700)
    min_disposal = st.slider('Minimum disposal', 100, 2000, 900)
    gini_target = st.slider("Gini Target", 0.1, 0.9, 0.30, format="%0.2f")
    inability_target = st.slider("Inability Target", 0.1, 1.0, 0.2, format="%0.2f")
    arrears_target = st.slider("Arrears Target", 0.01, 1.0, 0.07, format="%0.2f")

    st.markdown('---')
    st.header('Country')

    growth_boundaries = [st.slider("Growth boundary lower", 0.0, 1.0, 0.0, format="%0.2f"), st.slider("Growth boundary upper", 0.0, 1.0, 0.005, format="%0.2f")]
    prices = np.array([st.slider("Price of Yellow Fuel", 0.01, 0.5, 0.2, format="%0.2f"), st.slider("Price of Brown Fuel", 0.01, 0.2, 0.06, format="%0.2f")])
    shares_yellow = st.slider("Share of Yellow Fuel", 0.1, 1.0, 0.7, format="%0.2f")
    shares_p = np.array([shares_yellow, 1 - shares_yellow])
    growth_rate_lower_bound = np.array([st.slider("Growth rate lower bound for Yellow Fuel", -0.05, 0.0, -0.01, format="%0.2f"), st.slider("Growth rate lower bound for Brown Fuel", -0.05, 0.0, -0.01, format="%0.2f")])
    growth_rate_upper_bound = np.array([st.slider("Growth rate upper bound for Yellow Fuel", 0.0, 0.05, 0.02, format="%0.2f"), st.slider("Growth rate upper bound for Brown Fuel", 0.0, 0.05, 0.01, format="%0.2f")])
    

    st.markdown('---')
    st.subheader('Policies')

    st.markdown('---')
    st.caption('Allowence program')
    st.caption("Allowence program introduces allowences to agents. Each agent that is experiencing inability will get allowence each step, until the depleption of the budget")
    
    
    allowence_ACTIVE = st.checkbox("Allowence program", value=False)
    budget_allowence = st.number_input("Allowence program budget")

    restoration_ACTIVE = st.checkbox("Restoration program", value=False)
    price_shock = st.checkbox("Price shock", value=False)

    st.markdown('---')    
    
    st.header('Simulation tenure')

    periods = st.slider('Number of steps to simulate', 1, 100, 7)

    st.caption("It takes from 30 second up to 2 minutes to intialize and run the code. You can check that model is running calculation in top right corner.")

    #if st.button("Run Model"):
    model = Country(N, median_income, min_disposal, gini_target, inability_target, arrears_target, growth_boundaries, prices, shares_p, growth_rate_lower_bound, growth_rate_upper_bound,restoration_ACTIVE, allowence_ACTIVE, price_shock)
    for i in range(periods):
        model.step()


    agent_data = model.datacollector.get_agent_vars_dataframe()
    model_data = model.datacollector.get_model_vars_dataframe()

    # Plot the data using Matplotlib
    st.title('Model Outputs')

    model_plot = plot_model_data(model_data)
    st.plotly_chart(model_plot)

    prices_plot = plot_prices(model_data)
    st.plotly_chart(prices_plot)

