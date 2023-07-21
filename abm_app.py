import streamlit as st
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

from mesa.space import SingleGrid
from mesa.visualization.modules import CanvasGrid


from IPython.display import clear_output
import scipy.stats as stats
import random
import ipywidgets as widgets
from numpy.random import choice
from numpy import exp



## HELPER FUNCTIONS 

class GiniCalculationError(Exception):
    pass

def calculate_gini(incomes):
    # Calculate Gini coefficient for a list of incomes
    incomes = np.sort(incomes)
    n = len(incomes)
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n  - 1) * incomes)) / (n * np.sum(incomes)))

def generate_income_distribution(num_people, median_income, gini_target):
    alpha = (gini_target + 1) / (2 - gini_target)  # Set initial alpha
    for _ in range(10000):  # Limit the number of iterations
        incomes = stats.gamma.rvs(alpha, scale=median_income/alpha, size=num_people)  # Generate a random income distribution
        gini_current = calculate_gini(incomes)  # Calculate the current Gini coefficient
        if np.isclose(gini_current, gini_target, atol=0.01):  # Check if the current Gini coefficient is close to the target
            return incomes
        elif gini_current < gini_target:  # If the current Gini coefficient is too low, decrease alpha
            alpha *= 0.9
        else:  # If the current Gini coefficient is too high, increase alpha
            alpha *= 1.1

    # If we've reached this point, the desired Gini coefficient was not reached
    error_message = f"Failed to reach target Gini coefficient in 1000 iterations. Current Gini: {gini_current}"
    raise GiniCalculationError(error_message)

def calculate_wealth(incomes, periods):
    # Initialize lists to hold the wealth and Gini coefficients at each time step
    wealth_over_time = []
    gini_over_time = []
    wealth_distribution_over_time = []
    
    # Calculate the wealth for each time period
    for _ in range(periods):
        # Calculate the wealth by summing the incomes
        wealth = np.sum(incomes)
        
        # Add the wealth to the list
        wealth_over_time.append(wealth)
        
        # Calculate the Gini coefficient for the current income distribution
        gini = calculate_gini(incomes)
        
        # Add the Gini coefficient to the list
        gini_over_time.append(gini)
        
        # Calculate the wealth distribution for the current income distribution
        wealth_distribution = calculate_wealth_distribution(incomes)
        
        # Add the wealth distribution to the list
        wealth_distribution_over_time.append(wealth_distribution)
        
        # Set a random growth rate for the current period
        growth_rate = np.random.uniform(gr[0], gr[1])
        
        # Increase the incomes by the growth rate
        incomes *= (1 + growth_rate)
    
    return wealth_over_time, gini_over_time, wealth_distribution_over_time


def calculate_wealth_distribution(incomes, num_bins=10):
    # Calculate the total wealth
    total_wealth = np.sum(incomes)
    
    # Calculate the histogram of wealth
    hist, bin_edges = np.histogram(incomes, bins=num_bins)
    
    # Calculate the wealth in each bin
    bin_wealths = [(bin_edges[i+1] - bin_edges[i]) * hist[i] for i in range(num_bins)]
    
    # Normalize the wealths by total wealth to get wealth distribution
    wealth_distribution = bin_wealths / total_wealth
    
    return wealth_distribution

### MODEL

upgrade_cost_factor = 100


class Person(Agent):
    def __init__(self, unique_id, model, initial_income, inability = False, arrears = False, dwellings = 25, technology = 25, upgrade_cost_factor=100): ## THE HARDCODED VALUES SHOULD BE CHANGED LATER
        super().__init__(unique_id, model)
        self.disposable_income = initial_income
        self.wealth = 0  # The wealth is now separate from income
        self.energy_set = "bad"  # Assume everyone starts with a 'bad' energy set
        self.inability = inability
        self.dwellings = dwellings  # Assume everyone starts with low quality dwelling
        self.technology = technology  # Assume everyone starts with low quality technology
        self.arrears = arrears  # Assume no one has arrears to start
        self.energy_cost = 0  # Will be calculated each step based on energy_set and quality

    def minimal_energy_consumption(self):
        # Define how dwelling and technology quality affect energy consumption
        return self.dwellings * self.technology / 10000  # Replace this formula as needed

    def step(self):
        # Check if the person can afford to upgrade dwelling or technology
        dwelling_upgrade_cost = upgrade_cost_factor * exp(self.dwellings / 100)
        technology_upgrade_cost = upgrade_cost_factor * exp(self.technology / 100)

        # Determine which upgrade is more beneficial (reduces energy cost more)
        dwelling_savings = self.energy_cost * (self.dwellings / 100)
        technology_savings = self.energy_cost * (self.technology / 100)

        if dwelling_savings > dwelling_upgrade_cost and dwelling_savings > technology_savings:
            # If potential savings from upgrading dwelling are greater than the cost and more than from upgrading technology, upgrade dwelling
            self.dwellings += 1
            self.wealth -= dwelling_upgrade_cost  # Pay the cost to upgrade
        elif technology_savings > technology_upgrade_cost:
            # If potential savings from upgrading technology are greater than the cost, upgrade technology
            self.technology += 1
            self.wealth -= technology_upgrade_cost  # Pay the cost to upgrade

        # Check if the person can afford to switch to 'good' energy set
        if self.energy_set == "bad" and self.wealth > self.model.switch_cost:
            # Calculate potential savings from switching
            potential_savings = self.wealth * self.minimal_energy_consumption() * (self.model.bad_energy_cost_factor - self.model.good_energy_cost_factor) * self.model.switch_cost_payback
            if potential_savings > self.model.switch_cost:
                # If potential savings are greater than the switch cost, switch to 'good' energy set
                self.energy_set = "good"
                self.wealth -= self.model.switch_cost  # Pay the cost to switch

        # Calculate energy cost based on energy set, minimal energy consumption and renewable energy share
        cost_factor = self.model.bad_energy_cost_factor if self.energy_set == "bad" else self.model.good_energy_cost_factor
        renewable_factor = 1 / (1 + self.model.renewable_share)
        self.energy_cost = self.wealth * self.minimal_energy_consumption() * (1 - self.model.tech_progress) * cost_factor * renewable_factor

        # Subtract energy cost from disposable income
        self.disposable_income -= self.energy_cost

        # Add remaining disposable income to wealth
        self.wealth += self.disposable_income

        # Check if the person falls into energy poverty (if energy cost >= 2 * median energy cost of all agents)
        median_energy_cost = np.median([a.energy_cost for a in self.model.schedule.agents])
        if self.energy_cost >= 2 * median_energy_cost:
            # Take some action if the person falls into energy poverty
            pass  # Placeholder for energy poverty action

class Economy(Model):
    def __init__(self, num_people, median_income, gini_target, tech_progress_rate, renewable_share, switch_cost, switch_cost_payback,bad_energy_cost_factor, good_energy_cost_factor,growth_bounds, rd_share, rd_min):
        self.num_people = num_people
        self.tech_progress = 0  # Start with no technological progress
        self.tech_progress_rate = tech_progress_rate  # Rate at which technological progress occurs each step
        self.renewable_share = renewable_share  # Proportion of energy that comes from renewable sources
        self.switch_cost = switch_cost  # Cost for an agent to switch to the 'good' energy set
        self.switch_cost_payback = switch_cost_payback  # How many steps it takes for the switch cost to pay back in savings
        self.bad_energy_cost_factor = bad_energy_cost_factor # dont forhet to comment this
        self.good_energy_cost_factor = good_energy_cost_factor # dont forget to comment this
        self.growth_bounds = growth_bounds  # Bounds for random economic growth
        self.economic_growth = 0
        self.rd_share = rd_share  # Share of economic growth allocated to R&D
        self.rd_min = rd_min # research and development investments 
        self.schedule = RandomActivation(self)
        
        incomes = generate_income_distribution(num_people, median_income, gini_target)
        sorted_incomes = np.sort(incomes)


        # Define lower, middle and high income groups
        lower_income_threshold = np.percentile(sorted_incomes, 20)  # Bottom 20%
        middle_income_threshold = np.percentile(sorted_incomes, 50)  # Up to 50%
        high_income_threshold = np.percentile(sorted_incomes, 80)  # Up to 80%


        lower_income_group = sorted_incomes[sorted_incomes < lower_income_threshold]
        middle_income_group = sorted_incomes[(sorted_incomes >= lower_income_threshold) & (sorted_incomes < middle_income_threshold)]
        high_income_group = sorted_incomes[(sorted_incomes >= middle_income_threshold) & (sorted_incomes < high_income_threshold)]
        

        # Combine lower, middle and high income groups, with lower income group having more weight
        combined_group = np.concatenate([lower_income_group, middle_income_group, high_income_group])
        weights = [0.75 if income < lower_income_threshold else 0.5 if income < middle_income_threshold else 0.25 for income in combined_group]


        # Normalize the weights
        weights = weights / np.sum(weights)

        # Randomly assign inability, arrears, and dwellings based on weights
        inability_group = choice(combined_group, size=int(num_people * 0.15), replace=False, p=weights)  # 15% of agents with inability
        arrears_group = choice(combined_group, size=int(num_people * 0.07), replace=False, p=weights)  # 7% of agents with arrears
        # Randomly assign bad dwellings and bad technologies based on weights
        bad_dwellings_group = choice(combined_group, size=int(num_people * 0.07), replace=False, p=weights)  # 7% of agents with bad dwellings
        bad_technologies_group = choice(combined_group, size=int(num_people * 0.07), replace=False, p=weights)  # 7% of agents with bad technologies


        for i in range(num_people):
            inability = incomes[i] in inability_group
            arrears = incomes[i] in arrears_group
            # Now we determine the quality score for dwelling and technology based on income percentile
            income_percentile = stats.percentileofscore(sorted_incomes, incomes[i])

            # Agents in bad dwellings or with bad technologies start with a score of 25
            # Others start with a score of 25 + income percentile * proportionality parameter
            dwellings = 25 if incomes[i] in bad_dwellings_group else 25 + income_percentile * 0.5  # Proportionality parameter for dwellings is 0.5
            technology = 25 if incomes[i] in bad_technologies_group else 25 + income_percentile * 0.75  # Proportionality parameter for technology is 0.75


            a = Person(i, self, incomes[i], inability, arrears, dwellings, technology,upgrade_cost_factor)
            self.schedule.add(a)

        
        self.datacollector = DataCollector(
            model_reporters={
                "Total Wealth": lambda m: sum([agent.wealth for agent in m.schedule.agents]), 
                "Gini": lambda m: calculate_gini([agent.wealth for agent in m.schedule.agents]),
                "Wealth Distribution": lambda m: calculate_wealth_distribution([agent.wealth for agent in m.schedule.agents]),
                "2M indicator": lambda m: len([agent for agent in m.schedule.agents if agent.energy_cost >= 2 * np.median([a.energy_cost for a in m.schedule.agents])]),
                "Energy Poverty": lambda m: (sum([0.5*agent.inability + 0.25*(agent.dwellings=="bad") + 0.25*agent.arrears for agent in m.schedule.agents]) / m.num_people) * 100,
                "Tech Progress": lambda m: m.tech_progress,
                "Growth": lambda m: m.economic_growth,
            },
            agent_reporters={
                "Wealth": lambda a: a.wealth,
                "Energy Cost": lambda a: a.energy_cost
            }
        )
    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)

        # Determine the economic growth for this step
        self.economic_growth = np.random.uniform(self.growth_bounds[0], self.growth_bounds[1])

        # Update disposable income of the economy
        for agent in self.schedule.agents:
            agent.disposable_income *= (1 + self.economic_growth / 100)

        # Allocate a portion of the growth to R&D, and add it to technological progress
        rd_investment = max(self.economic_growth * self.rd_share, self.rd_min)
        self.tech_progress += rd_investment

        self.schedule.step()




# Page: Model
def model_page():
    st.title("The Model")
    # Import and run your ABM model code here

    # Define the input parameters using Streamlit widgets in two columns

    set1, set2 = st.columns(2)

    with set1:
        periods = st.slider("Number of periods to simulate",1,20,5,format="%d")
        num_people = st.slider("Number of People", 100, 2000, 1000, format="%d")
        median_income = st.slider("Median Income", 10000, 50000, 18000, format="%d")
        gini_target = st.slider("Gini Target", 0.1, 1.0, 0.25, format="%0.2f")
        tech_progress_rate = st.slider("Tech Progress Rate", 0.001, 0.2, 0.01, format="%0.3f")
        renewable_share = st.slider("Renewable Share", 0.1, 0.9, 0.2, format="%0.2f")
        switch_cost = st.slider("Switch Cost", 50, 200, 100, format="%d")

    with set2:
        switch_cost_payback = st.slider("Switch Cost Payback (Steps)", 1, 10, 2, format="%d")
        bad_energy_cost_factor = st.slider("Bad Energy Cost Factor", 1, 5, 2, format="%d")
        good_energy_cost_factor = st.slider("Good Energy Cost Factor", 1, 5, 1, format="%d")
        growth_bounds = st.slider("Growth Bounds", -0.5, 0.5, (0.05, 0.1), format="%0.2f")
        rd_share = st.slider("R&D Share", 0.01, 0.1, 0.03, format="%0.3f")
        rd_min = 0.01




    if st.button("Run Model"):

            # Run the model based on the input parameters
        model = Economy(num_people, median_income, gini_target, tech_progress_rate, renewable_share, switch_cost, switch_cost_payback, bad_energy_cost_factor, good_energy_cost_factor, growth_bounds, rd_share, rd_min)

        for i in range(periods):
            model.step()

        # Access the collected data
        model_df = model.datacollector.get_model_vars_dataframe()
    # Access the collected data
        model_df = model.datacollector.get_model_vars_dataframe()

        # Plot the data using Matplotlib
        st.subheader("Model Outputs")
        fig, axs = plt.subplots(6, 1, figsize=(10, 15))

        axs[0].plot(model_df["Total Wealth"])
        axs[0].set_title("Total Wealth Over Time")

        axs[1].plot(model_df["Gini"])
        axs[1].set_title("Gini Coefficient Over Time")

        axs[2].plot(model_df["2M indicator"])
        axs[2].set_title("2M Indicator Over Time")

        axs[3].plot(model_df["Energy Poverty"])
        axs[3].set_title("Energy Poverty Over Time")

        axs[5].plot(model_df["Growth"])
        axs[5].set_title("Economic Growth Over Time")

        # Adjust spacing and display the plot
        plt.tight_layout()
        st.pyplot(fig)
