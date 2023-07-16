import streamlit as st
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Citizen(Agent): 
    def __init__(self, unique_id, model, wealth,x,y):
        super().__init__(unique_id, model)
        self.wealth = wealth  
        self.x = x
        self.y = y

    def step(self):
        self.wealth -= self.model.fuel_price * self.model.fuel_units

class World(Model):  
    def __init__(self, N, fuel_price, poor_percentage, fuel_units):
        self.num_agents = N  
        self.fuel_price = fuel_price
        self.fuel_units = fuel_units
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"Energy Poverty": lambda m: self.calculate_poverty()},
            agent_reporters={"Wealth": lambda a: a.wealth}
        )

        poor_threshold = self.num_agents * poor_percentage
        for i in range(self.num_agents):
            x = i%100
            y = i // 100
            if i < poor_threshold:  
                a = Citizen(i, self, 100, x, y)  
            else:
                a = Citizen(i, self, 10000, x, y)  
            self.schedule.add(a)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def calculate_poverty(self):
        count = 0
        for agent in self.schedule.agents:
            if agent.wealth < self.fuel_price * self.fuel_units:  
                count += 1
        return count / self.num_agents
    
def plot_heatmap(year):
    agent_data = model.datacollector.get_agent_vars_dataframe()
    data = agent_data.xs(year, level="Step")["Wealth"]
    # Create a 100x100 array filled with zeros
    heatmap_data = np.zeros((100, 100))
    for i in range(model.num_agents):
        # For each agent, set the corresponding cell in the array to their wealth
        agent_wealth = data[i]
        x = model.schedule.agents[i].x
        y = model.schedule.agents[i].y
        heatmap_data[y][x] = agent_wealth
    # Normalize wealth values to range from 0 to 1 for better coloring
    heatmap_data = (heatmap_data - np.min(heatmap_data)) / (np.max(heatmap_data) - np.min(heatmap_data))
    # Plot the heatmap
    fig, ax = plt.subplots(figsize = (1,1))
    sns.heatmap(heatmap_data, cmap='coolwarm', ax=ax)
    st.pyplot(fig)


##########################
### STREAMLIT SETTINGS ###

# Set the width of the layout
st.set_page_config(layout="wide")
st.title('energy poverty and technological progress ABM')

# User inputs
N = st.slider('Number of citizens', min_value=100, max_value=10000, value=1000, step=100)
fuel_price = st.slider('Fuel price', min_value=1, max_value=100, value=50, step=1)
poor_percentage = st.slider('Percentage of poor citizens', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
fuel_units = st.slider('Fuel units', min_value=1, max_value=100, value=50, step=1)

model = World(N, fuel_price, poor_percentage, fuel_units)

for i in range(10):  
    model.step()

# Collect the data into a pandas DataFrame
data = model.datacollector.get_model_vars_dataframe()

# Plot the data
st.write('Energy Poverty Level over Time')
st.line_chart(data['Energy Poverty'])

# Interactive histogram
year = st.slider('Year', min_value=0, max_value=10, step=1, value=0)
agent_data = model.datacollector.get_agent_vars_dataframe()
data = agent_data.xs(year, level="Step")["Wealth"]

fig, ax = plt.subplots(figsize = (5,5))
plt.hist(data, bins=range(-1000, 11000, 500))
plt.xlabel('Wealth')
plt.ylabel('Number of Citizens')
plt.title('Wealth Distribution in Year {}'.format(year))
plt.grid(True)
st.pyplot(fig)

#Heatmap

plot_heatmap(year)
