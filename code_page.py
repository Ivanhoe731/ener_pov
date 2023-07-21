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

    """
    return source_code