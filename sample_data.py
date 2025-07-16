import random

def add_noise(linear_values, noise_level):
    return [y + random.uniform(-noise_level, noise_level) for y in linear_values]

advertising_spend_vs_sales = [
    [i for i in range(1, 31)],
    add_noise([3*i + 10 + (i*0.3) for i in range(1, 31)], 2),
    "Advertising Spend",
    "Units Sold",
    "Thousands of Dollars",
    "Units"
]

hours_studied_vs_test_scores = [
    [i for i in range(1, 31)],
    add_noise([5*i + 40 + (i*0.2) for i in range(1, 31)], 3),
    "Hours Studied",
    "Test Score",
    "Hours",
    "Points"
]

temperature_increase_over_time = [
    [i for i in range(1, 26)],
    add_noise([0.5*i + 15 + (i*0.1) for i in range(1, 26)], 0.5),
    "Time",
    "Temperature",
    "Years",
    "Degrees Celsius"
]

engine_size_vs_fuel_consumption = [
    [i for i in range(1000, 3250, 100)],
    add_noise([0.02*i + 5 for i in range(1000, 3250, 100)], 0.3),
    "Engine Size",
    "Fuel Consumption",
    "CC",
    "Litres per 100km"
]

years_vs_company_revenue = [
    [i for i in range(2000, 2025)],
    add_noise([2*(i-2000) + 50 + (i%3) for i in range(2000, 2025)], 2),
    "Year",
    "Revenue",
    "Year",
    "Millions USD"
]

production_volume_vs_cost = [
    [i for i in range(50, 1050, 50)],
    add_noise([0.8*i + 100 for i in range(50, 1050, 50)], 5),
    "Production Volume",
    "Production Cost",
    "Units",
    "Thousands USD"
]

months_vs_subscribers = [
    [i for i in range(1, 31)],
    add_noise([200*i + (i*5) for i in range(1, 31)], 20),
    "Month",
    "Subscribers",
    "Months",
    "Subscribers"
]

practice_time_vs_skill_level = [
    [i for i in range(1, 31)],
    add_noise([4*i + 20 + (i*0.4) for i in range(1, 31)], 3),
    "Practice Time",
    "Skill Level",
    "Hours",
    "Score"
]

days_vs_website_visits = [
    [i for i in range(1, 31)],
    add_noise([50*i + (i*1.5) for i in range(1, 31)], 10),
    "Day",
    "Website Visits",
    "Days",
    "Visits"
]

temperature_vs_electricity_usage = [
    [i for i in range(10, 40)],
    add_noise([25*i + 100 + (i*0.5) for i in range(10, 40)], 15),
    "Temperature",
    "Electricity Usage",
    "Degrees Celsius",
    "kWh"
]
