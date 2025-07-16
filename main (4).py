# Imports
import matplotlib.pyplot as plt
import numpy as np
import time, random
from sample_data import *

# Sample X && Y data && Dataset Preset
x_val = []
y_val = []
dataset_names = [
    "advertising_spend_vs_sales",
    "hours_studied_vs_test_scores",
    "temperature_increase_over_time",
    "engine_size_vs_fuel_consumption",
    "years_vs_company_revenue",
    "production_volume_vs_cost",
    "months_vs_subscribers",
    "practice_time_vs_skill_level",
    "days_vs_website_visits",
    "temperature_vs_electricity_usage"
]

# Average of X
def averageX():
    total = 0
    for x in x_val:
        total = total + x

    return total/len(x_val)

# Average of Y
def averageY():
    total = 0
    for y in y_val:
        total = total + y

    return total/len(y_val)

# Work out gradient m=∑(x[i]​−average(x))2∑(x[i]​−average(x))(y[i]​−average(y)​)​ 
def calculateLinearSlope(averageX, averageY):
    totalSigmaNotation = 0
    totalSigmaNotation2 = 0

    for index in range(len(x_val)):
        totalSigmaNotation = totalSigmaNotation + (x_val[index] - averageX)*(y_val[index] - averageY)
        totalSigmaNotation2 = totalSigmaNotation2 + (x_val[index] - averageX) ** 2

    return totalSigmaNotation / totalSigmaNotation2

# Return the equation in integer
def returnEquation():
    m = calculateLinearSlope(averageX(), averageY())
    c = averageY() - (m * averageX())
    return f"y = {m}x + {c}"

# Return the equation in indivisual form
def returnIndivisualNumbersOfEquation():
    m = calculateLinearSlope(averageX(), averageY())
    c = averageY() - (m * averageX())
    return [m, c]

# Used to predict value using y=mx+c
def predictValue(valType, val):
    averageXVal = averageX()
    averageYVal = averageY()
    m = calculateLinearSlope(averageXVal, averageYVal) # Gradient
    c = averageYVal - (m * averageXVal) # Y intercept

    if valType == "x":
         return (val-c) / m
    else:
        return (m*val) + c

# Error Calculation (based on actual points)
def calculateAverageErr(labelledXValue, labelledYValue, labelledXValueUnit, labelledYValueUnit):
    totalErrorMarginX = 0
    totalErrorMarginY = 0

    # Loop through all paired data points
    for i in range(len(x_val)):
        predicted_age = predictValue("x", y_val[i]) 
        error_x = abs(x_val[i] - predicted_age)   
        totalErrorMarginX += error_x

        predicted_weight = predictValue("y", x_val[i])
        error_y = abs(y_val[i] - predicted_weight) 
        totalErrorMarginY += error_y

    avg_error_x = totalErrorMarginX / len(x_val)
    avg_error_y = totalErrorMarginY / len(y_val)
    print("\nError Margin {} [X]: {:.2f} {}".format(labelledXValue, avg_error_x, labelledXValueUnit))
    print("Error Margin {} [Y]: {:.2f} {}\n".format(labelledYValue, avg_error_y, labelledYValueUnit))

# Main program && Added graph
def mainProgramInputPrediction():
    global x_val, y_val
    print("Welcome to DataLabs Beta, user. This program is for data scientists, programmers, mathemeticians.")
    print("New feature: liner regression algorithim, (aka line of best fit on computer) to predict x/y value.")
    print("Warning: Values may be unexpected, as this is soley a line of best fit.")
    print("Credits: Noah\n")
    
    # Code will be used soon, for when I add import data
    """
    labelledXValue = input("Label X Axis >> ")
    labelledXValueUnit = input("Label X Unit >> ")
    labelledYValue = input("\nLabel Y Axis >> ")
    labelledYValueUnit = input("Label Y Unit >> ")
    title = input("\nEnter title of data >> ")
    """
    
    labelledXValue = ""
    labelledXValueUnit = ""
    labelledYValue = ""
    labelledYValueUnit = ""

    while True:
        print("\nHere are some real world data sets, to choose from, and get a prediction model: \n")
        for name in dataset_names:
            print(name)

        dataSetChoice = input("\nOption >> ")
        
        if dataSetChoice == "advertising_spend_vs_sales":
            x_val = advertising_spend_vs_sales[0]
            y_val = advertising_spend_vs_sales[1]
            labelledXValue = advertising_spend_vs_sales[2]
            labelledYValue = advertising_spend_vs_sales[3]
            labelledXValueUnit = advertising_spend_vs_sales[4]
            labelledYValueUnit = advertising_spend_vs_sales[5]
            break

        elif dataSetChoice == "hours_studied_vs_test_scores":
            x_val = hours_studied_vs_test_scores[0]
            y_val = hours_studied_vs_test_scores[1]
            labelledXValue = hours_studied_vs_test_scores[2]
            labelledYValue = hours_studied_vs_test_scores[3]
            labelledXValueUnit = hours_studied_vs_test_scores[4]
            labelledYValueUnit = hours_studied_vs_test_scores[5]
            break

        elif dataSetChoice == "temperature_increase_over_time":
            x_val = temperature_increase_over_time[0]
            y_val = temperature_increase_over_time[1]
            labelledXValue = temperature_increase_over_time[2]
            labelledYValue = temperature_increase_over_time[3]
            labelledXValueUnit = temperature_increase_over_time[4]
            labelledYValueUnit = temperature_increase_over_time[5]
            break

        elif dataSetChoice == "engine_size_vs_fuel_consumption":
            x_val = engine_size_vs_fuel_consumption[0]
            y_val = engine_size_vs_fuel_consumption[1]
            labelledXValue = engine_size_vs_fuel_consumption[2]
            labelledYValue = engine_size_vs_fuel_consumption[3]
            labelledXValueUnit = engine_size_vs_fuel_consumption[4]
            labelledYValueUnit = engine_size_vs_fuel_consumption[5]
            break

        
        elif dataSetChoice == "years_vs_company_revenue":
            x_val = years_vs_company_revenue[0]
            y_val = years_vs_company_revenue[1]
            labelledXValue = years_vs_company_revenue[2]
            labelledYValue = years_vs_company_revenue[3]
            labelledXValueUnit = years_vs_company_revenue[4]
            labelledYValueUnit = years_vs_company_revenue[5]
            break

        elif dataSetChoice == "production_volume_vs_cost":
            x_val = production_volume_vs_cost[0]
            y_val = production_volume_vs_cost[1]
            labelledXValue = production_volume_vs_cost[2]
            labelledYValue = production_volume_vs_cost[3]
            labelledXValueUnit = production_volume_vs_cost[4]
            labelledYValueUnit = production_volume_vs_cost[5]
            break

        elif dataSetChoice == "months_vs_subscribers":
            x_val = months_vs_subscribers[0]
            y_val = months_vs_subscribers[1]
            labelledXValue = months_vs_subscribers[2]
            labelledYValue = months_vs_subscribers[3]
            labelledXValueUnit = months_vs_subscribers[4]
            labelledYValueUnit = months_vs_subscribers[5]
            break

        elif dataSetChoice == "practice_time_vs_skill_level":
            x_val = practice_time_vs_skill_level[0]
            y_val = practice_time_vs_skill_level[1]
            labelledXValue = practice_time_vs_skill_level[2]
            labelledYValue = practice_time_vs_skill_level[3]
            labelledXValueUnit = practice_time_vs_skill_level[4]
            labelledYValueUnit = practice_time_vs_skill_level[5]
            break

        elif dataSetChoice == "days_vs_website_visits":
            x_val = days_vs_website_visits[0]
            y_val = days_vs_website_visits[1]
            labelledXValue = days_vs_website_visits[2]
            labelledYValue = days_vs_website_visits[3]
            labelledXValueUnit = days_vs_website_visits[4]
            labelledYValueUnit = days_vs_website_visits[5]
            break

        elif dataSetChoice == "temperature_vs_electricity_usage":
            x_val = temperature_vs_electricity_usage[0]
            y_val = temperature_vs_electricity_usage[1]
            labelledXValue = temperature_vs_electricity_usage[2]
            labelledYValue = temperature_vs_electricity_usage[3]
            labelledXValueUnit = temperature_vs_electricity_usage[4]
            labelledYValueUnit = temperature_vs_electricity_usage[5]
            break
            
    title = input("Choose a title >> ")

    print("\nLoading DataSet..")
    print("Setting up settings..\n")
    time.sleep(3)
    print(f"Modelling Linear Equation: {returnEquation()}")

    calculateAverageErr(labelledXValue, labelledYValue, labelledXValueUnit, labelledYValueUnit)
    plt.scatter(x_val, y_val, color='blue', label='Data Points')
    plt.xlabel(labelledXValue + " (" + labelledXValueUnit + ")")
    plt.ylabel(labelledYValue + " (" + labelledYValueUnit + ")")

    indiviusalElements = returnIndivisualNumbersOfEquation()
    plt.title(title)
    x = np.linspace(x_val[0], x_val[-1], 100)
    y = (indiviusalElements[0] * x) + indiviusalElements[1]
    y_str = "y = " + str(indiviusalElements[0]) + "x + " + str(indiviusalElements[1])

    plt.plot(x, y, color='red', label=y_str)
    plt.legend()
    plt.show(block=False)
    
    # Input
    while True:
        inputType = input(f"Value predicting [{labelledXValue}, {labelledYValue}] >> ")

        if inputType.lower() == labelledXValue.lower():
            inputValue = input(f"Enter {labelledYValue}: ")
            print(f"\nPredicted {labelledXValue} Value: {str(predictValue("x", float(inputValue)))} {labelledXValueUnit}\n")
        elif inputType.lower() == labelledYValue.lower():
            inputValue = input(f"Enter {labelledXValue}: ")
            print(f"\nPredicted {labelledYValue} Value:  {str(predictValue("y", float(inputValue)))} {labelledYValueUnit}\n")
        elif inputType.lower() == "exit":
            mainProgramInputPrediction()
            return
        else:
            print("Invalid Request")

while True:
    try:
        mainProgramInputPrediction()
    except Exception as e:
        print(y_val)
        print(x_val)
        print(advertising_spend_vs_sales[0])
   
        print(f"\n❌ Error: {e}")
        choice = input("Do you want to try again? (y/n): ")
        if choice.lower() != "y":
            break
