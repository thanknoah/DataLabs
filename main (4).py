# Imports
import time, random

# Sample X data
x_val = [
0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
20, 21, 22, 23, 24
]

# Sample Y data
y_val = [
18.82026172983832,
14.500786041836117,
19.893689920528697,
28.70446599600729,
29.337789950749837,
17.613610600617946,
29.750442087627945,
26.74321395851151,
29.48390574103221,
34.55299250969186,
35.72021785580439,
44.77136753481488,
43.80518862573497,
43.10837508246414,
47.21931616372713,
49.168371636871335,
57.47039536578803,
51.47420868117099,
56.565338508254506,
53.229521303491374,
47.23505092082961,
65.7680929772018,
69.32218099429753,
63.78917489796779,
81.34877311993804
]



# Averages
def averageX():
    total = 0
    for x in x_val:
        total = total + x

    return total/len(x_val)

def averageY():
    total = 0
    for y in y_val:
        total = total + y

    return total/len(y_val)

# m=∑(x[i]​−average(x)) * ∑(x[i]​−average(x))  / (y[i]​−average(y)​)​,
def calculateLinearSlope(averageX, averageY):
    totalSigmaNotation = 0
    totalSigmaNotation2 = 0

    for index in range(len(x_val)):
        totalSigmaNotation = totalSigmaNotation + (x_val[index] - averageX)*(y_val[index] - averageY)
        totalSigmaNotation2 = totalSigmaNotation2 + (x_val[index] - averageX) ** 2

    return totalSigmaNotation / totalSigmaNotation2

def returnEquation():
    m = calculateLinearSlope(averageX(), averageY())
    c = averageY() - (m * averageX())
    return f"y = {m}x + {c}"


# y=mx+c
def predictValue(valType, val):
    averageXVal = averageX()
    averageYVal = averageY()
    m = calculateLinearSlope(averageXVal, averageYVal) # Gradient
    c = averageYVal - (m * averageXVal) # Y intercept

    if valType == "x":
         return (val-c) / m
    else:
        return (m*val) + c

# Error Calculation
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

# Main program
def mainProgramInputPrediction():
    print("Welcome to DataLabs Beta, user. This program is for data scientists, programmers, mathemeticians.")
    print("New feature: liner regression algorithim, (aka line of best fit on computer) to predict x/y value.")
    print("Warning: Values may be unexpected, as this is soley a line of best fit.")
    print("Credits: Noah\n")

    labelledXValue = input("Label X Axis >> ")
    labelledXValueUnit = input("Label X Unit >> ")
    labelledYValue = input("Label Y Axis >> ")
    labelledYValueUnit = input("Label Y Unit >> ")

    print("\nSetting up settings..\n")
    time.sleep(3)
    print(f"Modelling Linear Equation: {returnEquation()}")
    calculateAverageErr(labelledXValue, labelledYValue, labelledXValueUnit, labelledYValueUnit)
    
    # Input
    while True:
        inputType = input(f"Value predicting [{labelledXValue}, {labelledYValue}] >> ")

        if inputType.lower() == labelledXValue.lower():
            inputValue = input(f"Enter {labelledYValue}: ")
            print(f"\nPredicted {labelledYValue} Value: {str(predictValue("x", float(inputValue)))} {labelledYValueUnit}\n")
        elif inputType.lower() == labelledYValue.lower():
            inputValue = input(f"Enter {labelledXValue}: ")
            print(f"\nPredicted {labelledXValue} Value:  {str(predictValue("y", float(inputValue)))} {labelledXValueUnit}\n")
        else:
            print("Invalid")

mainProgramInputPrediction()
