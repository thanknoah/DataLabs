# Imports 
import matplotlib.pyplot as plt
import numpy as np
import time, random, re
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

# Colour text helper
def ctext(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

# Averages
def averageX():
    return sum(x_val) / len(x_val) if x_val else 0

def averageY():
    return sum(y_val) / len(y_val) if y_val else 0

# Calculate slope (m) using least squares
def calculateLinearSlope(avg_x, avg_y):
    numerator = sum((x_val[i] - avg_x) * (y_val[i] - avg_y) for i in range(len(x_val)))
    denominator = sum((x_val[i] - avg_x) ** 2 for i in range(len(x_val)))
    return numerator / denominator if denominator != 0 else 0

# Return formatted equation string
def returnEquation():
    m = calculateLinearSlope(averageX(), averageY())
    c = averageY() - (m * averageX())
    return f"y = {m:.6f}x + {c:.6f}"

def returnIndivisualNumbersOfEquation():
    m = calculateLinearSlope(averageX(), averageY())
    c = averageY() - (m * averageX())
    return [m, c]

# Predict using y = mx + c or x = (y - c)/m
def predictValue(valType, val):
    m = calculateLinearSlope(averageX(), averageY())
    c = averageY() - (m * averageX())
    if m == 0:
        return None
    return (val - c) / m if valType == "x" else (m * val) + c

# Error Calculation
def calculateAverageErr(labelledXValue, labelledYValue, labelledXValueUnit, labelledYValueUnit):
    err_x = sum(abs(x_val[i] - predictValue("x", y_val[i])) for i in range(len(x_val))) / len(x_val) if x_val else 0
    err_y = sum(abs(y_val[i] - predictValue("y", x_val[i])) for i in range(len(x_val))) / len(y_val) if y_val else 0
    print(ctext(f"\n📏 Error Margin {labelledXValue} [X]: +/-{err_x:.2f} {labelledXValueUnit}", "93"))
    print(ctext(f"⚡ Error Margin {labelledYValue} [Y]: +/-{err_y:.2f} {labelledYValueUnit}\n", "93"))

# Welcome screen
def print_welcome():
    print(ctext(r"""
 ___         _          _           _        
|   \  __ _ | |_  __ _ | |    __ _ | |__  ___
| |) |/ _` ||  _|/ _` || |__ / _` ||  _ \(_-/
|___/ \__/_| \__|\__/_||____|\__/_||____//__/
    """, "95"))
    print(ctext("=" * 60, "90")) 
    print(
        ctext("🤖💻👨‍💻  ", "95") + 
        ctext("Welcome to ", "96") + 
        ctext("DataLabs Beta [v1.1]", "94")
    )
    print()
    print(ctext("🔬 Explore and model real-world datasets using linear regression.", "92"))
    print(ctext("📈 Now with built-in Best Fit Line visualisation!", "96"))
    print(ctext("🧮 Includes powerful Calculus Tools: Differentiate & Integrate.", "95"))
    print()
    print(ctext("⚠️  Please note: All model outputs are approximations.", "93"))
    print()
    print(ctext("\n💻 Developed with care by ", "96") + ctext("Noah\n", "94") + ctext("📦 Credits: ", "90") + ctext("Userman2424", "92") +  ctext(" (Differentiation formatting algorithm), ", "90") + ctext("ChatGPT 🤖", "95") + ctext(" (Helped with GUI)\n", "90"))
    print(ctext("=" * 60 + "\n", "90"))

# Show available datasets
def print_dataset_menu():
    print(ctext("📂 Available Datasets:", "94"))
    for i, name in enumerate(dataset_names, 1):
        print(ctext(f"  {i}. {name}", "94"))
    print(ctext("\n🔍 Type the dataset name exactly as shown to select it.\n", "96"))

# Load dataset by name
def load_dataset(name):
    global x_val, y_val
    ds_map = {
        "advertising_spend_vs_sales": advertising_spend_vs_sales,
        "hours_studied_vs_test_scores": hours_studied_vs_test_scores,
        "temperature_increase_over_time": temperature_increase_over_time,
        "engine_size_vs_fuel_consumption": engine_size_vs_fuel_consumption,
        "years_vs_company_revenue": years_vs_company_revenue,
        "production_volume_vs_cost": production_volume_vs_cost,
        "months_vs_subscribers": months_vs_subscribers,
        "practice_time_vs_skill_level": practice_time_vs_skill_level,
        "days_vs_website_visits": days_vs_website_visits,
        "temperature_vs_electricity_usage": temperature_vs_electricity_usage
    }
    if name in ds_map:
        x_val, y_val, x_label, y_label, x_unit, y_unit = ds_map[name]
        return x_label, y_label, x_unit, y_unit
    return None, None, None, None

# Load Linear Regression Model
def linearRegressionModellingOption():
    print(ctext("\n📊 Linear Regression Modelling Selected.\n", "96"))
    while True:
        print_dataset_menu()
        dataSetChoice = input(ctext("⚙️ Option >> ", "96")).strip()
        labelledXValue, labelledYValue, labelledXValueUnit, labelledYValueUnit = load_dataset(dataSetChoice)
        if labelledXValue:
            print(ctext(f"✅ Dataset '{dataSetChoice}' loaded successfully!\n", "92"))
            break
        else:
            print(ctext("❌ Dataset not found. Please check spelling and try again.\n", "91"))

    title = input(ctext("📌 Choose a title for the graph >> ", "96")).strip()
    if not title:
        title = "Linear Regression Graph"
    print(ctext("🔧 Setting up dataset for modelling...", "96"))
    time.sleep(1)

    print(ctext(f"\n📈 Modelling Linear Equation: {returnEquation()}", "96"))

    calculateAverageErr(labelledXValue, labelledYValue, labelledXValueUnit, labelledYValueUnit)

    plt.style.use('ggplot')
    plt.scatter(x_val, y_val, color='blue', label='Data Points')
    plt.xlabel(f"{labelledXValue} ({labelledXValueUnit})", fontsize=11)
    plt.ylabel(f"{labelledYValue} ({labelledYValueUnit})", fontsize=11)
    plt.title(title, fontsize=13)
    m, c = returnIndivisualNumbersOfEquation()
    x_range = np.linspace(min(x_val), max(x_val), 100)
    plt.plot(x_range, m * x_range + c, color='red', label=f"y = {m:.4f}x + {c:.4f}")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    while True:
        print(ctext(f"\n🎯 Predict '{labelledXValue}' or '{labelledYValue}' (or type 'exit')", "94"))
        inputType = input(ctext("Your choice: ", "96")).strip().lower()

        if inputType == labelledXValue.lower():
            try:
                value = float(input(ctext(f"Enter {labelledYValue}: ", "96")))
            except ValueError:
                print(ctext("❌ Invalid number input. Please try again.", "91"))
                continue
            result = predictValue("x", value)
            if result is None:
                print(ctext("⚠️ Cannot predict X due to zero slope.", "93"))
            else:
                print(ctext(f"\n🔮 Predicted {labelledXValue}: {result:.2f} {labelledXValueUnit}\n", "92"))

        elif inputType == labelledYValue.lower():
            try:
                value = float(input(ctext(f"Enter {labelledXValue}: ", "96")))
            except ValueError:
                print(ctext("❌ Invalid number input. Please try again.", "91"))
                continue
            result = predictValue("y", value)
            if result is None:
                print(ctext("⚠️ Cannot predict Y due to zero slope.", "93"))
            else:
                print(ctext(f"\n🔮 Predicted {labelledYValue}: {result:.2f} {labelledYValueUnit}\n", "92"))

        elif inputType == "exit":
            confirm = input(ctext("🔄 Restart session? (y/n): ", "96"))
            if confirm.lower() == 'y':
                plt.close()
                print("\n")
                mainProgram()
            return

        else:
            print(ctext("❌ Invalid input. Please try again.", "91"))


# Divide terms into coefficient, sign, exponent (CREDITS TO USERMAN242 FOR BUILDING THIS)
def get_coefficient_and_exponent(term):
    pattern = r'(?P<sign>[+-]?)(?P<coef>\d+(\.\d+)?|)(x(\^(?P<exp>-?\d+(\.\d+)?)?)?)?'
    match = re.fullmatch(pattern, term)
    includes_x = 'x' in term

    if not match:
        return None

    sign = match.group("sign")
    sign = -1 if sign == '-' else 1
    coef = match.group("coef")

    if coef == '':
        coef = 1.0 if includes_x else 0.0
    else:
        coef = float(coef)

    coef *= sign

    if includes_x:
        if match.group("exp") is None:
            exponent = 1.0
        else:
            exponent = float(match.group("exp"))
    else:
        exponent = 0.0

    return coef, exponent, includes_x


# Calculus Calculations (integrating, differentiating)
def calculusCalculationOption():
    print(ctext("\n🧮 Calculus Calculation Selected.", "96"))
    exit = False
    while not exit:
        print(ctext("🔮 Format accepted examples: [2x^3, 2x, 2, 2x^2+5x+5] (EXTENDS TO NEGATIVE COEFFICENT AND POWER)", "96"))
        calculusMethod = input(ctext("\n⚙️ Choose mode: Integrate or Differentiate >> ", "96")).strip().lower()

        if calculusMethod not in ["differentiate", "integration"]:
            print(ctext("\n❌ Invalid input! Please type 'Differentiate' or 'Integration'.", "91"))
            continue

        print(ctext(f"\n✅ Successfully selected {calculusMethod} mode.", "92"))
        print(ctext("🛑 Type 'exitmode' to leave this mode or 'exit' to quit Calculus Calculation.\n", "93"))

        while True:
            equation = input(ctext("📝 Enter equation >> ", "96")).strip()
            if equation.lower() == "exitmode":
                print(ctext(f"\n🔄 Exited {calculusMethod} mode.\n", "92"))
                break
            if equation.lower() == "exit":
                print(ctext("\n👋 Exiting Calculus Calculation...", "93"))
                exit = True
                return

            expr = equation.replace(" ", "")

            term_pattern = r'[+-]?(\d+(\.\d+)?|)?x?(\^-?\d+(\.\d+)?)?'
            full_pattern = f'^({term_pattern})+$'

            if re.fullmatch(full_pattern, expr):
                if not expr.startswith(('+', '-')):
                    expr = '+' + expr

                terms = re.findall(r'[+-][^+-]+', expr)

                if calculusMethod == "differentiate":
                    groupOfTerms = []

                    for term in terms:
                        result = get_coefficient_and_exponent(term)
                        if result is None:
                            print(ctext("❌ Invalid term format detected, please try again.", "91"))
                            break

                        coef, exponent, includes_x = result

                        if not includes_x:
                            continue  # derivative of constant is 0

                        new_coef = coef * exponent
                        new_exponent = exponent - 1

                        if new_exponent == 0:
                            currentTerm = str(round(new_coef, 6))
                        elif new_exponent == 1:
                            currentTerm = f"{round(new_coef, 6)}x"
                        else:
                            currentTerm = f"{round(new_coef, 6)}x^{round(new_exponent, 6)}"

                        groupOfTerms.append(currentTerm)

                    result = 0 if len(groupOfTerms) == 0 else " + ".join(groupOfTerms).replace("+ -", "- ")
                    print(ctext(f"\n🧮 Derivative (dy/dx): {result}\n", "92"))
                else:
                    print(ctext("🚧 Integration mode coming soon! Stay tuned.\n", "93"))
            else:
                print(ctext("\n❌ Invalid expression. Use terms like 2x^2+3x+1 or 2x^-1+4\n", "91"))
    mainProgram()

# Main program
def mainProgram():
    print_welcome()
    while True:
        tool_usage = input(ctext("\n⚙️ Pick a tool / command: Linear Regression Modelling, Calculus Calculation, Update Log, Credits, Pledge >> ", "96")).strip().lower()
        if tool_usage == "linear regression modelling":
            linearRegressionModellingOption()
        elif tool_usage == "calculus calculation":
            calculusCalculationOption()
        elif tool_usage == "update log":
            print(ctext("\n📈 Beta V1.1 UPDATE LOG: New calculus mode introduced (differentation only for now), enhanced UI.", "95"))
        elif tool_usage == "credits":
                print(ctext("\n💻 Developed with care by ", "95") + ctext("Noah\n", "96") + ctext("📦 Credits: ", "95") + ctext("Userman2424", "95") +  ctext(" (Differentiation formatting algorithm), ", "95") + ctext("ChatGPT 🤖", "95") + ctext(" (Helped with GUI)", "95"))
        elif tool_usage == "pledge":
                print("\nComputer >> To enhance the experience for you, dear user! :)")
        else:
            print(ctext("\n❌ Invalid input! Please choose either 'Linear Regression Modelling' or 'Calculus Calculation'.", "91"))

# Run
while True:
    try:
        mainProgram()
    except Exception as e:
        print(ctext(f"\n❌ Error: {e}", "91"))
        if input(ctext("Try again? (y/n): ", "96")).lower() != 'y':
            print(ctext("\n👋 Goodbye!\n", "93"))
            break
        else:
            print("\n")
