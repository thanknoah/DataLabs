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

# Divide terms into coefficient, sign, exponent (CREDITS TO USERMAN242 FOR BUILDING THIS)
def get_coefficient_and_exponent(term, method, analysis):
    pattern = r'^(?P<sign>[+-]?)(?P<coef>\d+(\.\d+)?|)(x(\^(?P<exp>-?\d+(\.\d+)?))?)?$'
    match = re.fullmatch(pattern, term.strip())

    # Error handling
    if not match:
        return None

    # Handling Sign
    sign_str = match.group("sign")
    sign = -1 if sign_str == '-' else 1
    exp_str = match.group("exp")
    coef_str = match.group("coef")
    includes_x = 'x' in term

    # If we just want to soley analyse the coeff and exponent without touching it
    if analysis == True:
        if exp_str is None:
            exponent = 1.0
        else:
            exponent = float(exp_str)

        if coef_str == '':
            coef = 1
        else:
            coef = float(coef_str)

        coef *= sign
        return coef, exponent, includes_x

    # Adding X to plain integers for integration, removing X for plain integers in differentiation
    if coef_str == '':
       if includes_x: 
          coef = 1.0
       elif method == "Differentiation": 
          coef = 0.0
       elif method == "Integration":
          coef = 1.0
    else:
        coef = float(coef_str)

    # Assigning Sign to coefficent
    coef *= sign
   
    # Handling Coefficent
    if includes_x:
        if exp_str is None:
            exponent = 1.0
        else:
            exponent = float(exp_str)
    else:
        exponent = 0.0

    return coef, exponent, includes_x

# Differentiate
def differentiate(terms):
    groupOfTerms = []
    for term in terms:
        result = get_coefficient_and_exponent(term, "Differentiation", False)
        if result is None:
            print(ctext("\n❌ Invalid term format detected, please try again.\n", "91"))
            continue

        coef, exponent, includes_x = result
        if not includes_x:
            continue  # derivative of constant is 0

        new_exponent = exponent - 1
        new_coef = coef * exponent

        if new_exponent == 0:
            currentTerm = str(clean_num(new_coef))
        elif new_exponent == 1:
            currentTerm = f"{clean_num(new_coef)}x"
        else:
            currentTerm = f"{clean_num(new_coef)}x^{clean_num(new_exponent)}"

        if abs(new_coef) == 1:
           coef_str = "-" if new_coef < 0 else ""
        else:
           coef_str = str(clean_num(new_coef))
        groupOfTerms.append(currentTerm)

    result = 0 if len(groupOfTerms) == 0 else " + ".join(groupOfTerms).replace("+ -", "- ")
    return result

# f'(x), e.g f(3)
def applyValuesInFunctions(terms, x, type):
   total = 0

   if type == "Differentiation":
       terms = differentiate(terms)
   else:
       terms = " + ".join(terms).replace("+ -", "- ")

   expr = terms.replace(" ", "")
   if not expr.startswith(('+', '-')): expr = '+' + expr

   terms = re.findall(r'[+-][^+-]+', expr)

   for term in terms:
       coef, exponent, includes_x = get_coefficient_and_exponent(term, "Differentiation", True)
       if includes_x:
           total = total + (x ** exponent * coef)
       else:
           total = total + coef
        
   return float(total)

# Cleaning up number for output
def clean_num(n):
    return int(n) if n == int(n) else round(n, 6)

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
        ctext("\n🤖💻👨‍💻  ", "95") + 
        ctext("Welcome to ", "96") + 
        ctext("DataLabs Beta [v1.2]", "94")
    )
    print()
    print(ctext("🔬 Explore and model real-world datasets using linear regression.", "92"))
    print(ctext("📈 Now with built-in Best Fit Line visualisation!", "96"))
    print(ctext("🧮 Includes powerful Calculus Tools: Differentiate & Integrate.", "95"))
    print()
    print(ctext("⚠️  Please note: All model outputs are approximations.", "93"))
    print()
    print(ctext("💻 Developed with care by ", "96") + ctext("Noah\n", "94") + ctext("📦 Credits: ", "90") + ctext("Userman2424", "92") +  ctext(" (Differentiation formatting algorithm), ", "90") + ctext("ChatGPT 🤖", "95") + ctext(" (Helped with GUI)\n", "90"))
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

# Main program
def mainProgram():
    print_welcome()
    while True:
        print(ctext("⚙️ Non-tool Commands: Update Log, Credits, Pledge ", "96"))
        tool_usage = input(ctext("⚙️ Pick a tool: Linear Regression Modelling, Calculus Calculator, Equation Root Estimator >> ", "96")).strip().lower()
        if tool_usage == "linear regression modelling":
            linearRegressionModellingOption()
        elif tool_usage == "calculus calculator":
            calculusCalculationOption()
        elif tool_usage == "equation root estimator":
            equationRootEstimator()
        elif tool_usage == "update log":
            print(ctext("\n📈 Beta V1.2 UPDATE LOG: Added integration to Calculus Calculation, bug fixes!", "95"))
        elif tool_usage == "credits":
            print(ctext("\n💻 Developed with care by ", "95") + ctext("Noah\n", "96") + ctext("📦 Credits: ", "95") + ctext("Userman2424", "95") +  ctext(" (Differentiation formatting algorithm), ", "95") + ctext("ChatGPT 🤖", "95") + ctext(" (Helped with GUI)", "95"))
        elif tool_usage == "pledge":
            print("\nComputer >> To enhance the experience for you, dear user! :)")
        else:
            print(ctext("\n❌ Invalid input! Please choose either 'Linear Regression Modelling' or 'Calculus Calculation'.", "91"))

# Find 2 points where sign change [CREDIT USERMAN242]
def find_sign_changes(terms, start=-100, end=100, step=0.05):
    points = [start + i * step for i in range(int((end - start) / step) + 1)]
    sign_change_intervals = []
    
    for i in range(len(points) - 1):
        x0 = points[i]
        x1 = points[i + 1]
        f0 = applyValuesInFunctions(terms, x0, "Normal")
        f1 = applyValuesInFunctions(terms, x1, "Normal")

        if f0 is None or f1 is None:
            continue

        if f0 * f1 < 0 or f0 == 0 or f1 == 0:
            sign_change_intervals.append((x0, x1))
    return sign_change_intervals

# Estimate Roots Of Any Equation (NEWTON RAPHSON METHOD)
def equationRootEstimator():
    exit = False
    print(ctext("\n📊 Estimating Equation Roots Selected.", "96"))
    print(ctext("🔮 Format accepted examples: [2x^3, 2x, 2, 2x^2+5x+5] (EXTENDS TO NEGATIVE COEFFICENT AND POWER)", "96"))
    print(ctext("📈 Using Newton-Raphson Algorithim, not as advanced, and as a result can lead to weird results.\n", "96"))
    print(ctext("⚠️ Warning: With quartics and above, program may not provide all the roots. (Accuracy won't be effected)\n", "93"))

    while not exit:
        equation = input(ctext("📝 Enter equation >> ", "96")).strip()
        if equation.lower() == "exit":
            print(ctext("\n👋 Exiting Estimating Equation Roots...", "93"))
            exit = True
            return

        expr = equation.replace(" ", "")
        term_pattern = r'[+-]?(\d+(\.\d+)?(x(\^-?\d+(\.\d+)?)?)?|x(\^-?\d+(\.\d+)?)?)'
        full_pattern = f'^({term_pattern})+$'

        if re.fullmatch(full_pattern, expr):
            if not expr.startswith(('+', '-')):
                expr = '+' + expr

            terms = re.findall(r'[+-][^+-]+', expr)
            basePoints = find_sign_changes(terms)
            solutions = []

            for a,b  in basePoints:
                x_n = (a+b)/2
                for x in range(10):
                    if applyValuesInFunctions(terms, x_n, "Differentiation") == 0: break
                    x_next = x_n - applyValuesInFunctions(terms, x_n, "Normal") / applyValuesInFunctions(terms, x_n, "Differentiation")
                    prev_x = x_n
                    x_n = x_next
 
                    if abs(x_n - prev_x) < 1e-6: # If sequence diverges
                        break

                solutions.append(clean_num(x_n))

            deduped = []
            for s in solutions:
               if not any(abs(s - r) < 1e-8 for r in deduped):
                  deduped.append(s) 
            if len(deduped) == 0:
                print(ctext("\n⚠️ Equation has no real roots, most likely complex solutions.\n", "93"))

            print(ctext(f"\n🧮 Estimated Solutions: {solutions}\n", "92"))
        else:
            print(ctext("\n❌ Invalid expression. Use terms like 2x^2+3x+1 or 2x^-1+4\n", "91"))
    mainProgram()


# Load Linear Regression Model
def linearRegressionModellingOption():
    exit = False
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
    plt.plot(x_range, m * x_range + c, color='red', label=f"y = {m:.3f}x + {c:.3f}")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    while exit == False:
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
                print(ctext(f"\n🔮 Predicted {labelledXValue}: {result:.2f} {labelledXValueUnit}", "92"))

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
             print(ctext("\n👋 Exiting Linear Regression Modelling...", "93"))
             plt.close()
             exit = True
             return
        else:
            print(ctext("❌ Invalid input. Please try again.", "91"))
    mainProgram()

# Calculus Calculations (integrating, differentiating)
def calculusCalculationOption():
    print(ctext("\n🧮 Calculus Calculation Selected.", "96"))
    exit = False

    while not exit:
        print(ctext("🔮 Format accepted examples: [2x^3, 2x, 2, 2x^2+5x+5] (EXTENDS TO NEGATIVE COEFFICENT AND POWER)", "96"))
        calculusMethod = input(ctext("\n⚙️ Choose mode: Integration or Differentiate >> ", "96")).strip().lower()

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
            term_pattern = r'[+-]?(\d+(\.\d+)?(x(\^-?\d+(\.\d+)?)?)?|x(\^-?\d+(\.\d+)?)?)'
            full_pattern = f'^({term_pattern})+$'

            if re.fullmatch(full_pattern, expr):
                if not expr.startswith(('+', '-')):
                    expr = '+' + expr

                terms = re.findall(r'[+-][^+-]+', expr)

                if calculusMethod == "differentiate":
                    print(ctext(f"\n🧮 Derivative f'(x) = {differentiate(terms)}\n", "92"))
                elif calculusMethod == "integration":
                    groupOfTerms = []
                    
                    for term in terms:
                        result = get_coefficient_and_exponent(term, "Integration", False)
                        if result is None:
                            print(ctext("\n❌ Invalid term format detected, please try again.\n", "91"))
                            continue

                        coef, exponent, includes_x = result

                        if exponent == -1:
                            currentTerm = f"{str(clean_num(coef))}ln |x|"
                        elif exponent == 0 and not includes_x:
                            currentTerm = f"{str(clean_num(coef))}x"
                        else:
                            new_exponent = exponent + 1
                            new_coef = coef / new_exponent
                            currentTerm = f"{clean_num(new_coef)}x^{clean_num(new_exponent)}"

                            if abs(new_coef) == 1:
                               coef_str = "-" if new_coef < 0 else ""
                            else:
                               coef_str = str(clean_num(new_coef))

                        
                        groupOfTerms.append(currentTerm)
                        
                    result = 0 if len(groupOfTerms) == 0 else " + ".join(groupOfTerms).replace("+ -", "- ")
                    print(ctext(f"\n🧮 Integrated Expression ∫f(x) dx = {result} + C\n", "92"))

            else:
                print(ctext("\n❌ Invalid expression. Use terms like 2x^2+3x+1 or 2x^-1+4\n", "91"))
    mainProgram()

# Run
try:
    mainProgram()
except Exception as e:
    print(ctext(f"\n❌ Error: {e}", "91"))
    if input(ctext("Try again? (y/n): ", "96")).lower() != 'y':
        print(ctext("\n👋 Goodbye!\n", "93"))
    else:
        print("\n")
