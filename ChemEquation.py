import numpy as np
from fractions import Fraction

MAX_DENOM=100

def parse_elements(element_str):
    return element_str.split()

def parse_equation(equation_str):
    reactants, products = equation_str.split("->")
    reactants = reactants.strip().split("+")
    products = products.strip().split("+")
    return reactants, products

def construct_matrix(elements, reactants, products):
    num_elements = len(elements)
    num_reactants = len(reactants)
    num_products = len(products)
    matrix = np.zeros((num_elements, num_reactants + num_products + 1))

    for i, compound in enumerate(reactants):
        compound_elements = compound.strip().split()
        for element_count in compound_elements:
            element_name = ''.join(filter(str.isalpha, element_count))
            count = ''.join(filter(str.isdigit, element_count))
            if not count:
                count = '1'
            matrix[elements.index(element_name), i] = int(count)
    
    for i, compound in enumerate(products):
        compound_elements = compound.strip().split()
        for element_count in compound_elements:
            element_name = ''.join(filter(str.isalpha, element_count))
            count = ''.join(filter(str.isdigit, element_count))
            if not count:
                count = '1'
            matrix[elements.index(element_name), num_reactants + i] = -int(count)  # negative coefficients for products
    
    return matrix

def echelon_form(matrix):
    m, n = matrix.shape
    A = matrix.copy()
    lead = 0
    for r in range(m):
        if lead >= n:
            break
        while all(A[r:, lead] == 0):
            lead += 1
            if lead == n:
                break
        else:
            i = np.argmax(A[r:, lead]) + r
            A[[i, r]] = A[[r, i]]
            lv = A[r, lead]
            for i in range(r + 1, m):
                lv = A[i, lead]
                A[i] = A[i] - lv * A[r] / A[r, lead]
            lead += 1
    return A

def reduced_echelon_form(matrix):
    echelon_matrix = echelon_form(matrix)
    m, n = echelon_matrix.shape
    lead = 0
    for r in range(m):
        if lead >= n:
            break
        while all(echelon_matrix[r:, lead] == 0):
            lead += 1
            if lead == n:
                break
        else:
            echelon_matrix[r] = echelon_matrix[r] / echelon_matrix[r, lead]
            for i in range(r):
                echelon_matrix[i] = echelon_matrix[i] - echelon_matrix[i, lead] * echelon_matrix[r]
            lead += 1
    return echelon_matrix

def solve_equation(matrix):
    m, n = matrix.shape
    solution = {}
    coeffs = {}

    # Iterate over each row to find the pivot variable and calculate its value
    for i in range(m):
        pivot_column = np.argmax(matrix[i, :-1])
        if matrix[i, pivot_column] == 0:
            continue  # Skip trivial equations
        variable_index = pivot_column
        variable_value = matrix[i, -1] / matrix[i, pivot_column]
        solution[f'X{variable_index + 1}'] = variable_value

    # finding the free variables
    for j in range(n - 1):
        if f'X{j + 1}' not in solution:
            # here I am assiging 1 to the free variable so that I can calculate others based on it
            solution[f'X{j + 1}'] = 1
            free_var_index = j
            # Iterate through the matrix to adjust values of basic variables
            for i in range(m):
                if i != free_var_index:
                    coefficient = matrix[i, free_var_index]
                    solution[f'X{i + 1}'] += coefficient * solution[f'X{j + 1}']

    # I wanted to have integer answers in the end so here we convert numbers to integer by claculating the LCM
    for var, value in solution.items():
        if var.startswith('X'):
            coeffs[var] = value

    fractions = [Fraction(val).limit_denominator(MAX_DENOM) for val in coeffs.values()]
    ratios = np.array([(f.numerator, f.denominator) for f in fractions])
    factor = np.lcm.reduce(ratios[:,1])


    for var, value in solution.items():
        if var.startswith('X'):
            solution[var] *= factor

    return solution

def main():
    elements_input = input("Enter elements separated by spaces: ")
    elements = parse_elements(elements_input)

    equation_input = input("Enter chemical equation: ")
    reactants, products = parse_equation(equation_input)

    matrix = construct_matrix(elements, reactants, products)
    
    print("Original Augmented Matrix:")
    print(matrix)
    
    print("\nEchelon Augmented Matrix:")
    echelon_matrix = echelon_form(matrix)
    print(echelon_matrix)
    
    print("\nReduced Echelon Augmented Matrix:")
    reduced_echelon_matrix = reduced_echelon_form(echelon_matrix)
    print(reduced_echelon_matrix)
    
    print("\nSolution:")
    solution = solve_equation(reduced_echelon_matrix)
    for variable, value in solution.items():
        if value < 0:
            value = -1 * value
        print(f"{variable} = {Fraction(value).limit_denominator()}")

if __name__ == "__main__":
    main()
    input("press Enter to exit...")
