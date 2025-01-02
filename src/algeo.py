import numpy as np
import json
import sys

def load_data_from_json(filename):
    """Load input from a JSON file"""
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("Warning: JSON file not found. Please provide input manually.")
        return None

def convert_to_matrix(coefficients):
    """Convert a differential equation into a system of coupled first-order differential equations"""
    order = len(coefficients) - 1
    A = np.zeros((order, order))

    for i in range(order - 1):
        A[i, i + 1] = 1 

    for i in range(order):
        A[order - 1, i] = -coefficients[order - i] / coefficients[0]

    return A

def get_user_input():
    """Get user input from the terminal if a JSON file is not provided"""
    coefficients = list(map(float, input("Enter the coefficients (space-separated): ").split()))
    
    if len(coefficients) < 2:
        print("Error: The differential equation must atleast be a second-order differential equation")
        sys.exit(1)

    y0 = list(map(float, input(f"Enter initial conditions (length should be {len(coefficients) - 1}): ").split()))
    
    if len(y0) != len(coefficients) - 1:
        print(f"Error: The length of initial conditions must be {len(coefficients) - 1}.")
        sys.exit(1)

    x = float(input("Enter the value of x: "))
    
    return coefficients, y0, x

def main():
    if len(sys.argv) == 2:
        """Check if a JSON file is provided"""
        json_file = sys.argv[1]
        
        data = load_data_from_json(json_file)

        if data is None:
            coefficients, y0, x = get_user_input()
        else:
            coefficients = data.get('coefficients')
            y0 = data.get('y0')
            x = data.get('x')

            if coefficients is None or len(coefficients) < 2:
                print("Error: Coefficients must be provided.")
                sys.exit(1)

            if y0 is None or len(y0) != len(coefficients) - 1:
                print(f"Error: Initial conditions must be provided.")
                sys.exit(1)

            if x is None:
                print("Error: x value must be provided.")
                sys.exit(1)

    else:
        coefficients, y0, x = get_user_input()

    A = convert_to_matrix(coefficients) # The differential equation is converted to matrix form

    eigvals, eigvecs = np.linalg.eig(A) # Extract the eigenvalue and eigenvector using NumPy

    eig_rows, eig_columns = eigvecs.shape

    if eig_rows == eig_columns: # Diagonalizable

        # Construct the P matrix
        P = eigvecs

        # Construct the diagonal matrix
        D = np.diag(eigvals)

        if np.linalg.det(P) == 0:
            print("Warning: Matrix P is singular and cannot be inverted.")
            sys.exit(1)
        else:
            P_inv = np.linalg.inv(P)

        eDx = np.diag(np.exp(D.diagonal() * x))

        # Determine solution
        y_x = P @ eDx @ P_inv @ y0

        y_x_rounded = np.round(y_x, 4)

        order = len(coefficients)
        
        # Display the output
        print("\nCoefficients:")
        for idx, coeff in enumerate(coefficients):
            power = order - idx - 1
            if coeff == 0:
                continue
            if coeff > 0 and idx > 0:
                sign = "+"
            else:
                sign = ""

            if power == 0:
                print(f"{sign}{coeff}y", end=" ")
            elif power == 1:
                print(f"{sign}{coeff}y'", end=" ")
            elif power == 2:
                print(f"{sign}{coeff}y''", end=" ")
            elif power == 3:
                print(f"{sign}{coeff}y'''", end=" ")
            else:
                print(f"{sign}{coeff}y^({power})", end=" ")
        print("= 0")

        print("\nMatrix A (First-Order Representation):")
        print(A)
        
        print("\nMatrix P (Eigenmatrix):")
        print(P)
        
        print("\nMatrix P (inverted):")
        print(P_inv)
        
        print("\nDiagonalized Matrix:")
        print(D)
        
        print("\nInitial conditions:")
        for idx, value in enumerate(y0, start=1):
            print(f"y{idx}(0) = {value}")
        
        print("\nSolution at x =", x)
        print(y_x_rounded)
    else:
        print("Matrix is not diagonalizable.")

if __name__ == "__main__":
    main()
