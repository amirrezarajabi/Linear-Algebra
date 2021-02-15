#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
concatenate B, C
if shape of [B|C] = 2,3
"""
def copy_mat(B, C):
    A = np.zeros((2,3))
    A[0,0] = B[0,0]
    A[0,1] = B[0,1]
    A[1,0] = B[1,0]
    A[1,1] = B[1,1]
    A[0,2] = C[0]
    A[1,2] = C[1]
    return A

"""
concatenate B, C
if shape of [B|C] = 3,4
"""
def copy_mat2(B, C):
    A = np.zeros((3,4))
    A[0,0] = B[0,0]
    A[1,0] = B[1,0]
    A[2,0] = B[2,0]
    A[0,1] = B[0,1]
    A[1,1] = B[1,1]
    A[2,1] = B[2,1]
    A[0,2] = B[0,2]
    A[1,2] = B[1,2]
    A[2,2] = B[2,2]
    A[0,3] = C[0]
    A[1,3] = C[1]
    A[2,3] = C[2]
    return A

"""
code for reduce row echelon form to solve linear system
"""
def Interchange(M, i ,j):
    m = M.shape[1]
    for k in range(m):
        tmp = M[i, k]
        M[i, k] = M[j, k]
        M[j, k] = tmp

def Scaling(M , i, c):
    m = M.shape[1]
    for k in range(m):
        M[i, k] = M[i, k] * c

def Replacement(M , i, j, c):
    m = M.shape[1]
    for k in range(m):
        M[i, k] = M[i, k] + M[j, k] * c


def zero_column_forward_phase (M , i, j):
    value = M[i, j]
    Scaling(M, i, 1/value)
    n = M.shape[0]
    x = i + 1
    while (x < n):
        Replacement(M, x, i, -M[x, j])
        x = x + 1
def zero_column_backward_phase (M , i, j):
    value = M[i, j]
    n = M.shape[0]
    x = i - 1
    while (x > -1):
        Replacement(M, x, i, -M[x, j])
        x = x - 1

def check_zero_column(M, i ,j):
    x = i + 1
    while(x < M.shape[0]):
        if (M[x, j] != 0):
            return 0
        x = x + 1
    return 1

def find_pivot(M, i, j):
    if(check_zero_column(M, i, j) == 1):
        return -1
    x = i + 1
    while(x < M.shape[0]):
        if(M[x, j] != 0):
            return x
        x = x + 1


def reduced_row_echelon_form(M):
    a = M.astype(float)
    x = 0
    y = 0
    x_pivots = []
    y_pivots = []
    while(x < a.shape[0] and y < a.shape[1]):
        if(a[x, y] != 0):
            zero_column_forward_phase(a, x, y)
            x_pivots.append(x)
            y_pivots.append(y)
            x = x + 1
            y = y + 1
        else:
            if(check_zero_column(a, x, y) == 1):
                y = y + 1
            else:
                xx = x
                Interchange(a, xx, find_pivot(a, x, y))

    x_pivots.reverse()
    y_pivots.reverse()
    for i in range(len(x_pivots)):
        zero_column_backward_phase(a, x_pivots[i], y_pivots[i])

    return a, x_pivots, y_pivots
"""
end of reduced echelon form
"""

#import file
google = pd.read_csv("GOOGL.csv")

def main_program(part):

    #part = "Open" or "High" or "Close" "Low"

    #y of model
    G_O = google[part]

    #split model to train and test
    G_O_train = G_O[0:google.shape[0] - 10]
    G_O_test = G_O[-10:google.shape[0]]

    #create X, X2 for models and x for plot
    x = np.ones(((google.shape[0], 3)))
    X = np.ones((google.shape[0] - 10, 2))
    X2 = np.ones((google.shape[0] - 10, 3))
    for i in range(X.shape[0]):
        X[i,1] = i + 1
        X2[i,1] = i + 1
        X2[i,2] = (i + 1)*(i + 1)
    for i in range(x.shape[0]):
        x[i,1] = i + 1
        x[i,2] = (i + 1)*(i + 1)
        
    #create X for testing model
    X_test = np.ones((10, 2))
    X2_test = np.ones((10, 3))
    p = X[X.shape[0]-1,1] - 1
    for i in range(X_test.shape[0]):
        X_test[i,1] = i + 1 + p
        X2_test[i,1] = i + 1 + p
        X2_test[i,2] = (i + 1 + p)*(i + 1 + p)

    """
    transpose(X) * X * B = transpose(X) * y

    A is for transpose(X) * X
    C is for transpose(X) * y
    """
    A = np.dot(X.T, X)
    A2 = np.dot(X2.T, X2)

    C_O = np.dot(X.T, G_O_train)

    C_O2 = np.dot(X2.T, G_O_train)

    #L_S for linear system
    L_S_O = copy_mat(A, C_O)
    L_S_O2 = copy_mat2(A2, C_O2)

    #find B
    B_O = reduced_row_echelon_form(L_S_O)
    B_O2 = reduced_row_echelon_form(L_S_O2)

    B_O = B_O[0]
    B_O2 = B_O2[0]

    B_O = B_O[:, 2]
    B_O2 = B_O2[:, 3]

    print("X")
    for i in range(10):
        real = G_O_test[i+3009]
        xx = X_test[i,1]*B_O[1] + B_O[0]
        print("DAY " + str(i + 1))
        print("actual price: " + str(real))
        print("calculated price: " + str(xx))
        print("error: " + str(real - xx))
        print("\n#################################")

    print("\n\nX²\n\n")
        
    #print error
    for i in range(10):
        real = G_O_test[i+3009]
        x2 = X2_test[i,1]*B_O2[1] + B_O2[0] +X2_test[i,2]*B_O2[2]
        print("DAY " + str(i + 1))
        print("actual price: " + str(real))
        print("calculated price: " + str(x2))
        print("error: " + str(real - x2))
        print("\n#################################")

    
    print("X² is beter than X")

    #show plot
    plt.figure()
    plt.plot(x[:,1], G_O, 'g-')
    plt.plot(x[:,1],B_O2[0]+B_O2[1]*x[:,1]+B_O2[2]*x[:,2],color='r')
    plt.legend(["actual","X²"])
    plt.title(part + " PRICE")
    plt.xlabel("DAY")
    plt.ylabel("PRICE")
    plt.show()

main_program("Open")

### /\ /V\ [] [Z [Z (≡ ¯/_ /\ [Z /\ _] /\ !3 [] ###

