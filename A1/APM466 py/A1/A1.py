import csv
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any
from numpy import ndarray
from scipy.optimize import fsolve
from sympy import symbols, lambdify
from numpy.linalg import eig

# Import all used data
bond_list = []
coupon_list = []
close_price_list = []
with open("10_database.csv", mode="r") as data:
    data.readline()
    total_list = data.readlines()

date_list = [datetime.date(2022, 1, 10), datetime.date(2022, 1, 11),
             datetime.date(2022, 1, 12), datetime.date(2022, 1, 13),
             datetime.date(2022, 1, 14), datetime.date(2022, 1, 17),
             datetime.date(2022, 1, 18), datetime.date(2022, 1, 19),
             datetime.date(2022, 1, 20), datetime.date(2022, 1, 21)]

last_payment_date_list = [datetime.date(2021, 8, 1), datetime.date(2021, 9, 1),
                          datetime.date(2021, 8, 1), datetime.date(2021, 9, 1),
                          datetime.date(2021, 9, 1), datetime.date(2021, 9, 1),
                          datetime.date(2021, 9, 1), datetime.date(2021, 9, 1),
                          datetime.date(2021, 9, 1), datetime.date(2021, 9, 1)]

period_list = ["22.08", "23.03", "23.08", "24.03", "24.09", "25.03", "25.09",
               "26.03", "26.09", "27.03"]

next_payment_diff_list = [1/6, 2/6, 1/6, 2/6, 2/6, 2/6, 2/6, 2/6, 2/6, 2/6]

for line in total_list:
    line_list = list(line.split(","))
    bond_list.append(line_list[0].strip())
    coupon_list.append(float(line_list[1].strip()))
    small_list = []
    for i in range(2, len(line_list)):
        small_list.append(float(line_list[i].strip()))
    close_price_list.append(small_list)

all_face_value = 100.00

time_difference = []
for day_1 in last_payment_date_list:
    n1 = []
    for day_2 in date_list:
        n1.append(abs((day_2 - day_1).days))
    time_difference.append(n1)


dirty_price_list = []
for i in range(len(time_difference)):
    bond_price = []
    for j in range(len(time_difference[i])):
        bond_dirty_price = (time_difference[i][j] / 365) * coupon_list[i] \
                           + close_price_list[i][j]
        bond_price.append(bond_dirty_price)
    dirty_price_list.append(bond_price)


# Build ytm function to find ytm for each bond dirty price by Newton Method
def ytm_function(coupon_rate: float, y: float, face_value: float,
                 time_to_mature: int, dirty_price: float) -> float:
    coupon = coupon_rate * face_value / 2
    total_sum = face_value / ((1 + y/2) ** time_to_mature) - dirty_price
    for time in range(1, time_to_mature + 1):
        total_sum += coupon / ((1 + y/2) ** time)
    return total_sum


def derivative_ytm_function(coupon_rate: float, y: float, face_value: float,
                            time_to_mature: int) -> float:
    coupon = coupon_rate * face_value / 2
    total_sum = (face_value / ((1 + y/2) ** (time_to_mature + 1))) * (
        - time_to_mature)
    for time in range(1, time_to_mature + 1):
        total_sum += (coupon / ((1 + y/2) ** (time + 1))) * (- time)
    return total_sum


def ytm_newton(coupon_rate: float, face_value: float, time_to_mature: int,
               dirty_price: float) -> Any:
    ytm = 0
    epsilon = 0.0001
    max_iter = 100
    for n in range(0, max_iter):
        f_ytm = ytm_function(coupon_rate, ytm, face_value, time_to_mature,
                             dirty_price)
        if abs(f_ytm) < epsilon:
            return ytm
        d_ytm = derivative_ytm_function(coupon_rate, ytm, face_value,
                                        time_to_mature)
        if d_ytm == 0:
            return "No"
        ytm = ytm - f_ytm / d_ytm
    return "No"


ytm_matrix = []

for i in range(len(dirty_price_list)):
    bond_coupon = coupon_list[i]
    f_v = all_face_value
    bond_ytm = []
    for j in range(len(dirty_price_list[i])):
        t_t_m = j + 1
        bond_ytm.append(ytm_newton(bond_coupon, f_v, t_t_m,
                                   dirty_price_list[i][j]))
    ytm_matrix.append(bond_ytm)

ytm_matrix_transpose = []
for i in range(len(ytm_matrix)):
    day_list = []
    for j in range(len(ytm_matrix)):
        day_list.append(ytm_matrix[j][i])
    ytm_matrix_transpose.append(day_list)

# Plot ytm curve
ytm_plot = plt.figure()
for i in range(len(bond_list)):
    curve = ytm_matrix_transpose[i]
    plt.plot(period_list, curve, label=date_list[i])
plt.title("Yield to Maturity curve of 10 bonds")
plt.xlabel("The number of semiannual periods to mature")
plt.ylabel("Yield to Maturity (in decimal)")
plt.legend(loc="lower right")
plt.show()


# Build spot function to find spot rate for each bond dirty price
def spot_function(coupon_rate: float, dirty_price: float, n_periods: int,
                  face_value: float) -> tuple[ndarray, dict, int, str]:
    r = symbols('r')
    coupon = coupon_rate * face_value / 2
    spot_sum = (coupon + face_value) * np.power(np.e, -r * n_periods)
    for t_t in range(1, n_periods + 1):
        spot_sum += coupon * np.power(np.e, - r * t_t)
    spot_sum -= dirty_price
    func = lambdify(r, spot_sum, modules=["numpy"])
    return fsolve(func, np.array([0.0]))


def first_half_year_spot_function(coupon_rate: float, dirty_price: float,
                                  face_value: float, diff: float) -> float:
    coupon = coupon_rate * face_value / 2
    spot_rate = - (math.log(dirty_price / (coupon + face_value))) / diff
    return spot_rate


spot_matrix = []

for i in range(len(dirty_price_list)):
    bond_coupon = coupon_list[i]
    f_v = all_face_value
    bond_sp = [first_half_year_spot_function(bond_coupon,
                                             dirty_price_list[i][0], f_v,
                                             next_payment_diff_list[i])]
    for j in range(1, len(dirty_price_list[i])):
        periods = j + 1
        spot_r = spot_function(bond_coupon, dirty_price_list[i][j], periods,
                               f_v)
        bond_sp.append(spot_r)
    spot_matrix.append(bond_sp)

spot_matrix_transpose = []
for i in range(len(spot_matrix)):
    day_list = []
    for j in range(len(spot_matrix)):
        day_list.append(spot_matrix[j][i])
    spot_matrix_transpose.append(day_list)

# Plot Spot curve
spot_plot = plt.figure()
for i in range(len(bond_list)):
    curve = spot_matrix_transpose[i]
    plt.plot(period_list, curve, label=date_list[i])
plt.title("Spot curve of 10 bonds")
plt.xlabel("The number of semiannual periods to mature")
plt.ylabel("Spot Rate (in decimal)")
plt.legend(loc="lower right")
plt.show()

# Building forward rate function
forward_matrix = []
for i in range(len(spot_matrix)):
    for_r_list = []
    note_spot = spot_matrix[i][1]
    for_r_list.append(note_spot)
    for j in range(1, len(spot_matrix[i])):
        forward_r = (((1 + spot_matrix[i][j]) ** (j + 1)) / (1 + note_spot)) - 1
        for_r_list.append(forward_r)
    forward_matrix.append(for_r_list)

forward_matrix_transpose = []
for i in range(len(forward_matrix)):
    day_list = []
    for j in range(len(forward_matrix)):
        day_list.append(forward_matrix[j][i])
    forward_matrix_transpose.append(day_list)

# Plot Forward curve
forward_plot = plt.figure()
for i in range(len(bond_list)):
    curve = forward_matrix_transpose[i]
    plt.plot(period_list, curve, label=date_list[i])
plt.title("Forward curve of 10 bonds")
plt.xlabel("The number of semiannual periods to mature")
plt.ylabel("Forward Rate (in decimal)")
plt.legend(loc="lower right")
plt.show()

# Calculate covariance matrix
variable_matrix_1 = []
for i in range(5):
    variable_x = []
    for j in range(9):
        element_x = math.log(abs(ytm_matrix_transpose[i][j + 1] /
                                 ytm_matrix_transpose[i][j]))
        variable_x.append(element_x)
    variable_matrix_1.append(variable_x)

covar_matrix_1 = np.cov(variable_matrix_1)


variable_matrix_2 = []

for i in range(5):
    variable_x = []
    for j in range(9):
        element_x = math.log(forward_matrix_transpose[i][j + 1].astype("float")
                             / forward_matrix_transpose[i][j].astype("float"))
        variable_x.append(element_x)
    variable_matrix_2.append(variable_x)

covar_matrix_2 = np.cov(variable_matrix_2)

cov_matrix_1 = pd.DataFrame(covar_matrix_1, columns=["X1", "X2", "X3", "X4",
                                                     "X5"])

cov_matrix_2 = pd.DataFrame(covar_matrix_2, columns=["X1", "X2", "X3", "X4",
                                                     "X5"])

# Calculate eigenvalues and eigenvectors with respect to covariance matrix
eigenvalue_1, eigenvector_1 = eig(covar_matrix_1)
eigenvalue_2, eigenvector_2 = eig(covar_matrix_2)
