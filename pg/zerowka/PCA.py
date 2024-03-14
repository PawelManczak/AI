import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dane wejściowe
data = [[12, 140, 45],
        [13, 155, 58],
        [15, 162, 67],
        [17, 165, 75],
        [21, 167, 82],
        [23, 167, 90]]


# Definicja funkcji pomocniczych
def mean(data):
    return sum(data) / len(data)


def std_dev(data):
    m = mean(data)
    return (sum((x - m) ** 2 for x in data) / len(data)) ** 0.5


def covariance(x, y):
    n = len(x)
    return sum((x[i] - mean(x)) * (y[i] - mean(y)) for i in range(n)) / (n - 1)


def covariance_matrix(data):
    data = list(map(list, zip(*data)))  # Transpose data
    cov_matrix = [[covariance(col1, col2) for col1 in data] for col2 in data]
    return cov_matrix


def eigenvalue(A, v):
    n = len(A)
    Av = [0] * n
    for i in range(n):
        for j in range(n):
            Av[i] += A[i][j] * v[j]
    return sum(Av) / sum(v)


def eigenvector(A, tol=0.00001):
    n = len(A)
    start_vector = [1] * n
    eigenvalue1 = eigenvalue(A, start_vector)
    eigenvalue2 = 0
    while abs(eigenvalue1 - eigenvalue2) > tol:
        eigenvalue2 = eigenvalue1
        start_vector = [sum(A[i][j] * start_vector[j] for j in range(n)) for i in range(n)]
        norm = sum(x ** 2 for x in start_vector) ** 0.5
        start_vector = [x / norm for x in start_vector]
        eigenvalue1 = eigenvalue(A, start_vector)
    return start_vector


# Wykres danych wejściowych 3D
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(131, projection='3d')
ax.scatter(*zip(*data))
ax.set_title('Input data')

# Definicja funkcji pomocniczych
# [Kod funkcji pomocniczych]

# Standaryzacja danych
normalized_data = [[(x - mean(col)) / std_dev(col) for x in col] for col in zip(*data)]

# Wykres danych znormalizowanych 3D
ax = fig.add_subplot(132, projection='3d')
ax.scatter(*normalized_data)
ax.set_title('Normalized data')

# Obliczenie macierzy kowariancji
cov_matrix = covariance_matrix(normalized_data)

# Obliczenie wektorów własnych macierzy kowariancji
eigenvectors = [eigenvector(cov_matrix) for _ in range(len(cov_matrix))]

# Transformacja danych
transformed_data = [[sum(eigenvectors[i][j] * normalized_data[j][k] for j in range(len(normalized_data))) for i in range(len(eigenvectors))] for k in range(len(normalized_data[0]))]
transformed_data = list(map(list, zip(*transformed_data)))

# Wykres danych po transformacji PCA 2D
ax = fig.add_subplot(133)
ax.scatter(*transformed_data[:2])
ax.set_title('PCA Transformed data')

plt.show()
