import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def rastrigin_func(x, y):
    return 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))
def rastrigin_grad(x, y):
    return 2*x + 20*np.pi*np.sin(2*np.pi*x), 2*y + 20*np.pi*np.sin(2*np.pi*y)
def plot_rastrigin_3D():
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    x, y = np.meshgrid(x, y)
    z = rastrigin_func(x, y)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Rastrigin Function')

    plt.show()

plot_rastrigin_3D()


def random_search(func, low=-5, high=5, iter=10):
    x_min, y_min = np.random.uniform(low, high, 2)
    f_min = func(x_min, y_min)
    history = [(x_min, y_min, f_min)]

    for _ in range(iter):
        x, y = np.random.uniform(low, high, 2)
        f_val = func(x, y)
        if f_val < f_min:
            f_min = f_val
            x_min, y_min = x, y
            history.append((x, y, f_val))

    return x_min, y_min, history


def gradient_descent(func, grad, low=-5, high=5, alpha=0.01, iter=10):
    x_min, y_min = np.random.uniform(low, high, 2)
    f_min = func(x_min, y_min)
    history = [(x_min, y_min, f_min)]

    for _ in range(iter):
        grad_x, grad_y = grad(x_min, y_min)
        x_min, y_min = x_min - alpha * grad_x, y_min - alpha * grad_y
        f_val = func(x_min, y_min)
        if f_val < f_min:
            f_min = f_val
        history.append((x_min, y_min, f_val))

    return x_min, y_min, history


def simulated_annealing(func, low=-5, high=5, T=1000.0, cool_rate=0.5, iter=10):
    x_min, y_min = np.random.uniform(low, high, 2)
    f_min = func(x_min, y_min)
    history = [(x_min, y_min, f_min)]

    for i in range(iter):
        T *= cool_rate
        x, y = np.random.uniform(low, high, 2)
        f_val = func(x, y)
        delta = f_val - f_min
        if delta < 0 or np.exp(-delta / T) > np.random.rand():
            x_min, y_min = x, y
            f_min = f_val
            history.append((x_min, y_min, f_val))

    return x_min, y_min, history


def plot_function_history(history, title):
    fig, ax = plt.subplots()
    ax.plot([h[2] for h in history], 'o-', label=title)
    ax.legend()
    plt.show()


def plot_search_history_with_background(history, title):
    fig, ax = plt.subplots()

    # tworzenie siatki punktów
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    x, y = np.meshgrid(x, y)

    # obliczanie wartości funkcji na siatce punktów
    z = rastrigin_func(x, y)

    # rysowanie wartości funkcji jako tła
    c = ax.pcolormesh(x, y, z, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax)

    # rysowanie trajektorii poszukiwań
    x = [h[0] for h in history]
    y = [h[1] for h in history]
    ax.plot(x, y, 'o-', color='r')

    ax.set_title(title)
    plt.show()

# Testowanie
iter = 30
x_min_rand, y_min_rand, history_rand = random_search(rastrigin_func, iter=iter)
x_min_grad, y_min_grad, history_grad = gradient_descent(rastrigin_func, rastrigin_grad, iter=iter)
x_min_anneal, y_min_anneal, history_anneal = simulated_annealing(rastrigin_func, iter=iter)

print("random: ", rastrigin_func(x_min_rand, y_min_rand))
print("gradient: ", rastrigin_func(x_min_grad, y_min_grad))
print("annealing: ", rastrigin_func(x_min_anneal, y_min_anneal))

plot_search_history_with_background(history_rand, 'Random Search')
plot_search_history_with_background(history_grad, 'Gradient Descent')
plot_search_history_with_background(history_anneal, 'Simulated Annealing')

# Drawing plot for function value change
plot_function_history(history_rand, 'Random Search')
plot_function_history(history_grad, 'Gradient Descent')
plot_function_history(history_anneal, 'Simulated Annealing')




# Drawing plot for function value change
fig, ax = plt.subplots()
ax.plot([h[2] for h in history_rand], 'o-', label='Random Search')
ax.plot([h[2] for h in history_grad], 'o-', label='Gradient Descent')
ax.plot([h[2] for h in history_anneal], 'o-', label='Simulated Annealing')
ax.legend()
plt.show()
