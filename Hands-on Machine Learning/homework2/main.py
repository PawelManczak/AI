import numpy as np
import matplotlib.pyplot as plt

# funkcja celu
def obj_func(x, y):
    return np.sin(x)*np.sin(x) + np.cos(y)*np.sin(x)

# metoda gradientowa z wyżarzaniem
def simulated_annealing(x0, T0, alpha, beta, gamma, n_iter):
    x = x0
    T = T0
    fval = obj_func(*x)
    x_hist = [x]
    fval_hist = [fval]

    for i in range(n_iter):
        # wygeneruj nowe x w okolicy obecnego x
        x_new = x + beta*np.random.normal(size=2)
        # oblicz różnicę wartości funkcji celu
        delta_f = obj_func(*x_new) - fval

        if delta_f < 0:
            # akceptuj nowe x
            x = x_new
            fval = obj_func(*x)
        else:
            # prawdopodobieństwo odrzucenia nowego x
            p = np.exp(-delta_f/(gamma*T))
            if np.random.uniform() < p:
                x = x_new

        # zmniejsz temperaturę
        T = alpha*T

        # zapisz historię
        x_hist.append(x)
        fval_hist.append(fval)

    return x, fval, x_hist, fval_hist

# metoda szukania przypadkowego
def random_search(x0, n_iter):
    x = x0
    fval = obj_func(*x)
    x_hist = [x]
    fval_hist = [fval]

    for i in range(n_iter):
        # wygeneruj nowe x
        x_new = np.random.uniform(low=-10, high=10, size=2)
        # oblicz wartość funkcji celu
        fval_new = obj_func(*x_new)

        if fval_new < fval:
            # akceptuj nowe x
            x = x_new
            fval = fval_new

        # zapisz historię
        x_hist.append(x)
        fval_hist.append(fval)

    return x, fval, x_hist, fval_hist

# początkowe parametry
x0 = np.array([-7.0, 7.0])
T0 = 100
alpha = 0.99
beta = 0.1
gamma = 1
n_iter = 5000

# metoda gradientowa z wyżarzaniem
x_sa, fval_sa, x_hist_sa, fval_hist_sa = simulated_annealing(x0, T0, alpha, beta, gamma, n_iter)

# metoda szukania przypadkowego
x_rs, fval_rs, x_hist_rs, fval_hist_rs = random_search(x0, n_iter)

# wykres
fig, ax = plt.subplots(figsize=(10, 8))

# funkcja celu
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = obj_func(X, Y)
ax.contour(X, Y, Z, levels=50, cmap='coolwarm')

# metoda gradientowa z wyżarzaniem
ax.plot(x_hist_sa[0][0], x_hist_sa[0][1], 'go', label='start metoda gradientowa z wyżarzaniem')
for i in range(len(x_hist_sa)-1):
    ax.plot([x_hist_sa[i][0], x_hist_sa[i+1][0]], [x_hist_sa[i][1], x_hist_sa[i+1][1]], 'g-', alpha=0.5)

# metoda szukania przypadkowego
ax.plot(x_hist_rs[0][0], x_hist_rs[0][1], 'bo', label='start metoda szukania przypadkowego')
for i in range(len(x_hist_rs)-1):
    ax.plot([x_hist_rs[i][0], x_hist_rs[i+1][0]], [x_hist_rs[i][1], x_hist_rs[i+1][1]], 'b-', alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()