import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial, linalg


def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)


def GP(X1, y1, X2, kernel_func, noise=None):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1), 
    and the prior kernel function.

    Return: mean, covariance
    """
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1)

    if noise is not None:
        Σ11 = Σ11 + ((noise**2) * np.eye(X1.size))
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2)
    print(Σ12.shape)
    solved = linalg.solve(Σ11, Σ12, assume_a='pos').T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2


def plot_posterior():

    # Define the true function that we want to regress on
    f_sin = lambda x: (np.sin(x)).flatten()

    n1 = 8
    n2 = 75
    ny = 5
    domain = (-6, 6)

    X1 = np.random.uniform(domain[0]+2, domain[1]-2, size=(n1, 1))
    y1 = f_sin(X1)
    # Predict points at uniform spacing to capture function
    X2 = np.linspace(domain[0], domain[1], n2).reshape(-1, 1)
    # Compute posterior mean and covariance
    μ2, Σ2 = GP(X1, y1, X2, exponentiated_quadratic)
    print(Σ2.shape, "model sigma")
    # Compute the standard deviation at the test points to be plotted
    var = np.sqrt(np.diag(Σ2))

    # Draw some samples of the posterior
    y2 = np.random.multivariate_normal(mean=μ2, cov=Σ2, size=ny)

    # Plot the postior distribution and some samples
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 6))
    # Plot the distribution of the function (mean, covariance)
    ax1.plot(X2, f_sin(X2), 'b--', label='$sin(x)$')
    ax1.fill_between(X2.flat, μ2-2*var, μ2+2*var, color='red', 
                    alpha=0.15, label='$2 \sigma_{2|1}$')
    ax1.plot(X2, μ2, 'r-', lw=2, label='$\mu_{2|1}$')
    ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)
    ax1.set_title('Distribution of posterior and prior data.')
    ax1.axis([domain[0], domain[1], -3, 3])
    ax1.legend()
    ax2.plot(X2, y2.T, '-')
    ax2.set_xlabel('$x$', fontsize=13)
    ax2.set_ylabel('$y$', fontsize=13)
    ax2.set_title('5 different function realizations from posterior')
    ax1.axis([domain[0], domain[1], -3, 3])
    ax2.set_xlim([-6, 6])
    plt.tight_layout()
    plt.savefig("plot_posterior.jpg")






def plot_1():
    """
    5 different function realizations at 41 points sampled from a gaussian
    process with a exponential quadratic kernel.
    """
    nb_of_samples = 41
    number_of_functions = 5
    X = np.expand_dims(np.linspace(-4, 4, nb_of_samples), 1)
    sig = exponentiated_quadratic(X, X)

    # Draw samples from the prior at our data points.
    ys = np.random.multivariate_normal(
        mean=np.zeros(nb_of_samples), cov=sig, 
        size=number_of_functions)

    # Plot the sampled functions
    plt.figure(figsize=(6, 4))
    for i in range(number_of_functions):
        plt.plot(X, ys[i], linestyle='-', marker='o', markersize=3)
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title((
        '5 different function realizations at 41 points\n'
        'sampled from a Gaussian process with exponentiated quadratic kernel'))
    plt.xlim([-4, 4])
    plt.savefig("Plot_1.jpg")

if __name__ == "__main__":
    plot_posterior()