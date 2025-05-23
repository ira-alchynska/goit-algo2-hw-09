import random
import math


def sphere_function(x):
    """
    Sphere function for optimization.

    Args:
        x (list): A list of numbers representing a point in the search space.

    Returns:
        float: The value of the sphere function at the given point.
    """
    return sum(xi**2 for xi in x)


# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    """
    Hill Climbing optimization algorithm.

    Args:
        func (callable): The objective function to minimize.
        bounds (list of tuples): Bounds for each dimension as (lower_bound, upper_bound).
        iterations (int): Maximum number of iterations. Default is 1000.
        epsilon (float): Minimum improvement threshold to stop the algorithm. Default is 1e-6.

    Returns:
        tuple: A tuple containing the best solution found and its value.
    """
    num_dimensions = len(bounds)
    current_x = [random.uniform(lb, ub) for lb, ub in bounds]
    current_val = func(current_x)
    step_size = [(ub - lb) * 0.1 for lb, ub in bounds]

    for _ in range(iterations):
        neighbor_x = []
        for d in range(num_dimensions):
            lb, ub = bounds[d]
            delta = random.uniform(-step_size[d], step_size[d])
            xi = current_x[d] + delta
            xi = min(max(xi, lb), ub)
            neighbor_x.append(xi)
        neighbor_val = func(neighbor_x)

        if neighbor_val < current_val:
            improvement = abs(current_val - neighbor_val)
            current_x, current_val = neighbor_x, neighbor_val

            if improvement < epsilon:
                break

    return current_x, current_val


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    """
    Random Local Search optimization algorithm.

    Args:
        func (callable): The objective function to minimize.
        bounds (list of tuples): Bounds for each dimension as (lower_bound, upper_bound).
        iterations (int): Maximum number of iterations. Default is 1000.
        epsilon (float): Minimum improvement threshold to stop the algorithm. Default is 1e-6.

    Returns:
        tuple: A tuple containing the best solution found and its value.
    """

    best_x = [random.uniform(lb, ub) for lb, ub in bounds]
    best_val = func(best_x)

    for _ in range(iterations):
        candidate_x = [random.uniform(lb, ub) for lb, ub in bounds]
        candidate_val = func(candidate_x)
        if candidate_val < best_val:
            improvement = best_val - candidate_val
            best_x, best_val = candidate_x, candidate_val
            if improvement < epsilon:
                break

    return best_x, best_val


# Simulated Annealing
def simulated_annealing(
    func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6
):
    """
    Simulated Annealing optimization algorithm.

    Args:
        func (callable): The objective function to minimize.
        bounds (list of tuples): Bounds for each dimension as (lower_bound, upper_bound).
        iterations (int): Maximum number of iterations. Default is 1000.
        temp (float): Initial temperature for the annealing process. Default is 1000.
        cooling_rate (float): Cooling rate for temperature reduction. Default is 0.95.
        epsilon (float): Minimum improvement threshold to stop the algorithm. Default is 1e-6.

    Returns:
        tuple: A tuple containing the best solution found and its value.
    """
    num_dimensions = len(bounds)
    current_x = [random.uniform(lb, ub) for lb, ub in bounds]
    current_val = func(current_x)
    best_x, best_val = list(current_x), current_val
    step_size = [(ub - lb) * 0.1 for lb, ub in bounds]
    T = temp

    for _ in range(iterations):
        if T < epsilon:
            break

        neighbor_x = []
        for d in range(num_dimensions):
            lb, ub = bounds[d]
            delta = random.uniform(-step_size[d], step_size[d])
            xi = current_x[d] + delta
            xi = min(max(xi, lb), ub)
            neighbor_x.append(xi)
        neighbor_val = func(neighbor_x)
        delta = neighbor_val - current_val

        if delta < 0 or random.random() < math.exp(-delta / T):
            prev_val = current_val
            current_x, current_val = neighbor_x, neighbor_val

            if current_val < best_val:
                best_x, best_val = list(current_x), current_val

            if abs(prev_val - current_val) < epsilon:
                break

        T *= cooling_rate

    return best_x, best_val


if __name__ == "__main__":
    """
    Main execution block to run optimization algorithms on the Sphere function.
    """
    bounds = [(-5, 5), (-5, 5)]

    # Execute algorithms
    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    print("Solution:", hc_solution, "Value:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    print("Solution:", rls_solution, "Value:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
    print("Solution:", sa_solution, "Value:", sa_value)
