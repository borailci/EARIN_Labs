import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Our objective function: f(x, y) = 2*sin(x) + 3*cos(y)
def function(x, y):
    return 2 * np.sin(x) + 3 * np.cos(y)


def gradient_descent(initial_guess, learning_rate, tol=1e-6, max_iter=1000):
    """
    Performs gradient descent to minimize the function f(x, y).

    Parameters:
    - initial_guess: List or tuple of two numbers, the starting point [x, y].
    - learning_rate: The step size multiplier.
    - tol: Tolerance for stopping (when the norm of the gradient is very small).
    - max_iter: Maximum number of iterations to perform.

    Returns:
    - final_point: The approximation of the minimum as a tuple (x, y).
    - iterations: Number of iterations performed.
    - path: List of points (x, y) visited during the descent (for visualization).
    """
    x, y = initial_guess
    iterations = 0
    path = [(x, y)]

    for _ in range(max_iter):
        grad_x = 2 * np.cos(x)
        grad_y = -3 * np.sin(y)
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        if grad_norm < tol:
            break
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
        path.append((x, y))
        iterations += 1

    return (x, y), iterations, path


def create_visualization(ax1, ax2, learning_rate, initial_guess, iterations, path=None):
    """
    Creates two 3D plots of the function f(x, y) over the range [-5, 5] on the given axes.
    Each plot uses a different viewing angle so that the descent path can be seen more clearly.
    """
    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = function(X, Y)

    if path is not None:
        path_arr = np.array(path)
        start_point = path_arr[0]
        end_point = path_arr[-1]
        start_f = function(start_point[0], start_point[1])
        end_f = function(end_point[0], end_point[1])

    # --- First subplot ---
    surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    if path is not None:
        ax1.plot(
            path_arr[:, 0],
            path_arr[:, 1],
            function(path_arr[:, 0], path_arr[:, 1]),
            color="r",
            marker="o",
            markersize=4,  # Smaller markers
            label="Descent Path",
        )
        ax1.scatter(
            start_point[0], start_point[1], start_f, color="g", s=80, label="Start"
        )
        ax1.scatter(end_point[0], end_point[1], end_f, color="b", s=80, label="End")

        # Adjust text positioning and make it more compact
        ax1.text(
            start_point[0],
            start_point[1],
            start_f + 0.5,  # Offset in z direction
            f"Start: ({start_point[0]:+.2f}, {start_point[1]:+.2f})",
            color="g",
            fontsize=8,
        )
        ax1.text(
            end_point[0],
            end_point[1],
            end_f + 0.5,  # Offset in z direction
            f"End: ({end_point[0]:+.2f}, {end_point[1]:+.2f})",
            color="b",
            fontsize=8,
        )
        ax1.legend(fontsize=8, loc="upper right")

    title = f"Angle 1 (Initial: [{initial_guess[0]:+.2f}, {initial_guess[1]:+.2f}] | LR: {learning_rate:+.2f} | Steps: {iterations})"
    ax1.set_title(title, fontsize=9, pad=10)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x, y)")
    ax1.view_init(elev=30, azim=-60)

    # --- Second subplot ---
    surf2 = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
    if path is not None:
        ax2.plot(
            path_arr[:, 0],
            path_arr[:, 1],
            function(path_arr[:, 0], path_arr[:, 1]),
            color="r",
            marker="o",
            markersize=4,  # Smaller markers
            label="Descent Path",
        )
        ax2.scatter(
            start_point[0], start_point[1], start_f, color="g", s=80, label="Start"
        )
        ax2.scatter(end_point[0], end_point[1], end_f, color="b", s=80, label="End")

        # Adjust text positioning and make it more compact
        ax2.text(
            start_point[0],
            start_point[1],
            start_f + 0.5,  # Offset in z direction
            f"Start: ({start_point[0]:+.2f}, {start_point[1]:+.2f})",
            color="g",
            fontsize=8,
        )
        ax2.text(
            end_point[0],
            end_point[1],
            end_f + 0.5,  # Offset in z direction
            f"End: ({end_point[0]:+.2f}, {end_point[1]:+.2f})",
            color="b",
            fontsize=8,
        )
        ax2.legend(fontsize=8, loc="upper right")

    title = f"Angle 2 (Initial: [{initial_guess[0]:+.2f}, {initial_guess[1]:+.2f}] | LR: {learning_rate:+.2f} | Steps: {iterations})"
    ax2.set_title(title, fontsize=9, pad=10)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("f(x, y)")
    ax2.view_init(elev=60, azim=20)

    return surf1, surf2


def visualize_multi_page(results_list):
    """
    Create a multi-page figure with all visualizations.
    Navigate using left/right arrow keys.

    Parameters:
    - results_list: List of tuples, each containing
      (learning_rate, initial_guess, iterations, path)
    """
    # Increase figure size to provide more space
    fig = plt.figure(figsize=(16, 7))
    n_pages = len(results_list)
    axes_list = []
    surfs = []

    # Create all subplots (but hide them initially)
    for i in range(n_pages):
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        axes_list.append((ax1, ax2))
        surf1, surf2 = create_visualization(ax1, ax2, *results_list[i])
        surfs.append((surf1, surf2))

        # Hide all pages except the first one
        if i > 0:
            ax1.set_visible(False)
            ax2.set_visible(False)

    # Add page number indicator
    page_text = fig.text(
        0.5,
        0.01,
        f"Page 1/{n_pages} (use ← → arrows to navigate)",
        ha="center",
        va="center",
        fontsize=12,
    )

    current_page = [0]  # Using list for mutable reference

    def key_event(event):
        if event.key == "right" and current_page[0] < n_pages - 1:
            # Hide current page
            axes_list[current_page[0]][0].set_visible(False)
            axes_list[current_page[0]][1].set_visible(False)
            # Go to next page
            current_page[0] += 1
            # Show new page
            axes_list[current_page[0]][0].set_visible(True)
            axes_list[current_page[0]][1].set_visible(True)
            page_text.set_text(
                f"Page {current_page[0]+1}/{n_pages} (use ← → arrows to navigate)"
            )
            plt.draw()
        elif event.key == "left" and current_page[0] > 0:
            # Hide current page
            axes_list[current_page[0]][0].set_visible(False)
            axes_list[current_page[0]][1].set_visible(False)
            # Go to previous page
            current_page[0] -= 1
            # Show new page
            axes_list[current_page[0]][0].set_visible(True)
            axes_list[current_page[0]][1].set_visible(True)
            page_text.set_text(
                f"Page {current_page[0]+1}/{n_pages} (use ← → arrows to navigate)"
            )
            plt.draw()

    # Add colorbar for the current visible page
    cb1 = fig.colorbar(surfs[0][0], ax=axes_list[0][0], shrink=0.5, aspect=10)
    cb2 = fig.colorbar(surfs[0][1], ax=axes_list[0][1], shrink=0.5, aspect=10)

    fig.canvas.mpl_connect("key_press_event", key_event)

    # Add more spacing between subplots and adjust margins
    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.12,  # More room for page indicator
        wspace=0.3,  # More horizontal space between subplots
        top=0.9,  # More room for titles
    )

    return fig


# Example usage with two different initial guesses:
initial_guess_1 = [2.0, 2.0]
initial_guess_1_2 = [-2.0, 1.0]
learning_rate_1 = 0.1

# Run gradient descent for same learning rate different initial guesses:
minimum_1, iterations_1, path_1 = gradient_descent(initial_guess_1, learning_rate_1)
minimum_1_2, iterations_1_2, path_1_2 = gradient_descent(
    initial_guess_1_2, learning_rate_1
)

# Example usage with two different learning rates:
initial_guess_2 = [2.0, 2.0]
initial_guess_2_2 = [2.0, 2.0]
learning_rate_2 = 1
learning_rate_2_2 = 0.5

# Run gradient descent for different learning rate same initial guesses:
minimum_2, iterations_2, path_2 = gradient_descent(initial_guess_2, learning_rate_2)
minimum_2_2, iterations_2_2, path_2_2 = gradient_descent(
    initial_guess_2_2, learning_rate_2_2
)

# Collect all results to be visualized
results = [
    (learning_rate_1, initial_guess_1, iterations_1, path_1),
    (learning_rate_1, initial_guess_1_2, iterations_1_2, path_1_2),
    (learning_rate_2, initial_guess_2, iterations_2, path_2),
    (learning_rate_2_2, initial_guess_2_2, iterations_2_2, path_2_2),
]

# Create a multi-page figure with all visualizations
fig = visualize_multi_page(results)
plt.show()

# Print the final results to the console
print(
    f"Minimum approximation with initial guess {initial_guess_1}: {minimum_1}, Steps: {iterations_1}"
)
print(
    f"Minimum approximation with initial guess {initial_guess_1_2}: {minimum_1_2}, Steps: {iterations_1_2}"
)
print(
    f"Minimum approximation with initial guess {initial_guess_2}: {minimum_2}, Steps: {iterations_2}"
)
print(
    f"Minimum approximation with initial guess {initial_guess_2_2}: {minimum_2_2}, Steps: {iterations_2_2}"
)
