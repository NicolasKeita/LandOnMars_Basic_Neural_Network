import numpy as np
import matplotlib
from matplotlib import pyplot as plt, cm
matplotlib.use('Qt5Agg')


def create_graph(line: np.ndarray, title: str, ax, path_to_the_landing_spot):
    ax.set_xlim(0, 7000)
    ax.set_ylim(0, 3000)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    x, y = line[:, 0], line[:, 1]
    ax.plot(x, y, marker='o', label='Mars Surface')
    # Plot the path to the landing spot
    path_x, path_y = path_to_the_landing_spot[:, 0], path_to_the_landing_spot[:, 1]
    ax.plot(path_x, path_y, marker='x', linestyle='--', label='Path to Landing Spot')

    ax.legend()
    ax.grid(True)



def plot_terminal_state_rewards(rewards, ax, window_size=100):
    ax.clear()
    ax.plot(rewards, color='red', label='Episode Reward')

    # Calculate the mean over a specified window
    if len(rewards) >= window_size:
        means = [sum(rewards[i - window_size + 1:i + 1]) / window_size for i in range(window_size - 1, len(rewards))]
        ax.plot(range(window_size - 1, len(rewards)), means, label=f'Mean over {window_size} episodes', color='blue')

        # Annotate only the last mean value on the graph
        last_mean_index = len(rewards) - 1
        last_mean_val = means[-1]
        ax.annotate(f'{last_mean_val:.2f}', (last_mean_index, last_mean_val), textcoords="offset points", xytext=(0, 5), ha='center')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Progress - Terminal State Reward')
    ax.legend()
    plt.pause(0.001)


def display_graph(trajectories, id_lines_color: int, ax):
    cmap = cm.get_cmap('Set1')
    color = cmap(id_lines_color % 9)

    # Clear previous trajectories
    for line in ax.lines:
        if line.get_label() == 'Rocket':
            line.remove()

    x_values = [trajectory[0] for trajectory in trajectories]
    y_values = [trajectory[1] for trajectory in trajectories]

    ax.plot(x_values, y_values, marker='o',
            markersize=2, color=color, label='Rocket')
    plt.pause(0.001)
