import numpy as np
import matplotlib
from matplotlib import pyplot as plt, cm
matplotlib.use('Qt5Agg')


def create_graph(line, title: str, ax):
    ax.set_xlim(0, 7000)
    ax.set_ylim(0, 3000)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    x, y = line.xy
    ax.plot(x, y, marker='o', label='Mars Surface')
    ax.legend()
    ax.grid(True)


def plot_mean_rewards(rewards, ax):
    ax.plot(rewards, color='red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Progress - average rewards throughout entire horizon')
    plt.pause(0.001)


def plot_terminal_state_rewards(rewards, ax):
    ax.plot(rewards, color='red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Progress - terminal state reward')
    plt.pause(0.001)


def display_graph(trajectories: list[list[tuple[float, float]]], id_lines_color: int, ax):
    cmap = cm.get_cmap('Set1')
    color = cmap(id_lines_color % 9)

    # Clear previous trajectories
    for line in ax.lines:
        if line.get_label() != 'Mars Surface':
            line.remove()

    for trajectory in trajectories:
        x_values = [point[0] for point in trajectory]
        y_values = [point[1] for point in trajectory]
        ax.plot(x_values, y_values, marker='o',
                markersize=2, color=color, label=f'Rocket {id_lines_color}')
        plt.pause(0.001)
