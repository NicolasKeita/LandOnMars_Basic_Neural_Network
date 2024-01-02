import matplotlib
from matplotlib import pyplot as plt, cm
matplotlib.use('Qt5Agg')


def create_graph(line, title: str, ax, path_to_the_landing_spot):
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


def display_graph(trajectories, id_lines_color: int, ax):
    cmap = cm.get_cmap('Set1')
    color = cmap(id_lines_color % 9)

    # Clear previous trajectories
    for line in ax.lines:
        if line.get_label() == 'Rocket':
            line.remove()

    x_values = [trajectory[0] for trajectory in trajectories]
    y_values = [trajectory[1] for trajectory in trajectories]

    ax.plot(x_values, y_values, marker='o', markersize=2, color=color, label='Rocket')
    plt.pause(0.001)

