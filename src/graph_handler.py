from matplotlib import pyplot as plt, cm


def create_graph(points: list, title: str, ax):
    ax.set_xlim(0, 7000)
    ax.set_ylim(0, 3000)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.plot(*zip(*[(point[0], point[1]) for point in points]), marker='o', label='Mars Surface')
    ax.legend()
    ax.grid(True)
    # ax.draw()


def plot_rewards(rewards, ax):
    ax.plot(rewards)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Progress - Rewards')
    plt.pause(0.0001)
    # ax.pause(0.0001)
    # plt.show()
    # plt.pause(0.0001)


def display_graph(trajectories: list[list[tuple[float, float]]], id_lines_color: int, ax):

    cmap = cm.get_cmap('Set1')
    color = cmap(id_lines_color % 9)
    # ax = plt.gca()

    # Clear previous trajectories
    for line in ax.lines:
        if line.get_label() != 'Mars Surface':
            line.remove()

    for trajectory in trajectories:
        x_values = [point[0] for point in trajectory]
        y_values = [point[1] for point in trajectory]
        ax.plot(x_values, y_values, marker='o',
                 markersize=2, color=color, label=f'Rocket {id_lines_color}')
        plt.pause(0.0001)


