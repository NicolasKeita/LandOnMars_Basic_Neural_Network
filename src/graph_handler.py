from matplotlib import pyplot as plt, cm


def create_graph(points: list, title: str):
    plt.xlim(0, 7000)
    plt.ylim(0, 3000)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.plot(*zip(*[(point[0], point[1]) for point in points]), marker='o', label='Mars Surface')
    plt.legend()
    plt.grid(True)
    plt.draw()


def display_graph(trajectories: list[list[tuple[float, float]]], id_lines_color: int):
    cmap = cm.get_cmap('Set1')
    color = cmap(id_lines_color % 9)
    ax = plt.gca()

    # Clear previous trajectories
    for line in ax.lines:
        if line.get_label() != 'Mars Surface':
            line.remove()

    for trajectory in trajectories:
        x_values = [point[0] for point in trajectory]
        y_values = [point[1] for point in trajectory]
        plt.plot(x_values, y_values, marker='o',
                 markersize=2, color=color, label=f'Rocket {id_lines_color}')
        plt.pause(0.0001)


