import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import six_DoF_simulator as simulator
from matplotlib.widgets import Slider, Button


def visualize_parafoil_pose(
    euler_series,
    position_series,
    interval=100,
    slowmo_factor=1.0,
    save_path=None
):

    euler_series = np.array(euler_series)
    position_series = np.array(position_series)
    assert len(euler_series) == len(position_series), "Rotation and position series must be the same length."

    num_frames = len(euler_series)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.3)

    ax.set_xlabel('Global X')
    ax.set_ylabel('Global Y')
    ax.set_zlabel('Global Z')
    ax.set_title("Parafoil Pose with Trail + Controls")

    axes = np.eye(3)
    quivers, labels = [], []
    parafoil_poly = None
    trail_line = None
    trail_positions = []

    # Define parafoil shape and rigging angle
    wing_span = 2.0
    chord_length = 0.6
    half_span = wing_span / 2
    half_chord = chord_length / 2
    parafoil_body = np.array([
        [ half_chord, -half_span, 0],
        [ half_chord,  half_span, 0],
        [-half_chord,  half_span, 0],
        [-half_chord, -half_span, 0]
    ])
    Rp = np.array([0, 0, -1.11])
    rigging_angle_deg = -12
    rigging_rotation = R.from_euler('y', rigging_angle_deg, degrees=True)
    parafoil_body = rigging_rotation.apply(parafoil_body)
    Rp = rigging_rotation.apply(Rp)

    # States
    paused = [False]
    follow_mode = [True]
    current_frame = [0]

    # Precompute bounding box for full path view
    pos_min = position_series.min(axis=0)
    pos_max = position_series.max(axis=0)
    center = (pos_max + pos_min) / 2
    span = (pos_max - pos_min) / 2
    padding = 1.0
    limits = np.array([center - span - padding, center + span + padding])
    
    # UI elements (initially defined for later hiding)
    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    frame_slider = Slider(slider_ax, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%0.0f')

    button_ax = fig.add_axes([0.82, 0.11, 0.1, 0.04])
    play_button = Button(button_ax, 'Pause')

    toggle_ax = fig.add_axes([0.02, 0.11, 0.15, 0.04])
    toggle_button = Button(toggle_ax, 'Toggle View')

    # so we can clear the trail
    trail_line = None

    def draw_frame(frame):
        nonlocal quivers, labels, parafoil_poly, trail_line, trail_positions

        current_frame[0] = frame

        if follow_mode[0] and frame == 0:
            trail_positions.clear()

        roll, pitch, yaw = euler_series[frame]
        pos = position_series[frame]
        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)

        for q in quivers: q.remove()
        for l in labels: l.remove()
        if parafoil_poly: parafoil_poly.remove()
        if trail_line: trail_line.remove()

        rotated_axes = rotation.apply(axes)
        quivers, labels = [], []
        for i, color in enumerate(['r', 'g', 'b']):
            vec = rotated_axes[i]
            q = ax.quiver(*pos, *vec, color=color, length=1)
            label = ax.text(*(pos + vec * 1.5), ["Forward (X)", "Right (Y)", "Down (Z)"][i], color=color)
            quivers.append(q)
            labels.append(label)

        parafoil_pos = pos + rotation.apply(Rp)
        rotated_parafoil = rotation.apply(parafoil_body) + parafoil_pos
        parafoil_poly = Poly3DCollection([rotated_parafoil], alpha=0.4, color='gray')
        ax.add_collection3d(parafoil_poly)

        trail_positions.append(pos)
        trail_array = np.array(trail_positions)
        trail_line = ax.plot3D(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2], color='black')[0]

        if follow_mode[0]:
            zoom = 3
            ax.set_xlim([pos[0] - zoom, pos[0] + zoom])
            ax.set_ylim([pos[1] - zoom, pos[1] + zoom])
            ax.set_zlim([pos[2] - zoom, pos[2] + zoom])
        else:
            ax.set_xlim(limits[0][0], limits[1][0])
            ax.set_ylim(limits[0][1], limits[1][1])
            ax.set_zlim(limits[0][2], limits[1][2])

        fig.canvas.draw_idle()

    def slider_update(val):
        if follow_mode[0]:
            paused[0] = True
            play_button.label.set_text('Play')
            frame = int(frame_slider.val)
            draw_frame(frame)

    def toggle_play(event):
        if follow_mode[0]:
            paused[0] = not paused[0]
            play_button.label.set_text('Pause' if not paused[0] else 'Play')

    def set_axes_equal(ax):
        """Set 3D plot axes to equal scale."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        max_range = max([x_range, y_range, z_range]) / 2.0

        ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
        ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
        ax.set_zlim3d([z_middle - max_range, z_middle + max_range])

    def toggle_view(event):
        follow_mode[0] = not follow_mode[0]
        paused[0] = False

        # Hide/show slider + play/pause button
        frame_slider.ax.set_visible(follow_mode[0])
        play_button.ax.set_visible(follow_mode[0])

        ax.clear()
        ax.set_xlabel('Global X')
        ax.set_ylabel('Global Y')
        ax.set_zlabel('Global Z')

        if follow_mode[0]:
            # Resume animated follow view
            ax.set_title("Parafoil Pose with Trail + Controls")
            draw_frame(current_frame[0])
        else:
            # Show static full path view
            ax.set_title("Full Trajectory View (Static)")
            trail_array = np.array(position_series)
            ax.plot3D(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2], color='black')
            ax.set_xlim(limits[0][0], limits[1][0])
            ax.set_ylim(limits[0][1], limits[1][1])
            ax.set_zlim(limits[0][2], limits[1][2])
            set_axes_equal(ax)
            fig.canvas.draw_idle()


    frame_slider.on_changed(slider_update)
    play_button.on_clicked(toggle_play)
    toggle_button.on_clicked(toggle_view)

    # Animation update
    actual_interval = int(interval * slowmo_factor)
    def update(frame):
        if follow_mode[0]:
            if not paused[0]:
                draw_frame(current_frame[0])
                current_frame[0] = (current_frame[0] + 1) % num_frames
                # Don't set slider value here – it will trigger slider callback
                frame_slider.eventson = False
                frame_slider.set_val(current_frame[0])
                frame_slider.eventson = True
        else:
            draw_frame(frame)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=actual_interval, blit=False, repeat=True)

    if save_path:
        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=1000 // actual_interval)
        ani.save(save_path, dpi=200, writer=writer)
    else:
        plt.show()