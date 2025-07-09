import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import iplot


def yaw_sideslip_to_xy(yaw_rate, side_slip, time_array, car_params=None):
    """
    Convert yaw rate and side slip trajectory to X,Y position.

    Args:
        yaw_rate: array of yaw rates [rad/s]
        side_slip: array of side slip angles [rad]
        time_array: array of time points [s]
        car_params: dict with 'Vx' (longitudinal velocity), defaults to 30 m/s

    Returns:
        tuple: (x_positions, y_positions, headings)
    """
    if car_params is None:
        Vx = 30.0  # Default from your MzNonlinearCar
    else:
        Vx = car_params.get('Vx', 30.0)  # Use provided Vx or default to 30 m/s

    # Convert to numpy for easier manipulation
    yaw_rate = np.array(yaw_rate)
    side_slip = np.array(side_slip)
    time_array = np.array(time_array)

    n_points = len(yaw_rate)
    dt = time_array[1] - time_array[0] if len(time_array) > 1 else 0.01

    # Initialize position and heading
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    heading = np.zeros(n_points)

    # Integrate trajectory
    for i in range(1, n_points):
        # Integrate heading from yaw rate
        heading[i] = heading[i - 1] + yaw_rate[i] * dt

        # Current side slip and heading
        beta = side_slip[i]
        psi = heading[i]

        # Velocity components in vehicle frame
        vx_veh = Vx * np.cos(beta)
        vy_veh = Vx * np.sin(beta)

        # Transform to global frame
        vx_global = vx_veh * np.cos(psi) - vy_veh * np.sin(psi)
        vy_global = vx_veh * np.sin(psi) + vy_veh * np.cos(psi)

        # Integrate position
        x[i] = x[i - 1] + vx_global * dt
        y[i] = y[i - 1] + vy_global * dt

    return x, y, heading


def create_trajectory_animation(yaw_rate, side_slip, time_array, car_params=None,
                                title="Vehicle Trajectory Animation", grid_limits=None):
    """
    Create interactive trajectory animation with dual plots.

    Args:
        yaw_rate: array of yaw rates
        side_slip: array of side slip angles
        time_array: array of time points
        car_params: vehicle parameters dict
        title: plot title
        grid_limits: dict with 'yaw_rate' and 'side_slip' limits for phase plot

    Returns:
        Plotly figure with animation
    """
    # Convert to position coordinates
    x, y, heading = yaw_sideslip_to_xy(yaw_rate, side_slip, time_array, car_params)

    # Set default grid limits if not provided
    if grid_limits is None:
        # Default limits based on your grid setup (180 deg = π rad, 35 deg ≈ 0.61 rad)
        grid_limits = {
            'yaw_rate': [-np.pi, np.pi],  # ±180 degrees in rad/s
            'side_slip': [-35 * np.pi / 180, 35 * np.pi / 180]  # ±35 degrees in rad
        }

    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=("X-Y Position Trajectory", "Yaw Rate vs Side Slip"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )

    # Prepare animation frames
    frames = []
    n_points = len(x)

    for i in range(0, n_points):  # Limit to ~100 frames for performance

        # Current trajectory up to point i
        x_current = x[:i + 1] if i > 0 else x[:1]
        y_current = y[:i + 1] if i > 0 else y[:1]
        yaw_current = yaw_rate[:i + 1] if i > 0 else yaw_rate[:1]
        slip_current = side_slip[:i + 1] if i > 0 else side_slip[:1]

        # Create frame data
        frame_data = [
            # XY plot - full trajectory (gray)
            go.Scatter(x=x, y=y, mode='lines',
                       line=dict(color='lightgray', width=1),
                       name='Full Path', showlegend=False, xaxis='x1', yaxis='y1'),
            # XY plot - current trajectory (blue)
            go.Scatter(x=x_current, y=y_current, mode='lines',
                       line=dict(color='blue', width=3),
                       name='Current Path', showlegend=False, xaxis='x1', yaxis='y1'),
            # XY plot - current position (red dot)
            go.Scatter(x=[x[i]], y=[y[i]], mode='markers',
                       marker=dict(color='red', size=10),
                       name='Vehicle', showlegend=False, xaxis='x1', yaxis='y1'),

            # Yaw-Slip plot - full trajectory (gray)
            go.Scatter(x=yaw_rate, y=side_slip, mode='lines',
                       line=dict(color='lightgray', width=1),
                       name='Full Path', showlegend=False, xaxis='x2', yaxis='y2'),
            # Yaw-Slip plot - current trajectory (blue)
            go.Scatter(x=yaw_current, y=slip_current, mode='lines',
                       line=dict(color='blue', width=3),
                       name='Current Path', showlegend=False, xaxis='x2', yaxis='y2'),
            # Yaw-Slip plot - current state (red dot)
            go.Scatter(x=[yaw_rate[i]], y=[side_slip[i]], mode='markers',
                       marker=dict(color='red', size=10),
                       name='Current State', showlegend=False, xaxis='x2', yaxis='y2')
        ]

        frames.append(go.Frame(
            data=frame_data,
            name=f"frame_{i}",
            layout=dict(title=f"{title}<br>Time: {time_array[i]:.2f}s")
        ))

    # Add full trajectory as background FIRST (so it appears behind the blue line)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                             line=dict(color='lightgray', width=1),
                             name='Full Path', showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=yaw_rate, y=side_slip, mode='lines',
                             line=dict(color='lightgray', width=1),
                             name='Full Path', showlegend=False), row=1, col=2)

    # Initial frame - start with at least 2 points to show a line
    initial_points = min(2, len(x))
    fig.add_trace(go.Scatter(x=x[:initial_points], y=y[:initial_points], mode='lines+markers',
                             line=dict(color='blue', width=3),
                             marker=dict(color='red', size=10),
                             name='Vehicle'), row=1, col=1)

    fig.add_trace(go.Scatter(x=yaw_rate[:initial_points], y=side_slip[:initial_points], mode='lines+markers',
                             line=dict(color='blue', width=3),
                             marker=dict(color='red', size=10),
                             name='State'), row=1, col=2)

    # Update layout with animation controls
    fig.update_layout(
        title=title,
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 100, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 50}}],
                 "label": "Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate", "transition": {"duration": 0}}],
                 "label": "Pause", "method": "animate"}
            ],
            "direction": "left", "pad": {"r": 10, "t": 87},
            "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top", "xanchor": "left",
            "currentvalue": {"font": {"size": 20}, "prefix": "Frame:", "visible": True, "xanchor": "right"},
            "transition": {"duration": 50, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0,
            "steps": [{"args": [[f"frame_{k}"], {"frame": {"duration": 50, "redraw": True},
                                                 "mode": "immediate", "transition": {"duration": 50}}],
                       "label": str(k), "method": "animate"}
                      for k in range(0, n_points)]
        }],
    )

    fig.update(frames=frames)

    # Axis labels
    fig.update_xaxes(title_text="X Position [m]", row=1, col=1)
    fig.update_yaxes(title_text="Y Position [m]", row=1, col=1)
    fig.update_xaxes(title_text="Yaw Rate [rad/s]", row=1, col=2)
    fig.update_yaxes(title_text="Side Slip Angle [rad]", row=1, col=2)

    # Equal aspect ratio for position plot
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

    # SET GRID LIMITS FOR PHASE PLOT (second subplot)
    fig.update_xaxes(range=grid_limits['yaw_rate'], row=1, col=2)
    fig.update_yaxes(range=grid_limits['side_slip'], row=1, col=2)

    return fig


def plot_trajectory_comparison(yaw_rate, side_slip, time_array, car_params=None):
    """
    Create static comparison plot showing both coordinate systems.

    Args:
        yaw_rate: array of yaw rates
        side_slip: array of side slip angles
        time_array: array of time points
        car_params: vehicle parameters dict

    Returns:
        Plotly figure
    """
    x, y, heading = yaw_sideslip_to_xy(yaw_rate, side_slip, time_array, car_params)

    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=("X-Y Position", "Yaw Rate vs Side Slip",
                        "Position vs Time", "Yaw Rate & Side Slip vs Time"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    # XY trajectory
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                             line=dict(color='blue', width=2),
                             marker=dict(size=4), name='XY Path'), row=1, col=1)

    # Yaw rate vs side slip
    fig.add_trace(go.Scatter(x=yaw_rate, y=side_slip, mode='lines+markers',
                             line=dict(color='red', width=2),
                             marker=dict(size=4), name='Yaw-Slip'), row=1, col=2)

    # Position vs time
    fig.add_trace(go.Scatter(x=time_array, y=x, mode='lines',
                             name='X position', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_array, y=y, mode='lines',
                             name='Y position', line=dict(color='green')), row=2, col=1)

    # Yaw rate and side slip vs time
    fig.add_trace(go.Scatter(x=time_array, y=yaw_rate, mode='lines',
                             name='Yaw rate', line=dict(color='red')), row=2, col=2)
    fig.add_trace(go.Scatter(x=time_array, y=side_slip, mode='lines',
                             name='Side slip', line=dict(color='orange')), row=2, col=2)

    # Update layout
    fig.update_layout(height=800, title="Trajectory Analysis")

    # Axis labels
    fig.update_xaxes(title_text="X [m]", row=1, col=1)
    fig.update_yaxes(title_text="Y [m]", row=1, col=1)
    fig.update_xaxes(title_text="Yaw Rate [rad/s]", row=1, col=2)
    fig.update_yaxes(title_text="Side Slip [rad]", row=1, col=2)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Position [m]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="Angular [rad or rad/s]", row=2, col=2)

    return fig