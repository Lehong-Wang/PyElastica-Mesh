import numpy as np
import time

def stream_npz_timesteps(npz_path, dt, start_time=0.0, end_time=None):
    """
    Stream simulation state snapshots every dt seconds from an NPZ file.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file.
    dt : float
        Time interval (seconds) between outputs.
    start_time : float, optional
        Start time for sampling (default: 0.0).
    end_time : float or None, optional
        End time for sampling (default: None = last time in file).

    Yields
    ------
    dict
        Dictionary containing state data for one timestep.
    """
    data = np.load(npz_path)

    time = data["time"]
    mesh_position = data["mesh_position"]
    mesh_director = data["mesh_director"]
    rod_position = data["rod_position"]
    rod_director = data["rod_director"]
    # print(time.shape, mesh_position.shape, mesh_director.shape, rod_position.shape, rod_director.shape)

    if end_time is None:
        end_time = time[-1]

    target_times = np.arange(start_time, end_time + 1e-12, dt)

    # For each target time, find closest actual timestep
    for t_target in target_times:
        idx = np.argmin(np.abs(time - t_target))

        yield {
            "time": time[idx],
            "mesh_position": mesh_position[idx],        # (3,)
            "mesh_director": mesh_director[idx],        # (3, 3)
            "rod_position": rod_position[idx],          # (3, 21)
            "rod_director": rod_director[idx],          # (3, 3, 20)
        }


if __name__ == "__main__":
    npz_file = "output/mesh_rod_collision_state_snapshot.npz"
    start = time.time()
    t = 0
    state_generator = stream_npz_timesteps(npz_file, dt=0.1)
    while t < 4.5:
        if time.time() - start < .1:
            time.sleep(0.01)
            continue
        state = next(state_generator)
        t = state["time"]
        print(state["time"], state["rod_position"].shape)
        start = time.time()

