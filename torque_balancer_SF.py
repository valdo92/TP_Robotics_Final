import time
import gymnasium as gym
import numpy as np
import upkie.envs
import csv

upkie.envs.register()

GAIN = 10.0
FORCE_DURATION = 1.0
FORCE_STEP = 1.0
MAX_FORCE = 50.0

def run(env):
    action = env.unwrapped.get_neutral_action()

    for j in ["left_hip", "left_knee", "right_hip", "right_knee"]:
        action[j]["position"] = 0.0

    action["left_wheel"]["kd_scale"] = 0.0
    action["right_wheel"]["kd_scale"] = 0.0

    force_values = np.arange(FORCE_STEP, MAX_FORCE + FORCE_STEP, FORCE_STEP)
    results = []

    for FORCE_N in force_values:
        obs, info = env.reset()
        t0 = time.time()
        force_active = True
        success = True

        # Listes pour stocker pitch et temps
        pitch_list = []
        time_list = []

        while True:
            pitch = info["spine_observation"]["base_orientation"]["pitch"]
            print(pitch)
            pitch_list.append(pitch)
            time_list.append(time.time() - t0)

            action["left_wheel"]["feedforward_torque"]  = +GAIN * pitch
            action["right_wheel"]["feedforward_torque"] = -GAIN * pitch

            if force_active and time.time() - t0 < FORCE_DURATION:
                env.unwrapped.set_bullet_action({
                    "external_forces": {
                        "base": {
                            "force": [FORCE_N, 0.0, 0.0],
                            "position": [0.0, 0.0, 0.0],
                        }
                    }
                })
            elif force_active:
                env.unwrapped.set_bullet_action({})
                force_active = False

            obs, _, terminated, truncated, info = env.step(action)

            if pitch >= np.pi/2 or pitch <= -np.pi/2:
                success=False
                break

            if terminated or truncated:
                success=False
                break

            if not force_active and time.time() - t0 > FORCE_DURATION + 0.5:
                break

        results.append((FORCE_N, success))
        print(f"Force {FORCE_N:>6.1f} N -> {'Success' if success else 'Failure'}")

        # Sauvegarde CSV pour ce test
        """filename = f"pitch_force_{int(FORCE_N)}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "pitch"])
            for t, p in zip(time_list, pitch_list):
                writer.writerow([t, p])"""

        if not success:
            break

    print("\nSummary of tested forces:")
    for f, s in results:
        print(f"  {f:6.1f} N -> {'Success' if s else 'Failure'}")

if __name__ == "__main__":
    with gym.make("UpkieServos-v5", frequency=200.0) as env:
        run(env)
