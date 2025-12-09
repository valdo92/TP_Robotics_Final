from MFOS_linear_pitch_feedback import run_one_simulation
import numpy as np

list_kp = [1., 2., 5., 10., 20., 50.]
step_force = 1.
best_forces = 0.
results = {}

if __name__ == "__main__":
    for pitch_kp in list_kp:
        actual_force = min(best_forces - 2., 0.)
        success = run_one_simulation(pitch_kp=pitch_kp, force=actual_force)
        if not success:
            results[pitch_kp] = None
        else:
            while success:
                actual_force += step_force
                success = run_one_simulation(pitch_kp=pitch_kp, force=actual_force)
            results[pitch_kp] = actual_force - step_force
    print(results)
