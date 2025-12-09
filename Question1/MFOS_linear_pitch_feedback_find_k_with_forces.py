from MFOS_linear_pitch_feedback import run_one_simulation
import numpy as np
import json

list_kp = [5., 10., 20., 50.]
step_force = 1.
best_forces = 0.
results = {}
save_path = "Question1/results/linear_pitch_feedback_results.json"

if __name__ == "__main__":
    for pitch_kp in list_kp:
        actual_force = max(best_forces - 1., 0.)
        success = run_one_simulation(pitch_kp=pitch_kp, force=actual_force)
        if not success:
            results[pitch_kp] = None
        else:
            while success:
                if best_forces < actual_force:
                    best_forces = actual_force
                actual_force += step_force
                success = run_one_simulation(pitch_kp=pitch_kp, force=actual_force)
            results[pitch_kp] = actual_force - step_force
        with open(save_path, 'w') as f:
            json.dump(results, f)
    print(results)
