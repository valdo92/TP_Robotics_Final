import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import gymnasium as gym
import numpy as np
import upkie.envs
from upkie.utils.raspi import configure_agent_process, on_raspi

upkie.envs.register()


FORCE_DURATION = 1
FORCE_STEP = 0.5
MAX_FORCE = 10

force_values = np.arange(FORCE_STEP, MAX_FORCE + FORCE_STEP, FORCE_STEP)
results = []



upkie.envs.register()


def main(
    env,
    pitch_kp: float =10.0,
    pitch_ki: float = 1,
    position_kp: float = 10,
    position_ki: float = 1,
):
    force_values = np.arange(FORCE_STEP, MAX_FORCE + FORCE_STEP, FORCE_STEP)
    results = []

    for FORCE_N in force_values:
        obs, info = env.reset()
        simtime = 0.0
        force_active = True
        success = True

        pitch_integrator = 0.0
        position_integrator = 0.0
        dt = env.unwrapped.dt
        observation, _ = env.reset()  # connects to the spine
        pitches=[]
        action = 0.0 * env.action_space.sample()  # 1D action: [velocity]
        while True:
            simtime +=dt
            pitch = observation[0]
            pitches.append(pitch)
            position = observation[1]
            pitch_integrator += pitch * dt
            position_integrator += position * dt
            commanded_velocity = (
                pitch_kp * pitch
                + pitch_ki * pitch_integrator
                + position_kp * position
                + position_ki * position_integrator
            )
            action[0] = commanded_velocity
            if force_active and simtime < FORCE_DURATION:
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

            observation, _, terminated, truncated, info = env.step(action)

            if pitch >= np.pi/2 or pitch <= -np.pi/2:
                success=False

                break

            if terminated or truncated:
                
                observation, _ = env.reset()
                pitch_integrator = 0.0
                position_integrator = 0.0
                success=False
                break

            if not force_active and simtime > FORCE_DURATION + 2:
                break

        results.append((FORCE_N, success, pitches))
        print(f"Force {FORCE_N:>6.1f} N -> {'Success' if success else 'Failure'}")

        if not success:
            break


    print("\nSummary of tested forces:")
    for f, s, picthes in results:
        print(f" param : {(pitch_kp,pitch_ki,position_kp,position_ki)} :  {f:6.1f} N -> {'Success' if s else 'Failure'} ; {pitches}")
    return(f)     



if __name__ == "__main__":
    if on_raspi():
        configure_agent_process()


    F_max=0
    best_param = 0
    pitch_kp_list = [5,8,10,15,20]
    pitch_ki_list = [2,5,8,10]
    position_kp_list = [5,8,10,15,20]
    position_ki_list = [2,5,8,10]
    for k in pitch_kp_list:
        for i in pitch_ki_list:
            for j in position_ki_list:
                for l in position_kp_list:
                    with gym.make("UpkieGroundVelocity-v4", frequency=100.0) as env:
                        F = main(env, pitch_kp = k, pitch_ki= i, position_ki=j, position_kp=l)
                        if F>F_max:
                            best_param = (k,i,j,l)
                            F_max =F
    print("les meilleurs params avec le MFOS :" (best_param, F_max))