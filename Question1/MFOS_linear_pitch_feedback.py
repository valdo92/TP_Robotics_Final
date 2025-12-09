import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import gymnasium as gym
import numpy as np
import upkie.envs
from upkie.utils.raspi import configure_agent_process, on_raspi

upkie.envs.register()


force_duration = 1
time_after_force = 5
time_before_force = 3
pitch_kp = 10

results = []



upkie.envs.register()


def main(
    env,
    pitch_kp =0,
    pitch_ki = 0,
    position_kp: float = 0,
    position_ki: float = 0,
    force=0.,
    force_duration = 1,
    time_after_force = 5,
    time_before_force = 1,
    failure_angle = np.pi/6
    
):
    results = []

    obs, info = env.reset()
    simtime = 0.0
    force_active = False
    success = True

    pitch_integrator = 0.0
    position_integrator = 0.0
    dt = env.unwrapped.dt
    observation, _ = env.reset()  # connects to the spine
    pitches=[]
    action = 0.0 * env.action_space.sample()  # 1D action: [velocity]
    while True:
        simtime += dt
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
        is_in_force_window = (
            simtime > time_before_force and 
            simtime < time_before_force + force_duration
            )
        if (
            force_active
            and is_in_force_window
            ):
            env.unwrapped.set_bullet_action({
                "external_forces": {
                    "base": {
                        "force": [force, 0.0, 0.0],
                        "position": [0., 0., 0.]
                    }
                }
            })
        elif force_active:
            env.unwrapped.set_bullet_action({})
            force_active = False
        elif is_in_force_window:
            force_active = True
            env.unwrapped.set_bullet_action({
                "external_forces": {
                    "base": {
                        "force": [force, 0.0, 0.0],
                        "position": [0., 0., 0.]
                    }
                }
            })
        else: 
            env.unwrapped.set_bullet_action({}) 



        observation, _, terminated, truncated, info = env.step(action)

        if pitch >= failure_angle or pitch <= -failure_angle:
            success=False
            break

        if terminated or truncated:   
            observation, _ = env.reset()
            pitch_integrator = 0.0
            position_integrator = 0.0
            success=False
            break
        
        if not force_active and simtime > force_duration + time_after_force + time_before_force:
            success = True
            break
    results.append((
        force, success, pitches))
    print(f"Force {
        force:>6.1f} N -> {'Success' if success else 'Failure'}")
    return success

def run_one_simulation(
    pitch_kp =0,
    pitch_ki = 0,
    position_kp = 0,
    position_ki = 0,
    force=0.,
    force_duration = 1,
    time_after_force = 5,
    time_before_force = 1,
    failure_angle=np.pi/6
    ):
    if on_raspi():
        configure_agent_process()
    with gym.make("UpkieGroundVelocity-v4", frequency=100.0) as env: 
        F = main(
            pitch_kp,
            pitch_ki,
            position_kp,
            position_ki,
            force,
            force_duration,
            time_after_force, 
            time_before_force,
            failure_angle=failure_angle
            )
    return F

 



if __name__ == "__main__":
    if on_raspi():
        configure_agent_process()
    with gym.make("UpkieGroundVelocity-v4", frequency=100.0) as env:
        F = main(env, pitch_kp = pitch_kp, force=4., failure_angle=np.pi/6)
        print(F)