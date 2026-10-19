# PPO-Based Robot Navigation in Webots

## 1. Project Overview

This project investigates the use of Proximal Policy Optimization (PPO), a reinforcement learning algorithm, to train an e-puck mobile robot to navigate toward a target in the Webots simulation environment.

The project uses a custom Gymnasium environment to connect the Python training code with the Webots robot simulation.

The current implementation represents a single-agent PPO baseline. Its performance must be evaluated experimentally before drawing conclusions about navigation success or obstacle avoidance

The following diagram illustrates the interaction between the PPO training script, the custom Gymnasium environment, and the Webots simulation.

```mermaid
flowchart TD
    A["PPO Training<br/>train_ppo.py"]
    B["Gymnasium Environment<br/>WebotsGymEnv.py"]
    C["TCP/IP Socket<br/>JSON Messages"]
    D["Webots Simulation<br/>e-puck Controller"]

    A -->|"Wheel-speed action"| B
    B -->|"Send action"| C
    C -->|"Command"| D
    D -->|"Robot state"| C
    C -->|"Received state"| B
    B -->|"Observation and reward"| A
```

*Figure 1. High-level architecture of the PPO–Webots interaction loop.*

## 2. Project Objectives

- Implement a PPO-based training pipeline for a simulated mobile robot.
- Connect the Webots simulation to a custom Gymnasium environment.
- Define observations, wheel-speed actions, and a reward function for navigation.
- Save trained models and normalization statistics.
- Establish a reproducible baseline for future experiments.

## 3. Software and Tools

- **Webots:** Robot simulation environment.
- **Python:** Training and environment integration.
- **Gymnasium:** Interface for defining the reinforcement learning environment.
- **Stable-Baselines3:** PPO implementation and training utilities.
- **NumPy and PyTorch:** Numerical operations and model computation.

## 4. Environment and Robot

The simulated robot is an e-puck mobile robot. The Python environment communicates with the Webots controller through a local TCP/IP socket using JSON messages.

The observation vector contains 10 values:

1. Robot x-coordinate.
2. Robot y-coordinate.
3. Target x-coordinate.
4. Target y-coordinate.
5. Obstacle x-coordinate.
6. Obstacle y-coordinate.
7. Distance to the target.
8. Distance to the obstacle.
9. Angular difference between the robot heading and the target direction.
10. Robot yaw angle.

The action consists of two continuous values representing the left and right wheel velocities. The action space is bounded by approximately -7.536 to +7.536 rad/s for each wheel.

Although the Webots controller reads eight proximity sensors, their readings are not included in the current 10-value observation vector.

## 5. Training Configuration

The training script uses PPO from Stable-Baselines3 with an MLP policy.

| Parameter | Configured value |
|---|---:|
| Algorithm | PPO |
| Policy | MlpPolicy |
| Training timesteps per run | 200,000 |
| Random seed | 42 |
| Learning rate | 0.0003 |
| Rollout steps (`n_steps`) | 2,048 |
| Batch size | 256 |
| Training epochs (`n_epochs`) | 10 |
| Discount factor (`gamma`) | 0.995 |
| GAE parameter (`gae_lambda`) | 0.95 |
| Clipping range | 0.2 |
| Entropy coefficient | 0.001 |
| Checkpoint frequency | Every 20,000 callback steps |

Observation and reward normalization are configured using Stable-Baselines3 `VecNormalize`.

These values describe the configuration in the training script. They do not, by themselves, establish the final performance of the trained policy.

## 6. Reward Function

The reward function is designed to encourage the robot to reach the target and make progress toward it.

- **Target reached:** A reward of +10 is given when the target is reached for the first time in the current approach.
- **Progress toward target:** The reward includes 100 times the reduction in distance to the target since the previous step.
- **Heading alignment:** A small term, 0.1 times the cosine of the angular difference, encourages the robot to face the target.
- **Obstacle proximity:** A penalty is applied when the robot is closer than 0.20 m to the defined obstacle.
- **Boundary violation:** A penalty of -5 is applied when the robot's distance from the environment origin exceeds 0.8 m.

The reward function is a design choice and must be assessed through controlled experiments and evaluation metrics.

## 7. Saved Models and Training Artifacts

The supplied project archive contains model files and normalization statistics in `models_stage_Obs/`.

Important files include:

- `ppo_stage_obs.zip`: Saved PPO model.
- `vec_normalize.pkl`: Saved observation and reward normalization statistics.
- `ppo_stage1_*_steps.zip`: Training checkpoint files.

The presence of saved model files does not, on its own, prove successful navigation. The model must be evaluated in the simulation using a reliable evaluation procedure.

## 8. Evaluation and Results

The project includes a script named `test_ppo.py` intended to evaluate a saved policy over multiple episodes.

Quantitative results, including success rate, final target distance, episode reward, episode length, and collision statistics, should be reported only after the evaluation procedure has been checked and the experiments have been run.

No verified performance figures are reported in this README.

## 9. Current Limitations

- The current setup is a single-agent PPO baseline; multi-agent performance has not been established.
- The Webots controller currently assigns the target a fixed position of (0.5, 0.0), despite the target-randomization function name.
- The Gymnasium environment and the Webots controller should be checked for consistency in their target coordinates.
- Proximity sensor readings are available from the controller but are not included in the observation vector.
- The evaluation script expects information such as target distance and collision status, but the environment currently returns an empty `info` dictionary.
- Claims about success rate, generalization, and obstacle avoidance require corrected evaluation and experimental evidence.

## 10. Future Work

- Verify consistency between the environment and Webots controller.
- Correct and validate the evaluation pipeline.
- Run repeated evaluation episodes and record quantitative metrics.
- Save plots and simulation screenshots or videos as experimental evidence.
- Test target variation and obstacle configurations after implementing and verifying them.
- Investigate multi-agent reinforcement learning only after the single-agent baseline is reliable.

## 11. References

1. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. https://arxiv.org/abs/1707.06347
2. Schulman, J., et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438. https://arxiv.org/abs/1506.02438
3. Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. Journal of Machine Learning Research, 22(268), 1–8. https://jmlr.org/papers/v22/20-1364.html
