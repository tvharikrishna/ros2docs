✅ STEP 1: Add a Site to tcp_link
Find this part of your MJCF:

xml
Copy
Edit
<body name="wrist_link" ...>
Inside that body (after the last <geom>), add this:

xml
Copy
Edit
<site name="ee_site" pos="0.399557 -0.001508 -0.40462" size="0.005" rgba="1 0 0 1"/>
So it looks like this:

xml
Copy
Edit
<body name="wrist_link" ...>
  ...
  <geom ... />
  <geom ... />
  
  <!-- ✅ Add this: -->
  <site name="ee_site" pos="0.399557 -0.001508 -0.40462" size="0.005" rgba="1 0 0 1"/>
</body>
✅ This marks the end-effector site for tracking and reward computation.

✅ STEP 2: Update Python Code
Now update your _get_obs() method to extract the site position:

python
Copy
Edit
def _get_obs(self):
    qpos = self.mujoco_data.qpos
    qvel = self.mujoco_data.qvel
    ee_pos = self.mujoco_data.site("ee_site").xpos
    return np.concatenate([qpos, qvel, ee_pos])
And update _compute_reward() and _check_done() to use that ee_pos.

✅ Summary:
Change	What
Add <site> in MJCF	Tells MuJoCo what the EE is
Use data.site("ee_site").xpos in Python	Lets you get its position
Reward = -distance to goal	Teaches arm to reach the goal

Would you like me to regenerate your full Python training-ready file with these updates next? 


import mujoco
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO


class IOHandRLEnv(gym.Env):
    def __init__(self):
        self.model_path = "io_hand_with_site.xml"  # your MJCF with <site name="ee_site" />
        self.mujoco_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

        self.viewer_launched = False
        self.goal = np.array([0.3, 0.1, 0.2])  # static target

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.mujoco_model.nu,), dtype=np.float32)
        obs_dim = self.mujoco_model.nq + self.mujoco_model.nv + 3  # qpos, qvel, ee_pos
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self):
        mujoco.mj_resetData(self.mujoco_model, self.mujoco_data)
        return self._get_obs()

    def step(self, action):
        self._apply_action(action)
        mujoco.mj_step(self.mujoco_model, self.mujoco_data)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._check_done(obs)

        return obs, reward, done, {}

    def render(self):
        if not self.viewer_launched:
            mujoco.viewer.launch(self.mujoco_model, self.mujoco_data)
            self.viewer_launched = True

    def _get_obs(self):
        qpos = self.mujoco_data.qpos
        qvel = self.mujoco_data.qvel
        try:
            ee_pos = self.mujoco_data.site("ee_site").xpos
        except:
            ee_pos = np.zeros(3)
        return np.concatenate([qpos, qvel, ee_pos])

    def _apply_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.mujoco_data.ctrl[:] = action

    def _compute_reward(self, obs):
        ee_pos = obs[-3:]
        return -np.linalg.norm(ee_pos - self.goal)

    def _check_done(self, obs):
        ee_pos = obs[-3:]
        return np.linalg.norm(ee_pos - self.goal) < 0.05


if __name__ == "__main__":
    env = IOHandRLEnv()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)

    model.save("ppo_iohand_final")
    print("✅ Training complete. Model saved as 'ppo_iohand_final'.")













