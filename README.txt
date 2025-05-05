import mujoco
import mujoco.viewer
import gym
from gym import spaces
import numpy as np

class IOHandRLEnv(gym.Env):
    def __init__(self):
        model_path = r"C:\Users\harik\Desktop\mujoco\____\models\io_hand_urdf\io_hand.xml"
        self.mujoco_model = mujoco.MjModel.from_xml_path(model_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.viewer_launched = False

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
        print("We Will Win, Krsna")
        if not self.viewer_launched:
            mujoco.viewer.launch(self.mujoco_model, self.mujoco_data)
            self.viewer_launched = True

    def _get_obs(self):
        # Get joint positions and velocities
        qpos = self.mujoco_data.qpos
        qvel = self.mujoco_data.qvel

        # Get end-effector (TCP) position — assuming site named "ee_site"
        try:
            ee_pos = self.mujoco_data.site("ee_site").xpos
        except Exception:
            ee_pos = np.zeros(3)

        return np.concatenate([qpos, qvel, ee_pos])

    def _apply_action(self, action):
        # Clip and apply action as control
        action = np.clip(action, -1.0, 1.0)
        self.mujoco_data.ctrl[:] = action

    def _compute_reward(self, obs):
        # Extract end-effector position from obs
        ee_pos = obs[-3:]
        goal = np.array([0.3, 0.1, 0.2])  # static goal for now

        # Negative distance to goal as reward
        return -np.linalg.norm(ee_pos - goal)

    def _check_done(self, obs):
        # Done if end-effector is very close to goal
        ee_pos = obs[-3:]
        goal = np.array([0.3, 0.1, 0.2])
        return np.linalg.norm(ee_pos - goal) < 0.05

if __name__ == "__main__":
    env = IOHandRLEnv()
    env.render()
