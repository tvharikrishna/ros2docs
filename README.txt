<mujoco model="my_first_robot">
    <option gravity="0 0 -9.81" timestep="0.01"/>

    <asset>
        <material name="ground" rgba="0.3 0.3 0.3 1" />
        <material name="cube" rgba="0.8 0.1 0.1 1" />
    </asset>

    <worldbody>
    
        <light pos="0 0 2" dir="0 0 -1" />

        <geom name="floor" type="plane" size="5 5 0.1" material="ground" />

        <body name="cube" pos="0 0 5">
            <joint type="free" />
            <geom type="box" size="0.1 0.1 0.1" material="cube" mass="1" />
        </body>

    </worldbody>

</mujoco>



import mujoco
import mujoco.viewer
import os

model_path = os.path.join("..", "models", "myfirstrobot.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)


