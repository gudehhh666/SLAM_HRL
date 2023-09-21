import habitat_sim
import habitat_sim.agent
import random
import os


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "HFOV": 58, #80
            "NEAR": 0.02,
            "FAR": 500, #500
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "HFOV": 58,
            "NEAR": 0.02,
            "FAR": 500,
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "HFOV": 58,
            "NEAR": 0.02,
            "FAR": 500,
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            # sensor_spec = habitat_sim.SensorSpec()   ##这是habitat7的
            sensor_spec = habitat_sim.CameraSensorSpec()  #这是habitat9的
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.hfov = sensor_params["HFOV"]
            sensor_spec.near = sensor_params["NEAR"]
            sensor_spec.far = sensor_params["FAR"]
            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.02)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.0)
        ),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def choose_rand_sence():
    path_dir = os.path.split(os.path.realpath(__file__))[0] + '/data/'
    num = random.randint(1, 5)
    path = path_dir + str(num)+'.glb'
    print(path)
    return path


# rgb_sensor = True  # @param {type:"boolean"}
# depth_sensor = True  # @param {type:"boolean"}
# semantic_sensor = True  # @param {type:"boolean"}
sim_settings = {
    "width": 640,  # Spatial resolution of the observations
    "height": 480,
    "scene": None,  # Scene path
    "default_agent": 0,
    "sensor_height": 0.45,  # Height of sensors in meters
    "color_sensor": True,  # RGB sensor
    "depth_sensor": True,  # Depth sensor
    "semantic_sensor": False,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}

if __name__ == "__main__":
    scene = choose_rand_sence()
    print(scene)
