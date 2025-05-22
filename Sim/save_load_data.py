import pickle 
import numpy as np
import os

class RobotData():
    """
        Save data from a single robot from N runs of length L
    """
    def __init__(self, x_log: np.ndarray, P_log: np.ndarray, IMU_nis_log: np.ndarray, RB_nis_log: np.ndarray, pos: np.ndarray, id: int, rb_ids: np.ndarray):
        self.x_log = x_log # state logged by robot
        self.P_log = P_log # covariance logged by robot
        self.IMU_nis_log = IMU_nis_log # IMU NIS logged by robot
        self.RB_nis_log = RB_nis_log # RB NIS logged by robot
        self.pos = pos # the reference position
        self.id = id
        self.rb_ids = rb_ids

class RobotCollection():
    """
        collection of robot datas, for easier organization
    """
    def __init__(self, robotlist: list, anchorlist: list):
        # Robot data
        self.robotlist = robotlist
        # Anchor data (mainly just their position) 
        self.anchorlist = anchorlist 

def save_data(obj: RobotCollection, filename):
    folder = "dataSim"
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    filepath = os.path.join(folder, filename + ".pickle")
    
    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_object(filename):
    folder = "dataSim"
    filepath = os.path.join(folder, filename + ".pickle")

    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)