import pickle 
import numpy as np
import os
import gzip

class RobotData():
    """
        Save data for N robots from k runs of length L
    """
    def __init__(self, x_log: np.ndarray, 
                 P_log: np.ndarray, 
                 IMU_nis_log: np.ndarray, 
                 RB_nis_log: np.ndarray, 
                 pos: np.ndarray,
                 biases: np.ndarray, 
                 ids: np.ndarray, 
                 rb_ids: np.ndarray):
        self.x_log = x_log # state logged by robot
        self.P_log = P_log # covariance logged by robot
        self.IMU_nis_log = IMU_nis_log # IMU NIS logged by robot
        self.RB_nis_log = RB_nis_log # RB NIS logged by robot
        self.pos = pos # the reference position
        self.biases = biases # the constant bias applied during this run
        self.ids = ids
        self.rb_ids = rb_ids

def save_data(obj: RobotData, filename):
    folder = "dataSim"
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    filepath = os.path.join(folder, filename + ".pkl.gz")
    
    try:
        with gzip.open(filepath, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_object(filename) -> RobotData:
    folder = "dataSim"
    filepath = os.path.join(folder, filename + ".pkl.gz")

    try:
        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)