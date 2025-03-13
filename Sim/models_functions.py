from dataclasses import dataclass
import math

@dataclass
class State:
    theta: float
    x: float
    y: float

@dataclass 
class Input:
    w: float
    ax: float
    ay: float

@dataclass
class Meas:
    phi: float
    r: float

def meas_model(si: State, sj: State) -> Meas:
    """
    Simple measurement model. Assumes the antenna has no offset

    :param si: State of the robot recieving the transmission
    :param s2: State of the robot sending the transmission
    :returns: The predicted measurement
    """
    
    thetai = si.theta
    xi = si.x
    yi = si.y

    thetaj = sj.theta
    xj = sj.x
    yj = sj.y

    mout = Meas()

    mout.phi = thetaj - thetai
    mout.r = math.sqrt((xj-xi)**2 + (yj-yi)**2)

    return mout

def meas_model2(si: State, sj: State, di, dj) -> Meas:
    """
    More advanced measurement model. Assumes the antenna has a slight forward translation

    :param si: State of the robot recieving the transmission
    :param s2: State of the robot sending the transmission
    :param di: The forward translation for robot i
    :param dj: The forward translation for robot j
    :returns: The predicted measurement
    """

    thetai = si.theta
    xi = si.x
    yi = si.y

    thetaj = sj.theta
    xj = sj.x
    yj = sj.y

    mout = Meas()

    cdi = math.cos(thetai)*di
    sdi = math.sin(thetai)*di
    cdj = math.cos(thetaj)*dj
    sdj = math.sin(thetaj)*dj
    mout.phi = thetaj - thetai + math.atan((xj + cdj - xi + cdi)/(yj + sdj - yi + sdi))
    mout.r = math.sqrt((xj + cdj - xi + cdi)**2 + (yj + sdj - yi + sdi)**2)

    return mout


def motion_model(s: State, inp: Input ) -> State:
    return s