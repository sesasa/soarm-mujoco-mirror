"""Measured endpoint tolerances; transit poses allow modest gravity sag."""
import numpy as np


def endpoint_settled(stage,error):
    error=np.abs(np.asarray(error,float))
    if error.shape!=(6,) or not np.isfinite(error).all():return False
    arm_degrees=4 if stage in ('basket_lift','plate_exit') else 3
    # An object can stop the jaws just short of the commanded firm-grip gap.
    # Recorded biscuit closure settled at 5.63 degrees of jaw error.
    jaw_degrees=6 if stage in ('close','lift','withdraw','return','basket_lift','basket_transfer') else 2
    return bool(max(error[:5])<=np.deg2rad(arm_degrees) and error[5]<=np.deg2rad(jaw_degrees))
