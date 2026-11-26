"""Move the proven approach pose with the currently selected tabletop target."""
import numpy as np
from scipy.spatial.transform import Rotation

def retarget(solver,mapping,saved,region,opening):
    reference=np.asarray(saved['plan']['goal'],float).copy();reference[5]=opening
    old_center=np.asarray(saved['surface_center_base_m'],float);new_center=np.asarray(region[0],float)
    shift=new_center-old_center
    if shift.shape!=(3,) or not np.isfinite(shift).all():raise ValueError('Invalid target position')
    body=solver.model.body('gripper').id
    original_position=solver.position(reference);original_rotation=solver.data.xmat[body].reshape(3,3).copy()
    solver.bounds=np.asarray(mapping.limits).copy()
    # Stay on the same local arm configuration, avoiding an alternative wrist flip.
    solver.bounds[:5,0]=np.maximum(solver.bounds[:5,0],reference[:5]-np.deg2rad(30))
    solver.bounds[:5,1]=np.minimum(solver.bounds[:5,1],reference[:5]+np.deg2rad(30))
    solved=solver.solve(reference,original_position+shift);destination=solved['goal'];mapping.check(destination)
    solver.position(destination);rotation=solver.data.xmat[body].reshape(3,3).copy()
    changed=float(np.rad2deg(Rotation.from_matrix(original_rotation.T@rotation).magnitude()))
    if changed>15:raise ValueError('Target movement needs more than a small posture adjustment')
    return destination,{'current_surface_center_base_m':new_center.tolist(),'previous_surface_center_base_m':old_center.tolist(),
                        'target_shift_m':shift.tolist(),'target_base_m':(original_position+shift).tolist(),
                        'position_error_m':solved['error_m'],'orientation_change_deg':changed}


def camera_left_offset(plane,center,distance):
    """A metric tabletop displacement toward decreasing overhead-image x."""
    distance=float(distance)
    if not np.isfinite(distance):raise ValueError('Invalid grasp correction')
    board=np.linalg.inv(plane.base_from_board)@np.r_[center,1.]
    pixel=plane.H@np.r_[board[:2],1.];pixel=pixel[:2]/pixel[2]
    points=plane.board_xy(np.array([pixel,pixel+[-1.,0.]]))
    direction=plane.base_from_board[:3,:2]@(points[1]-points[0])
    return direction/np.linalg.norm(direction)*distance
