"""Estimate a checkerboard plane from a saved raw overhead image, without robot I/O."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from camera_calibration import detect, object_points


def register(image_path, calibration_path, output, on_table=False):
    c = json.loads(calibration_path.read_text())
    frame = cv2.imread(str(image_path))
    if frame is None or list(frame.shape[1::-1]) != c['size']:
        raise ValueError('Image must match calibrated resolution')
    found, corners = detect(frame)
    if not found:
        raise ValueError('Full checkerboard not detected')
    K, dist = np.array(c['matrix']), np.array(c['distortion'])
    obj = object_points(c['square_m'])
    ok, r, t = cv2.solvePnP(obj, corners, K, dist)
    if not ok:
        raise ValueError('Pose estimation failed')
    R = cv2.Rodrigues(r)[0]
    if np.any((obj @ R.T + t.ravel())[:, 2] <= 0):
        raise ValueError('Board behind camera')
    projected = cv2.projectPoints(obj, r, t, K, dist)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected-corners.reshape(-1, 2))**2, axis=1))))
    if rms > 1.0:
        raise ValueError(f'Plane fit requires review: {rms:.3f} px')
    H = K @ np.column_stack((R[:, 0], R[:, 1], t.ravel()))
    camera = (-R.T @ t).ravel()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t.ravel()
    result = {
        'status': 'measured_board_plane_robot_registration_pending',
        'source_image': str(image_path.resolve()),
        'intrinsics_source': str(calibration_path.resolve()),
        'size': c['size'], 'square_m': c['square_m'],
        'frame_convention': 'Origin is first detected inner corner; X along 9-corner row, Y along 6-corner column, Z=X cross Y. Checkerboard orientation is ambiguous by 180 degrees between captures.',
        'T_camera_from_board': T.tolist(),
        'camera_position_in_board_m': camera.tolist(),
        'camera_distance_to_printed_plane_m': float(abs(camera[2])),
        'board_xy_m_to_undistorted_pixel_homography': H.tolist(),
        'undistorted_pixel_to_board_xy_m_homography': np.linalg.inv(H).tolist(),
        'rms_px': rms, 'detected_corners_px': corners.reshape(-1,2).tolist(),
        'limitations': ['Undistort raw pixels with recorded K and distortion before applying homography.',
                       'Mapping applies to printed plane only; elevated objects produce parallax.',
                       'Backing thickness has not been measured; board-to-table height offset is unknown.',
                       'Board-to-robot/world transform is unknown; simulation configuration is unchanged.',
                       'Single-view reprojection error is not independent metric validation.']}
    if on_table:
        result['printed_plane_above_table_m'] = 0.0
        result['table_contact_source'] = 'User confirmed sheet directly on tabletop'
        result['limitations'].remove('Backing thickness has not been measured; board-to-table height offset is unknown.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'board_plane.json').write_text(json.dumps(result, indent=2)+'\n')
    annotated = frame.copy()
    cv2.drawChessboardCorners(annotated, (9,6), corners, True)
    cv2.drawFrameAxes(annotated, K, dist, r, t, .075)
    cv2.imwrite(str(output/'board_pose.jpg'), annotated)
    # Rectification in board coordinates at 1 mm per pixel, 1.2 m square.
    S = np.array([[1000.,0,500],[0,1000.,500],[0,0,1]])
    undistorted = cv2.undistort(frame, K, dist, None, K)
    top = cv2.warpPerspective(undistorted, S @ np.linalg.inv(H), (1200,1200))
    cv2.imwrite(str(output/'board_plane_top_view.jpg'), top)
    # Check geometry convention independently by recovering all 54 known points.
    ideal = cv2.projectPoints(obj, r, t, K, np.zeros(5))[0]
    back = cv2.perspectiveTransform(ideal, np.linalg.inv(H)).reshape(-1,2)
    err = float(np.max(np.linalg.norm(back-obj[:,:2],axis=1)))
    if err > 1e-5:
        raise ValueError('Plane mapping round-trip failed')
    print(json.dumps({'rms_px': rms, 'height_above_print_m': abs(camera[2]),
                      'mapping_round_trip_max_m': err, 'output': str(output)}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--intrinsics', type=Path, default=Path('calibration/cameras/overhead_640x480.json'))
    parser.add_argument('--output', type=Path, default=Path('calibration/table_registration'))
    parser.add_argument('--on-table', action='store_true', help='Use only when sheet is confirmed directly on tabletop')
    args = parser.parse_args()
    register(args.image, args.intrinsics, args.output, args.on_table)
