"""Read-only motor probe and interactive LeRobot calibration for this arm."""
import os
import argparse
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PORT=os.environ.get('SOARM_PORT','/dev/ttyACM0')
JOINTS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')

def make_robot():
    from lerobot.robots.so_follower import SO101Follower,SO101FollowerConfig
    return SO101Follower(SO101FollowerConfig(port=PORT,id='soarm_mirror',calibration_dir=ROOT/'calibration',use_degrees=True))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['probe','audit','calibrate','observe','stream'])
    args=parser.parse_args()
    robot=make_robot()
    # Deliberately bypass Robot.connect(): it configures motors and may enable torque.
    bus=robot.bus
    try:
        bus.connect()
        if args.mode=='calibrate':
            input('Support the arm; clear its surroundings. Enter to disable torque and begin hand-guided calibration: ')
            robot.calibrate()
        elif args.mode in ('observe','stream'):
            if not robot.calibration or not bus.is_calibrated:
                raise RuntimeError('Run this project calibration first; stored and motor calibration must match.')
            while True:
                print(json.dumps({'time':time.time(),'positions':bus.sync_read('Present_Position')}),flush=True)
                if args.mode=='observe':break
                time.sleep(1/15)
        elif args.mode=='audit':
            registers=('Torque_Enable','Operating_Mode','Present_Position','Goal_Position',
                       'P_Coefficient','I_Coefficient','D_Coefficient','Minimum_Startup_Force',
                       'CW_Dead_Zone','CCW_Dead_Zone','Torque_Limit','Max_Torque_Limit',
                       'Protection_Current','Overload_Torque','Protective_Torque',
                       'Acceleration','Goal_Velocity','Present_Load','Present_Current',
                       'Present_Voltage','Present_Temperature','Status')
            report={'port':PORT,'read_only':True,'time':time.time(),'registers':{},'errors':{}}
            for register in registers:
                try:report['registers'][register]=bus.sync_read(register,normalize=False)
                except (ValueError,KeyError,NotImplementedError) as error:report['errors'][register]=str(error)
            (ROOT/'calibration/motor_settings_audit.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
        else:
            report={'port':PORT,'read_only':True,'positions_raw':bus.sync_read('Present_Position',normalize=False),
                    'torque_enabled':bus.sync_read('Torque_Enable',normalize=False),
                    'project_calibration_exists':bool(robot.calibration)}
            (ROOT/'calibration').mkdir(exist_ok=True)
            (ROOT/'calibration/probe.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        # No automatic torque change on probe/observe or on cancellation at the first prompt.
        # LeRobot calibration itself disables torque; it remains disabled afterward.

if __name__=='__main__':main()
