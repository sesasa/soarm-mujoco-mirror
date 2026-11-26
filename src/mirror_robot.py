"""Read physical joint feedback into MuJoCo without commanding or changing torque."""
import os
import argparse
import json
import queue
import subprocess
import threading
import time
import numpy as np
from simulate import Simulation, JOINTS, ROOT

PYTHON=os.environ.get('LEROBOT_PYTHON','python')

def convert(positions,mapping):
    values=np.array([positions[name] for name in JOINTS],dtype=float)
    if not np.isfinite(values).all():raise ValueError('Non-finite motor position')
    q=np.empty(6)
    q[:5]=np.deg2rad(values[:5])*mapping['sign']+mapping['offset_rad']
    q[5]=mapping['gripper_closed_rad']+values[5]/100*(mapping['gripper_open_rad']-mapping['gripper_closed_rad'])
    return q

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=300)
    args=parser.parse_args()
    mapping=json.loads((ROOT/'config/joint_mapping.json').read_text())
    s=Simulation()
    proc=subprocess.Popen([PYTHON,str(ROOT/'src/robot_setup.py'),'stream'],stdout=subprocess.PIPE,text=True,bufsize=1)
    updates=queue.Queue(maxsize=1)
    def reader():
        for line in proc.stdout:
            try:
                record=json.loads(line)
                q=convert(record['positions'],mapping)
                if updates.full():
                    try:updates.get_nowait()
                    except queue.Empty:pass
                updates.put_nowait((time.monotonic(),q))
            except (ValueError,KeyError,queue.Full):continue
    threading.Thread(target=reader,daemon=True).start()
    try:
        try:stamp,q=updates.get(timeout=10)
        except queue.Empty:raise RuntimeError('No calibrated feedback. Check calibration and serial-port availability.')
        from mujoco import viewer
        import mujoco
        print('READ-ONLY MIRROR. Joint mapping is provisional; no actuator commands or torque changes.')
        print('Close the window to stop. Feedback older than one second stops the mirror.')
        with viewer.launch_passive(s.model,s.data,show_right_ui=False) as v:
            v.cam.lookat[:]=[.25,0,.10];v.cam.distance=1.6;v.cam.azimuth=135;v.cam.elevation=-40
            deadline=time.monotonic()+args.seconds
            while v.is_running() and time.monotonic()<deadline:
                try:stamp,q=updates.get_nowait()
                except queue.Empty:pass
                if proc.poll() is not None:raise RuntimeError('Robot feedback process exited')
                if time.monotonic()-stamp>1:raise RuntimeError('Robot feedback stale; mirror stopped')
                s.data.qpos[s.addresses]=q
                s.data.qvel[:]=0
                mujoco.mj_forward(s.model,s.data)
                v.sync()
                time.sleep(1/30)
    finally:
        if proc.poll() is None:
            # SIGINT lets the child close the bus without changing torque.
            import signal
            proc.send_signal(signal.SIGINT)
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=3)
        if proc.stdout:proc.stdout.close()

if __name__=='__main__':main()
