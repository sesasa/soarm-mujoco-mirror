"""Six MuJoCo actuator sliders with speed-limited setpoints. No hardware I/O."""
import argparse
import os
# Use bundled X11 GLFW under XWayland; native Wayland crashed on viewer shutdown.
if os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
import time
import json
from pathlib import Path
import mujoco
import numpy as np
from build_scene import build, ROOT

JOINTS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')

class Simulation:
    def __init__(self):
        self.config=build()
        self.model=mujoco.MjModel.from_xml_path(str(ROOT/'scene.xml'))
        self.data=mujoco.MjData(self.model)
        self.addresses=np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.target=np.array(self.config['home_rad'],dtype=float)
        self.command=self.target.copy()
        self.data.qpos[self.addresses]=self.target
        self.data.ctrl[:]=self.target
        mujoco.mj_forward(self.model,self.data)

    def step(self):
        self.target=np.clip(self.target,*self.model.actuator_ctrlrange.T)
        delta=self.config['command_speed_rad_s']*self.model.opt.timestep
        self.command+=np.clip(self.target-self.command,-delta,delta)
        self.data.ctrl[:]=self.command
        mujoco.mj_step(self.model,self.data)

    def record(self):
        return {'time':self.data.time,'joints':JOINTS,'target_rad':self.target.tolist(),
                'command_rad':self.command.tolist(),'measured_sim_rad':self.data.qpos[self.addresses].tolist(),
                'hardware_enabled':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--seconds',type=float,default=5)
    p.add_argument('--render',type=Path)
    p.add_argument('--log',type=Path,help='Record simulated commands and joint feedback as JSONL')
    args=p.parse_args()
    s=Simulation()
    if args.log:args.log.parent.mkdir(parents=True,exist_ok=True)
    log=args.log.open('w') if args.log else None
    try:
        if args.headless:
            for _ in range(int(args.seconds/s.model.opt.timestep)):
                s.step()
                if log:log.write(json.dumps(s.record())+'\n')
            assert np.isfinite(s.data.qpos).all()
            print(json.dumps(s.record(),indent=2))
        else:
            from mujoco import viewer as mj_viewer
            with mj_viewer.launch_passive(s.model,s.data,show_left_ui=True,show_right_ui=True) as viewer:
                viewer.cam.lookat[:]=[0.25,-0.18,0.12]
                viewer.cam.distance=1.8
                viewer.cam.azimuth=135
                viewer.cam.elevation=-35
                print('SIMULATION ONLY. Open the right-side Control section for six joint sliders (radians).')
                print('Camera selection is under Rendering. No serial port is opened.')
                while viewer.is_running():
                    start=time.monotonic()
                    # Expose requested targets to the sliders; physics receives ramped commands.
                    s.data.ctrl[:]=s.target
                    viewer.sync()
                    s.target=s.data.ctrl.copy()
                    for _ in range(10):s.step()
                    if log:log.write(json.dumps(s.record())+'\n')
                    time.sleep(max(0,10*s.model.opt.timestep-(time.monotonic()-start)))
        if args.render:
            args.render.parent.mkdir(parents=True,exist_ok=True)
            from PIL import Image
            camera=mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:]=[0.25,-0.18,0.12];camera.distance=1.8;camera.azimuth=135;camera.elevation=-40
            with mujoco.Renderer(s.model,height=720,width=960) as renderer:
                renderer.update_scene(s.data,camera=camera)
                Image.fromarray(renderer.render()).save(args.render)
    finally:
        if log:log.close()

if __name__=='__main__':main()
