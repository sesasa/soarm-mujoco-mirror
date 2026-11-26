"""Commission six simulated joint controls with live physical feedback."""
import argparse
import json
import queue
import signal
import subprocess
import threading
import time
import numpy as np
from simulate import Simulation,ROOT
from mirror_robot import PYTHON

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seconds',type=float,default=3600);parser.add_argument('--step-file',type=str);parser.add_argument('--base-test',action='store_true');parser.add_argument('--direct',action='store_true');args=parser.parse_args()
    if sum(map(bool,(args.base_test,args.step_file,args.direct)))>1:parser.error('Choose one control mode')
    s=Simulation();messages=queue.Queue(maxsize=1);keys=queue.Queue()
    proc=subprocess.Popen([PYTHON,'-u',str(ROOT/'src/control_worker.py'),'--seconds',str(args.seconds)]+(['--direct'] if args.direct else ['--base-test'] if args.base_test else []),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def reader():
        for line in proc.stdout:
            try:
                data=json.loads(line)
                if messages.full():
                    try:messages.get_nowait()
                    except queue.Empty:pass
                messages.put_nowait(data)
            except (ValueError,queue.Full):continue
    threading.Thread(target=reader,daemon=True).start()
    def send(op,**kw):
        proc.stdin.write(json.dumps({'op':op,'time':time.monotonic(),**kw})+'\n');proc.stdin.flush()
    try:
        try:state=messages.get(timeout=10)
        except queue.Empty:raise RuntimeError('No robot feedback; check terminal errors')
        from mujoco import viewer
        import mujoco
        target=np.array(state['q']);previous=None;last_send=0;last_record=0
        telemetry=ROOT/'calibration/control_live.json'
        step=None;step_started=0.;step_used=False;step_notice=''
        prepared=json.loads(open(args.step_file).read()) if args.step_file else None
        if prepared and prepared['joint_mapping']!=json.loads((ROOT/'config/joint_mapping.json').read_text()):
            raise ValueError('Step mapping no longer matches configuration')
        def key(code):
            op={65:'arm',32:'hold',88:'release',80:'step'}.get(code)
            if op:keys.put(op)
        with viewer.launch_passive(s.model,s.data,show_right_ui=True,key_callback=key) as v:
            v.cam.lookat[:]=[.25,0,.15];v.cam.distance=1.5;v.cam.azimuth=135;v.cam.elevation=-30
            print('A: ENABLE MOTOR CONTROL | Space: HOLD | X: RELEASE TORQUE',flush=True)
            if args.base_test:print('BASE TEST: shoulder_pan only, +/-10 degrees, 3 degrees/second. Other joints locked.',flush=True)
            if prepared:print('P: run the prepared 3-degree step ONCE; gripper opening unchanged',flush=True)
            print('Close window: release torque and exit. Support the arm before release.',flush=True)
            ghost=mujoco.MjData(s.model)
            deadline=time.monotonic()+args.seconds
            while v.is_running() and time.monotonic()<deadline:
                try:state=messages.get_nowait()
                except queue.Empty:pass
                if proc.poll() is not None:raise RuntimeError('Robot worker exited')
                if time.monotonic()-state['time']>1:raise RuntimeError('Stale robot feedback')
                if time.monotonic()-last_record>=.2:
                    temporary=telemetry.with_suffix('.tmp')
                    temporary.write_text(json.dumps({**state,'recorded_at_unix':time.time(),
                        'step_notice':step_notice,'step_running':step is not None,
                        'joint_mapping':json.loads((ROOT/'config/joint_mapping.json').read_text())},indent=2))
                    temporary.replace(telemetry);last_record=time.monotonic()
                    if prepared or args.base_test or args.direct:
                        with (ROOT/'calibration/approach_tracking.jsonl').open('a') as audit:
                            audit.write(json.dumps({'time':time.time(),'state':state['state'],'q':state['q'],
                                'command':state['command'],'step_running':step is not None,'notice':step_notice})+'\n')
                status=(state['state'],state['reason'])
                if status!=previous:
                    print(f'{status[0]}: {status[1]}',flush=True)
                    target=np.array(state['q']);previous=status
                    if state['state']!='ACTIVE':step=None
                while not keys.empty():
                    op=keys.get_nowait()
                    if op=='step':
                        try:
                            if prepared is None or step_used or state['state']!='ACTIVE':
                                raise ValueError('Step requires ACTIVE state and an unused prepared plan')
                            from approach_step import validate_step
                            prepared=json.loads(open(args.step_file).read())
                            if prepared['joint_mapping']!=json.loads((ROOT/'config/joint_mapping.json').read_text()):
                                raise ValueError('Step mapping changed')
                            step=validate_step(prepared['plan'],state['q'],state['limits'])
                            # Start exactly at measured feedback after verifying the prepared pose.
                            step['start']=np.array(state['q'])
                            step['duration']=max(step['duration'],1.875*np.max(np.abs(step['goal']-step['start']))/np.deg2rad(1))
                            step=validate_step(step,state['q'],state['limits'])
                            step_started=time.monotonic();step_used=True
                            step_notice='Running one short approach step; Space stops'
                        except ValueError as error:step=None;step_notice=str(error)
                        print(step_notice,flush=True)
                    else:
                        step=None;target=np.array(state['q']);send(op)
                if step is not None and state['state']=='ACTIVE':
                    from simulated_reach import trajectory
                    elapsed=time.monotonic()-step_started;target=trajectory(step,elapsed)
                    if elapsed>=step['duration']:
                        step=None;step_notice='Short step complete; holding endpoint. No further automatic movement.'
                        print(step_notice,flush=True)
                if state['state']!='ACTIVE':target=np.array(state['q'])
                s.model.actuator_ctrlrange[:]=state['limits']
                s.data.qpos[s.addresses]=state['q'];s.data.qvel[:]=0;s.data.ctrl[:]=target
                mujoco.mj_forward(s.model,s.data)
                with v.lock():
                    # Display the requested pose as a translucent overlay; solid robot is feedback.
                    v.user_scn.ngeom=0
                    ghost.qpos[:]=s.data.qpos;ghost.qpos[s.addresses]=target
                    mujoco.mj_forward(s.model,ghost)
                    if state['state']=='ACTIVE':
                        for gid in range(s.model.ngeom):
                            if s.model.geom_group[gid]!=2:continue
                            idx=v.user_scn.ngeom
                            if idx>=v.user_scn.maxgeom:break
                            g=v.user_scn.geoms[idx]
                            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.004,0,0]),ghost.geom_xpos[gid],np.eye(3).ravel(),np.array([0,.8,1,.6],dtype=np.float32))
                            v.user_scn.ngeom+=1
                v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"{state['state']} | A: enable | Space: hold | X: release\n{state['reason']}\n{('DIRECT: six sliders, full working range, 7 deg/s' if args.direct else 'BASE TEST: shoulder_pan +/-10 deg, 3 deg/s; other joints locked' if args.base_test else 'Initial test: +/-5 deg. P: prepared short step (1 deg/s).')}\n{step_notice}",''))
                v.sync()
                if state['state']=='ACTIVE' and step is None:target=s.data.ctrl.copy()
                now=time.monotonic()
                if now-last_send>=1/15:
                    send('target',q=target.tolist()) if state['state']=='ACTIVE' else send('heartbeat')
                    last_send=now
                time.sleep(.01)
    finally:
        if proc.poll() is None:
            proc.stdin.close() # EOF triggers release and disconnect in the worker
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGINT)
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=3)
        if proc.stdout:proc.stdout.close()

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:pass
