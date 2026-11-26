"""Overhead change footprints in MuJoCo. Read-only encoders; no robot commands."""
import os
import argparse
import json
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from preview_camera_alignment import Preview, ROOT

OUT = ROOT / 'captures/scene_changes'
WINDOW = 'Table changes - read only'
# Conservative tabletop work area in RAW camera pixels, excluding robot/desk edge.
RAW_ROI = np.array([[100,120],[440,110],[490,260],[385,340],[185,285],[65,235]], np.float32)

class Changes:
    def __init__(self, mask, stable_seconds=.5):
        self.mask = mask.astype(bool)
        if not np.isfinite(stable_seconds) or stable_seconds<0:raise ValueError('Invalid stability time')
        self.stable_seconds=float(stable_seconds)
        self.baseline = None
        self.age = np.zeros(mask.shape, np.float32)

    @staticmethod
    def features(frame):
        return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), (5,5), 0).astype(np.float32)

    def reset(self, frame):
        self.baseline = self.features(frame)
        self.age.fill(0)

    def update(self, frame, dt, excluded=None):
        if self.baseline is None:
            self.reset(frame)
        valid = self.mask.copy()
        if excluded is not None:
            valid &= ~excluded.astype(bool)
        if not valid.any():
            self.age.fill(0)
            return [], 'Work area obscured'
        delta = self.features(frame) - self.baseline
        # Compensate approximately for camera-wide exposure/white balance changes.
        delta -= np.median(delta[valid], axis=0)
        changed = ((np.max(np.abs(delta), axis=2)>24) & valid).astype(np.uint8)
        if changed[valid].mean() > .40:
            self.age.fill(0)
            return [], 'Large change: clear hands / check lighting, then B if needed'
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        changed[~valid] = 0
        self.age = np.where(changed, self.age+min(dt,.2), 0)
        stable = (self.age>self.stable_seconds).astype(np.uint8)
        contours,_ = cv2.findContours(stable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c.reshape(-1,2) for c in contours if cv2.contourArea(c)>=90]
        return sorted(contours, key=cv2.contourArea, reverse=True)[:12], 'Changes from baseline (including old positions)'

def merge_contained_fragments(contours):
    """Absorb disconnected interior markings without joining nearby silhouettes.

    Merge interior label islands or small aligned tip fragments. Width, length,
    area and gap checks constrain tip joins; this remains a geometric heuristic,
    not reliable identity for touching or overlapping objects.
    """
    groups=[]
    for contour in sorted(contours,key=cv2.contourArea,reverse=True):
        points=np.asarray(contour,np.int32).reshape(-1,2)
        for index,existing in enumerate(groups):
            hull=cv2.convexHull(existing)
            contained=all(cv2.pointPolygonTest(hull,tuple(map(float,p)),False)>=0 for p in points)
            # A small tip can lose its faint connection to an elongated body.
            # Join only a short, aligned continuation that preserves body width.
            box=cv2.boxPoints(cv2.minAreaRect(existing))
            edges=np.roll(box,-1,axis=0)-box
            axis=edges[np.argmax(np.linalg.norm(edges,axis=1))]
            length=np.linalg.norm(axis);axis=axis/max(length,1e-6)
            normal=np.array([-axis[1],axis[0]])
            along=existing@axis;extra=points@axis
            width=np.ptp(existing@normal)
            combined=np.vstack([existing,points])
            gap=max(extra.min()-along.max(),along.min()-extra.max(),0)
            continuation=(length>3*width and cv2.contourArea(points)<.3*cv2.contourArea(existing)
                          and np.ptp(extra)<.5*length and gap<.22*length
                          and np.ptp(combined@normal)<1.2*width)
            if contained or continuation:
                groups[index]=cv2.convexHull(np.vstack([existing,points])).reshape(-1,2)
                break
        else:
            groups.append(points)
    return groups


def refine_current_contours(frame, changes):
    """Separate present foreground from local tabletop inside each changed area.

    Assumes a locally plain tabletop. Moderate neutral darkening alone is treated
    as shadow. Unresolved/empty regions are omitted rather than sized as objects.
    """
    output = []
    for contour in changes:
        x,y,w,h=cv2.boundingRect(np.asarray(contour,np.int32))
        x0,y0=max(0,x-16),max(0,y-16)
        crop=frame[y0:min(frame.shape[0],y+h+16),x0:min(frame.shape[1],x+w+16)]
        lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB).astype(np.float32)
        support = np.zeros(crop.shape[:2],np.uint8)
        cv2.drawContours(support,[np.asarray(contour,np.int32)-[x0,y0]],-1,255,-1)
        allowed = cv2.dilate(support,np.ones((9,9),np.uint8))
        outer = cv2.dilate(allowed,np.ones((15,15),np.uint8))
        ring = (outer>0)&(allowed==0)
        if ring.sum()<40:continue
        background = np.median(lab[ring],axis=0)
        residual = lab-background
        chroma = np.linalg.norm(residual[:,:,1:],axis=2)
        strong = ((np.abs(residual[:,:,0])>45)|(chroma>18))&(allowed>0)
        if strong.sum()<30:continue
        # Colour models fitted to the current image, not the changed silhouette.
        labels = np.full(crop.shape[:2],cv2.GC_BGD,np.uint8)
        labels[allowed>0] = cv2.GC_PR_BGD
        labels[((np.abs(residual[:,:,0])>30)|(chroma>12))&(allowed>0)] = cv2.GC_PR_FGD
        labels[strong] = cv2.GC_FGD
        cv2.grabCut(crop,labels,None,np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_MASK)
        foreground = np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
        # Reject neutral shadow pixels even if GrabCut connects them across a gap.
        foreground[(np.abs(residual[:,:,0])<60)&(chroma<18)] = 0
        foreground = cv2.morphologyEx(foreground,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        found,_ = cv2.findContours(foreground,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in found:
            if cv2.contourArea(c)>=90:
                output.append(c.reshape(-1,2)+[x0,y0])
    # Overlapping change candidates may rediscover the same current foreground.
    merged=np.zeros(frame.shape[:2],np.uint8)
    for contour in output:cv2.drawContours(merged,[contour.astype(np.int32)],-1,255,-1)
    contours,_=cv2.findContours(merged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return merge_contained_fragments(contours)[:12]

class TargetSelection:
    """Conservative preview selection, cleared on loss or ambiguous association."""
    def __init__(self):
        self.polygon=None
        self.index=None
        self.message='Click an object to select'

    def clear(self, message='Selection cleared'):
        self.polygon=None;self.index=None;self.message=message

    def click(self, point, polygons):
        hits=[i for i,p in enumerate(polygons) if cv2.pointPolygonTest(np.asarray(p,np.float32),point,False)>=0]
        if len(hits)!=1:
            self.clear('Click away from overlapping outlines' if hits else 'Click inside an object outline')
            return
        self.index=hits[0];self.polygon=np.asarray(polygons[self.index],np.float32).copy()
        self.message='Selected tabletop position - preview only'

    def update(self, polygons):
        self.index=None
        if self.polygon is None:return
        scores=[]
        for p in polygons:
            p=np.asarray(p,np.float32)
            intersection=cv2.intersectConvexConvex(self.polygon,p)[0]
            union=cv2.contourArea(self.polygon)+cv2.contourArea(p)-intersection
            scores.append(intersection/max(union,1e-6))
        order=np.argsort(scores)[::-1]
        if not len(order) or scores[order[0]]<.5 or (len(order)>1 and scores[order[0]]-scores[order[1]]<.2):
            self.clear('Target lost or ambiguous - click again')
            return
        self.index=int(order[0]);self.polygon=np.asarray(polygons[self.index],np.float32).copy()


class Plane:
    def __init__(self, K, camera_from_base, camera_from_board):
        self.K = K
        self.H = K @ camera_from_board[:3,[0,1,3]]
        self.base_from_board = np.linalg.inv(camera_from_base) @ camera_from_board

    def board_xy(self, pixels):
        return cv2.perspectiveTransform(np.asarray(pixels,dtype=np.float64).reshape(1,-1,2),np.linalg.inv(self.H))[0]

    def footprint(self, contour):
        # Fit in metric tabletop coordinates, not camera pixels: perspective matters.
        xy = self.board_xy(contour).astype(np.float32)
        (cx,cy),(width,height),angle = cv2.minAreaRect(xy)
        theta = np.deg2rad(angle)
        c,s = np.cos(theta),np.sin(theta)
        local_rotation = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        center = self.base_from_board @ np.array([cx,cy,0,1])
        rotation = self.base_from_board[:3,:3] @ local_rotation
        return center[:3], np.array([max(width/2,.0005),max(height/2,.0005),.001]), rotation

    def projected_footprint(self, region):
        center,size,rotation = region
        corners = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])*size
        base = corners @ rotation.T + center
        board = (np.linalg.inv(self.base_from_board) @ np.c_[base,np.ones(4)].T).T
        return cv2.perspectiveTransform(board[None,:,:2],self.H)[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--reuse-baseline', action='store_true',help='Keep the previous lens-corrected baseline after restarting')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    preview=Preview()
    plane_info=json.loads((OUT/'plane.json').read_text())
    plane=Plane(preview.K,preview.T,np.array(plane_info['T_camera_from_board']))
    roi=cv2.undistortPoints(RAW_ROI.reshape(-1,1,2),preview.K,preview.dist,P=preview.K).reshape(-1,2).round().astype(np.int32)
    mask=np.zeros((480,640),np.uint8);cv2.fillPoly(mask,[roi],255)
    from measured_objects import highlighter_proxy
    measured_profile=json.loads((ROOT/'config/measured_objects.json').read_text())['pink_highlighter']
    detector=Changes(mask)
    selection=TargetSelection()
    polygons=[]
    cap=proc=None
    reach_preview=None;reach_plan=None;reach_progress=None
    reach_notice=""
    reach_status="R: position reach | G: aligned hover (simulation only)"
    try:
        if args.offline:
            frame=cv2.imread(str(OUT/'setup.png'))
            positions=json.loads((ROOT/'calibration/wrist_extrinsics_secured/1789762157381228096/pose_009/joints.json').read_text())['positions']
        else:
            spec=preview.meta['camera']
            cap=cv2.VideoCapture(spec['path'],cv2.CAP_V4L2)
            if not cap.isOpened():raise RuntimeError('Overhead camera unavailable')
            cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*spec['fourcc']))
            cap.set(3,640);cap.set(4,480)
            for _ in range(15):ok,frame=cap.read()
            if not ok:raise RuntimeError('Overhead read failed')
            proc=subprocess.Popen([os.environ.get('LEROBOT_PYTHON','python'),str(ROOT/'src/robot_setup.py'),'stream'],stdout=subprocess.PIPE,text=True,bufsize=1)
            records=queue.Queue(maxsize=1)
            def reader():
                for line in proc.stdout:
                    try:
                        item=(time.monotonic(),json.loads(line))
                        if records.full():records.get_nowait()
                        records.put_nowait(item)
                    except (ValueError,queue.Empty,queue.Full):pass
            threading.Thread(target=reader,daemon=True).start()
            stamp,record=records.get(timeout=15)
            positions=record['positions']
            cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,1440,360)
            def on_mouse(event,x,y,flags,param):
                if event==cv2.EVENT_LBUTTONDOWN and 28<=y<448:
                    selection.click((float(x%640),float(y)),polygons)
            cv2.setMouseCallback(WINDOW,on_mouse)
        real=cv2.remap(frame,*preview.maps,cv2.INTER_LINEAR)
        baseline=cv2.imread(str(OUT/'baseline.png')) if args.reuse_baseline else real
        if baseline is None or baseline.shape!=real.shape:raise RuntimeError('Saved baseline unavailable or wrong size')
        detector.reset(baseline)
        if not args.offline:cv2.imwrite(str(OUT/'baseline.png'),baseline)
        prior=time.monotonic();last_save=0
        print('Scene-change preview open. B: baseline, S: save, Esc: close. NO robot commands.',flush=True)
        while True:
            if not args.offline:
                ok,frame=cap.read()
                if not ok:raise RuntimeError('Overhead read failed')
                try:stamp,record=records.get_nowait()
                except queue.Empty:pass
                if proc.poll() is not None or time.monotonic()-stamp>1:raise RuntimeError('Stale encoders; stopping preview')
                positions=record['positions']
            now=time.monotonic();dt=now-prior;prior=now
            real=cv2.remap(frame,*preview.maps,cv2.INTER_LINEAR)
            excluded=cv2.dilate(getattr(preview,'robot_mask',np.zeros_like(mask)),np.ones((25,25),np.uint8))
            contours,status=detector.update(real,dt,excluded)
            change_contours=contours
            contours=refine_current_contours(real,change_contours)
            if change_contours:status='Current appearance in changed areas - approximate'
            regions=[plane.footprint(c) for c in contours]
            if args.offline:
                # Explicit synthetic footprint for rendering verification only.
                contours=[cv2.boxPoints(((260,190),(65,16),35))];regions=[plane.footprint(contours[0])];status='OFFLINE synthetic footprint - rendering check'
            display_regions,measured_index,measured_report=highlighter_proxy(real,contours,regions,measured_profile)
            polygons=[plane.projected_footprint(r).round().astype(np.int32) for r in regions]
            selection.update(polygons)
            target=regions[selection.index][0] if selection.index is not None else None
            result=preview.draw(frame,positions,regions=display_regions,show_marker=False,target=target)
            cv2.polylines(result,[roi],True,(0,220,0),2)
            if measured_report:
                cv2.putText(result,'Highlighter model: 27 mm wide, 17 mm high',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
            polygons=[plane.projected_footprint(r).round().astype(np.int32) for r in regions]
            for index,(contour,polygon) in enumerate(zip(contours,polygons)):
                cv2.polylines(result,[contour.astype(np.int32)],True,(255,220,0),1)
                cv2.polylines(result,[polygon],True,(0,150,255),2)
                cv2.polylines(result,[polygon+np.array([1280,0])],True,(0,150,255),2)
                if index==selection.index:
                    for offset in (0,1280):
                        shifted=polygon+np.array([offset,0])
                        cv2.polylines(result,[shifted],True,(0,255,0),2)
                        center=tuple(np.rint(shifted.mean(axis=0)).astype(int))
                        cv2.drawMarker(result,center,(0,255,0),cv2.MARKER_CROSS,16,2)
            cv2.rectangle(result,(0,418),(1920,447),(20,20,20),-1)
            cv2.putText(result,selection.message+' | Click: select   C: clear   R: reach   G: align   O: open   D/U: down/up   X: stop',(8,438),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            cv2.rectangle(result,(0,448),(1920,480),(20,20,20),-1)
            cv2.putText(result,f'{status} | {len(contours)} areas | B: baseline  S: save  Esc: close | NO MOTION',(8,470),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1)
            if args.offline:
                cv2.imwrite(str(OUT/'offline_check.jpg'),result);break
            reach_image=None
            if reach_plan is not None:
                reach_progress.update(regions,dt)
                reach_status=reach_notice or reach_progress.message
                from simulated_reach import trajectory
                virtual_q=trajectory(reach_plan,reach_progress.elapsed)
                reach_image=reach_preview.draw(frame,positions,regions=display_regions,show_marker=False,
                                              target=reach_plan['target'],q_override=virtual_q)
                reach_image[:,640:1280]=reach_preview.draw_side(plane.base_from_board,reach_plan['target'],display_regions)
                cv2.rectangle(reach_image,(0,0),(1920,30),(20,20,20),-1)
                hover_height=float(np.dot(reach_plan['target']-reach_plan['surface'],plane.base_from_board[:3,2]))
                cv2.putText(reach_image,f'SIMULATION ONLY | green line: {abs(hover_height)*100:.0f} cm above table | D: down 1 cm  U: up 1 cm | real arm stays still',
                            (8,21),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),1)
                cv2.rectangle(reach_image,(0,440),(1920,480),(20,20,20),-1)
                phase='Holding hover pose' if reach_progress.elapsed>=reach_plan['duration'] else 'Animating virtual arm'
                message=f"{phase} | model error {reach_plan['error_m']*1000:.2f} mm | initial limit adjustment {reach_plan['start_adjustment_deg']:.1f} deg | position only; no collision plan"
                if reach_plan.get('mode') in ('aligned_hover','vertical_step','jaw_opening'):
                    message=f"{phase} | position {reach_plan['error_m']*1000:.2f} mm | down tilt {reach_plan['downward_tilt_deg']:.1f} deg | jaw alignment {reach_plan['jaw_alignment_error_deg']:.1f} deg | no collision plan"
                if reach_plan.get('jaw_opening'):
                    from simulated_reach import fingertip_gap
                    gap=fingertip_gap(reach_solver,virtual_q)*1000
                    cv2.putText(reach_image,f'Virtual fingertip gap: {gap:.1f} mm | object width: 27 mm | no contact simulation',(8,60),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                cv2.putText(reach_image,message,(8,465),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                clearance=reach_plan.get('table_clearance')
                if clearance:
                    cv2.rectangle(reach_image,(0,375),(1920,405),(20,20,20),-1)
                    label=f"SAMPLED table clearance: moving arm {clearance['path_minimum_m']*1000:.1f} mm; gripper/holder {clearance['gripper_holder_path_minimum_m']*1000:.1f} mm | approximate model; objects/self-collision unchecked"
                    cv2.putText(reach_image,label,(8,397),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,210,255),1)
                if reach_plan.get('measured_object') and clearance:
                    margin=clearance['gripper_holder_path_minimum_m']-reach_plan['measured_object']['height_m']
                    cv2.putText(reach_image,f'Model gripper/holder above object top plane: {margin*1000:.1f} mm (sampled)',(8,420),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,210,255),1)
                if reach_notice:
                    cv2.rectangle(reach_image,(0,405),(1920,439),(20,20,20),-1)
                    cv2.putText(reach_image,reach_notice,(8,430),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,180,255),1)
                if reach_progress.paused:
                    cv2.rectangle(reach_image,(0,405),(1920,439),(20,20,20),-1)
                    cv2.putText(reach_image,reach_progress.message,(8,430),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,180,255),1)
                cv2.imshow('SIMULATED REACH - real arm stays still',reach_image)
                if cv2.getWindowProperty('SIMULATED REACH - real arm stays still',cv2.WND_PROP_VISIBLE)<1:
                    reach_plan=None;reach_status="Virtual reach window closed"
            cv2.putText(result,reach_status,(650,45),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,0),1)
            cv2.imshow(WINDOW,result)
            key=cv2.waitKey(1)&255
            if key in (ord('r'),ord('R'),ord('g'),ord('G')):
                if target is None:
                    reach_status='Select an object first, then press R or G'
                else:
                    from simulated_reach import ReachSolver,hover_point,ReachProgress
                    from mirror_robot import convert
                    try:
                        if reach_preview is None:reach_preview=Preview()
                        reach_solver=ReachSolver(reach_preview.model)
                        destination=hover_point(target,plane.base_from_board[:3,2])
                        if key in (ord('g'),ord('G')):
                            region=regions[selection.index]
                            sizes=region[1][:2]
                            if max(sizes)/min(sizes)<1.5:raise ValueError('Object direction unclear: choose an elongated object')
                            axis=region[2][:,np.argmax(sizes)]
                            reach_plan=reach_solver.solve_grasp(convert(positions,preview.mapping),destination,
                                                               plane.base_from_board[:3,2],axis)
                        else:
                            reach_plan=reach_solver.solve(convert(positions,preview.mapping),destination)
                            reach_plan['mode']='position_only'
                        from table_clearance import TableClearance
                        checker=TableClearance(reach_preview.model,plane.base_from_board[:3,3],plane.base_from_board[:3,2])
                        reach_plan['table_clearance']=checker.path(reach_plan)
                        if reach_plan['table_clearance']['path_minimum_m']<0:
                            raise ValueError('Preview blocked: sampled arm/table intersection')
                        reach_notice=''
                        if measured_report is not None and measured_index==selection.index:
                            reach_plan['measured_object']=measured_report
                        reach_plan['surface']=target.copy()
                        reach_progress=ReachProgress(regions[selection.index])
                        cv2.namedWindow('SIMULATED REACH - real arm stays still',cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('SIMULATED REACH - real arm stays still',1440,360)
                        reach_status='Simulation-only reach active | X: stop'
                        (OUT/'last_simulated_reach.json').write_text(json.dumps({
                            **{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in reach_plan.items()},
                            'time':time.time(),'hardware_commands':False,'position_only':reach_plan['mode']=='position_only',
                            'collision_validated':False,'gripper_site':'gripperframe'},indent=2))
                    except ValueError as error:
                        reach_plan=None;reach_status=str(error)
            if key in (ord('o'),ord('O')):
                if reach_plan is None:
                    reach_notice='Select the highlighter and run G first'
                elif reach_progress.paused or reach_progress.elapsed<reach_plan['duration']:
                    reach_notice='Wait for the pose to finish and target to be stable'
                else:
                    from simulated_reach import open_for_object
                    from table_clearance import TableClearance
                    try:
                        checker=TableClearance(reach_preview.model,plane.base_from_board[:3,3],plane.base_from_board[:3,2])
                        next_plan=open_for_object(reach_solver,reach_plan,checker)
                        reach_plan=next_plan;reach_notice=''
                        reach_progress.elapsed=0.;reach_progress.stable=0.;reach_progress.paused=True
                        reach_status='Opening virtual jaw for measured object'
                        (OUT/'last_simulated_reach.json').write_text(json.dumps({
                            **{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in reach_plan.items()},
                            'time':time.time(),'hardware_commands':False,'position_only':False,
                            'collision_validated':False,'gripper_site':'gripperframe'},indent=2))
                    except ValueError as error:reach_notice=str(error)
            if key in (ord('d'),ord('D'),ord('u'),ord('U')):
                if reach_plan is None or reach_plan.get('mode') not in ('aligned_hover','vertical_step','jaw_opening'):
                    reach_status='Select an object and run G before descending';reach_notice=reach_status
                elif reach_progress.paused or reach_progress.elapsed<reach_plan['duration']:
                    reach_status='Wait for the pose to finish and the target to be stable';reach_notice=reach_status
                else:
                    from simulated_reach import vertical_step
                    from table_clearance import TableClearance
                    try:
                        checker=TableClearance(reach_preview.model,plane.base_from_board[:3,3],plane.base_from_board[:3,2])
                        next_plan=vertical_step(reach_solver,reach_plan,-.01 if key in (ord('d'),ord('D')) else .01,checker)
                        reach_notice=''
                        reach_plan=next_plan
                        reach_progress.elapsed=0.;reach_progress.stable=0.;reach_progress.paused=True
                        reach_status=f"Virtual step to {reach_plan['height_m']*100:.0f} cm above table"
                        (OUT/'last_simulated_reach.json').write_text(json.dumps({
                            **{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in reach_plan.items()},
                            'time':time.time(),'hardware_commands':False,'position_only':False,
                            'collision_validated':False,'gripper_site':'gripperframe'},indent=2))
                    except ValueError as error:
                        # Keep the last verified pose/window when a step is refused.
                        reach_status=str(error)
                        reach_notice=reach_status
            if key in (ord('x'),ord('X')):
                if reach_plan is not None:cv2.destroyWindow('SIMULATED REACH - real arm stays still')
                reach_plan=None;reach_status='Virtual reach stopped'
            if key in (ord('c'),ord('C'),ord('b'),ord('B')):
                if reach_plan is not None:cv2.destroyWindow('SIMULATED REACH - real arm stays still')
                reach_plan=None;reach_status='Virtual reach stopped by clear/reset'
            if key in (ord('c'),ord('C')):selection.clear()
            if key in (ord('b'),ord('B')):
                selection.clear();target=None
                detector.reset(real);cv2.imwrite(str(OUT/'baseline.png'),real)
            if now-last_save>2 or key in (ord('s'),ord('S')):
                cv2.imwrite(str(OUT/'latest.jpg'),result)
                if reach_image is not None:cv2.imwrite(str(OUT/'reach_latest.jpg'),reach_image)
                (OUT/'latest.json').write_text(json.dumps({'time':time.time(),'status':status,'contours_undistorted_px':[c.tolist() for c in contours],'footprints_undistorted_px':[p.tolist() for p in polygons],'dimensions_table_m':[(r[1][:2]*2).tolist() for r in regions],'rotations_base':[r[2].tolist() for r in regions],'centers_base_m':[r[0].tolist() for r in regions],'preview_only':True,'height_estimated':False,'measured_object':measured_report,'footprint_method':'local_current_appearance','change_area_count':len(change_contours),'selected_target_base_m':target.tolist() if target is not None and selection.index is not None else None,'selection_status':selection.message,'simulated_reach_status':reach_status,'simulated_reach_active':reach_plan is not None,'simulated_reach_paused':reach_progress.paused if reach_plan is not None else False,'simulated_reach_elapsed':reach_progress.elapsed if reach_plan is not None else None},indent=2))
                last_save=now
            if key in (ord('s'),ord('S')):
                capture=OUT/f'saved_{time.time_ns()}'
                cv2.imwrite(str(capture.with_suffix('.jpg')),result)
                cv2.imwrite(str(capture.with_name(capture.name+'_raw').with_suffix('.png')),frame)
                capture.with_suffix('.json').write_text(json.dumps({
                    'time':time.time(),'positions':positions,
                    'joint_mapping':preview.mapping,
                    'registration':preview.registration,
                    'intrinsics':{'matrix':preview.K.tolist(),'distortion':preview.dist.tolist()},
                    'encoder_sample_time':record.get('time') if not args.offline else None,
                    'encoder_receipt_age_s':time.monotonic()-stamp if not args.offline else None,
                    'hardware_synchronized':False,'hardware_commands':False,
                    'note':'Compare stationary poses; camera and encoders are sampled separately.'
                },indent=2))
                print(f'Saved image, raw frame and joint readings: {capture.name}',flush=True)
            if key==27 or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
    finally:
        if cap is not None:cap.release()
        if proc is not None:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=3)
            proc.stdout.close()
        if reach_preview is not None:reach_preview.close()
        preview.close();cv2.destroyAllWindows()

if __name__=='__main__':main()
