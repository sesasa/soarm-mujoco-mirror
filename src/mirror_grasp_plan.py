"""Preflight a complete grasp for replay through the direct joint controller."""
import numpy as np
from scipy.optimize import brentq
from command_mapping import DirectTrajectory,ROOT
from grasp_planning import make_grasp_report
from simulated_reach import ReachSolver,fingertip_gap
from table_clearance import TableClearance

def build_sequence(model,mapping,plane,state,region,reach_only=False,basket_config=None,target_profile_override=None,return_to_sitting=True):
    start=np.asarray(state['command'],float)
    DirectTrajectory(mapping,start)
    import json
    target_path=ROOT/'config/grasp_target.json'
    target_profile=json.loads(target_path.read_text()) if target_path.exists() else {}
    if target_profile_override is not None:target_profile=dict(target_profile_override)
    packet=target_profile.get('kind')=='packet'
    width=float(2*min(np.asarray(region[1])[:2]))
    opening_gap=width+target_profile.get('opening_margin_m',.008) if packet else .035
    closing_gap=float(target_profile.get('closing_gap_m',max(.005,width-target_profile.get('closing_margin_m',.004)))) if packet else .020
    original=start.copy()
    up=plane.base_from_board[:3,2].copy()
    if up[2]<0:up=-up
    checker=TableClearance(model,plane.base_from_board[:3,3],up)
    prefix=[]
    if checker.at(start)['gripper_holder_minimum_m']<.035:
        if min(checker.at(start)['minimum_m'],checker.at(np.asarray(state['q']))['minimum_m'])<.002:
            raise ValueError('Resting jaw is too close to table for automatic wrist clearance')
        for degrees in (5,10,15,20,25,30):
            lifted=original.copy();lifted[3]-=np.deg2rad(degrees)
            if lifted[3]<mapping.limits[3][0]:continue
            if checker.at(lifted)['gripper_holder_minimum_m']<.04:continue
            duration=max(4.,1.875*np.deg2rad(degrees)/np.deg2rad(4))
            plan={'start':original,'goal':lifted,'duration':duration}
            from simulated_reach import trajectory
            gaps=[checker.at(trajectory(plan,t))['minimum_m'] for t in np.linspace(0,duration,161)]
            if min(gaps)<.002 or min(np.diff(gaps))<-.0001:continue
            prefix=[{'stage':'clear_wrist','plan':{'start':original.tolist(),'goal':lifted.tolist(),'duration':duration},'clearance':checker.path(plan,161)}]
            start=lifted;break
        else:raise ValueError('No wrist-up path found that increases table clearance')
    goal=start.copy()
    for i,(lo,hi) in enumerate(mapping.limits):
        if start[i]<lo:goal[i]=lo+np.deg2rad(1)
        elif start[i]>hi:goal[i]=hi-np.deg2rad(1)
    solver=ReachSolver(model)
    def gap(angle):
        q=goal.copy();q[5]=angle;return fingertip_gap(solver,q)-opening_gap
    goal[5]=max(goal[5],brentq(gap,mapping.limits[5][0],min(1.,mapping.limits[5][1])))
    up=plane.base_from_board[:3,2].copy()
    if up[2]<0:up=-up
    checker=TableClearance(model,plane.base_from_board[:3,3],up)
    duration=max(4.,1.875*max(abs(goal-start))/np.deg2rad(4))
    ready={'stage':'ready','plan':{'start':start.tolist(),'goal':goal.tolist(),'duration':duration}}
    ready['clearance']=checker.path({'start':start,'goal':goal,'duration':duration},161)
    if ready['clearance']['path_minimum_m']<.005:raise ValueError('Opening/startup path too close to table')
    reports=prefix+[ready];synthetic={'command':goal.tolist(),'q':goal.tolist(),'grasp_stage':None}
    import json
    reference=ROOT/'calibration/reach_reference.json'
    if reference.exists():previous=json.loads(reference.read_text())
    else:
        saved=json.loads((ROOT/'captures/mirror_grasp/1789823637866453514/plan.json').read_text())
        previous=next(r for r in saved if r['stage']=='approach')
    if previous['joint_mapping']!=mapping.mapping:raise ValueError('Saved approach mapping changed')
    from reach_retarget import retarget,camera_left_offset
    if packet:
        destination=np.asarray(previous['plan']['goal'],float).copy();destination[5]=goal[5]
        shift=np.asarray(region[0])-np.asarray(previous['surface_center_base_m'])
        height=float(target_profile.get('height_m',.03))
        previous_height=float(previous.get('retarget',{}).get('object_height_m',.03))
        height_shift=height-previous_height
        offset=camera_left_offset(plane,np.asarray(region[0]),target_profile.get('camera_left_offset_m',0.))
        previous_offset=np.asarray(previous.get('retarget',{}).get('grasp_offset_base_m',[0.,0.,0.]))
        target=solver.position(destination)+shift+up*height_shift+offset-previous_offset
        retarget_info={'current_surface_center_base_m':np.asarray(region[0]).tolist(),'target_base_m':target.tolist(),'target_shift_m':shift.tolist(),'object_height_m':height,'height_adjustment_m':height_shift,'grasp_offset_base_m':offset.tolist(),'camera_left_offset_m':target_profile.get('camera_left_offset_m',0.)}
    else:destination,retarget_info=retarget(solver,mapping,previous,region,float(goal[5]))
    if packet:
        solver.bounds=np.asarray(mapping.limits)
        axis=np.asarray(region[2])[:,np.argmax(region[1][:2])]
        target=np.asarray(retarget_info['target_base_m'])
        if reach_only and target_profile.get('position_only_reach',False):
            candidates=[solver.solve(goal,target)]
        else:candidates=solver.solve_natural_grasp(goal,target,up,axis,return_candidates=True)
    else:candidates=[{'goal':destination}]
    failures=[]
    for aligned in candidates:
        destination=aligned['goal'].copy();destination[5]=goal[5]
        info=dict(retarget_info)
        for key in ('jaw_alignment_error_deg','downward_tilt_deg','roll_change_deg','jaw_orientation_sign'):
            if key in aligned:info[key]=aligned[key]
        info.update({'target_name':target_profile.get('name','pink highlighter'),'estimated_width_m':width,'opening_gap_m':opening_gap,'closing_gap_m':closing_gap})
        duration=max(4.,1.875*max(abs(destination-goal))/np.deg2rad(6))
        plan={'start':goal,'goal':destination,'duration':duration};clearance=checker.path(plan,161)
        if clearance['path_minimum_m']<.005 or clearance['gripper_holder_path_minimum_m']<.015:
            failures.append(f"approach table clearance: {clearance['path_minimum_m']*1000:.1f} mm, holder {clearance['gripper_holder_path_minimum_m']*1000:.1f} mm, at {clearance['minimum_at_fraction']:.2f}");continue
        trial=list(reports)+[{'stage':'approach','plan':{'start':goal.tolist(),'goal':destination.tolist(),'duration':duration},'clearance':clearance,
                             'retarget':info,'joint_mapping':mapping.mapping,'source':'Current target; both wrist-roll directions evaluated from actual starting pose'}]
        accelerate_first_eighty(trial,tail_factor=2.)
        if reach_only:
            if target_profile.get('preflight_grip',False):
                from grasp_tail import build_tail
                try:build_tail(model,mapping,plane,destination,closing_gap=closing_gap,lift_height=0,descent_m=target_profile.get('descent_m',.025))
                except ValueError as exc:failures.append(str(exc));continue
            return apply_sequence_speed(trial,target_profile.get('sequence_speed_multiplier',1.))
        from grasp_tail import build_tail
        from grasp_return import append_return
        try:
            trial.extend(build_tail(model,mapping,plane,destination,closing_gap=closing_gap,lift_height=0,descent_m=target_profile.get('descent_m',.015)))
            if target_profile.get('after_grasp')=='basket':
                from basket_place import append_basket
                append_basket(trial,checker,model,mapping,region,up,basket_config if basket_config is not None else json.loads((ROOT/'config/basket.json').read_text()),target_profile=target_profile,return_to_sitting=return_to_sitting)
            else:append_return(trial,checker,model,region,up)
        except ValueError as exc:failures.append(str(exc));continue
        return apply_sequence_speed(trial,target_profile.get('sequence_speed_multiplier',1.))
    raise ValueError('Wrist alternatives checked; no complete path: '+ '; '.join(dict.fromkeys(failures)))


def accelerate_first_eighty(reports,tail_factor=1.):
    """Quadruple playback speed for the first 80% of the whole nominal timeline."""
    boundary=.8*sum(r['plan']['duration'] for r in reports)
    elapsed=0.
    for report in reports:
        plan=report['plan'];fast=max(0.,min(plan['duration'],boundary-elapsed))
        plan['speedup_until_s']=fast
        plan['speedup_factor']=4.
        plan['tail_speedup_factor']=tail_factor
        plan['playback_duration']=(plan['duration']-fast)/tail_factor+fast/4
        elapsed+=plan['duration']
    return reports


def apply_sequence_speed(reports,multiplier):
    multiplier=float(multiplier)
    if not np.isfinite(multiplier) or not 1<=multiplier<=2:raise ValueError('Invalid sequence speed multiplier')
    for report in reports:
        plan=report['plan']
        plan['playback_rate']=multiplier
        plan['playback_duration']=plan.get('playback_duration',plan['duration'])/multiplier
    return reports
