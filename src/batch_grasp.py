"""Fresh-frame selection and planning for the packet followed by the small box."""
import copy,json
import cv2,numpy as np
from command_mapping import ROOT
from basket_tracking import measure_basket
from mirror_grasp_plan import build_sequence


def table_targets(contours,regions,basket):
    plate=np.asarray(basket['inner_rim_pixels'],np.float32)
    candidates=[]
    for contour,region in zip(contours,regions):
        contour=np.asarray(contour,np.float32)
        width,length=sorted(2*np.asarray(region[1])[:2])
        if not (.025<=width<=.080 and .040<=length<=.18 and length/width<4):continue
        hull=cv2.convexHull(contour).reshape(-1,2)
        overlap=cv2.intersectConvexConvex(hull,cv2.convexHull(plate))[0]
        if overlap/max(cv2.contourArea(hull),1)> .2:continue
        kind='packet' if length>=.080 else 'box'
        candidates.append({'key':kind,'region':region,'width_m':float(width),'length_m':float(length),'contour':contour})
    return candidates


def prepare_batch(model,mapping,plane,state,frame,contours,regions,camera_from_base,completed,preview_remaining=True):
    profile=json.loads((ROOT/'config/grasp_target.json').read_text())
    reference=json.loads((ROOT/'config/basket_reference.json').read_text())
    basket=measure_basket(frame,reference,plane,camera_from_base)
    candidates=table_targets(contours,regions,basket)
    if completed:basket['occupied_height_m']=float(profile['height_m'])
    for key in completed:
        if any(c['key']==key for c in candidates):raise ValueError(f'The {key} is still visible outside the plate; inspect the previous grasp')
    wanted=[key for key in ('packet','box') if key not in completed]
    for key in wanted:
        if sum(c['key']==key for c in candidates)!=1:raise ValueError(f'Need one clearly separated {key} outside the plate')
    selected=next(c for c in candidates if c['key']==wanted[0])
    target_profile=copy.deepcopy(profile)
    if selected['key']=='box':
        box=json.loads((ROOT/'config/batch_objects.json').read_text())['box']
        if box.get('height_m') is None:raise ValueError('Small box height needed before its grasp can be planned')
        target_profile.update(name='small box',height_m=box['height_m'],closing_gap_m=max(.005,selected['width_m']-.004))
    if len(wanted)==2:basket['next_target_pixels']=next(c['contour'].tolist() for c in candidates if c['key']=='box')
    plans=build_sequence(model,mapping,plane,state,selected['region'],basket_config=basket,target_profile_override=target_profile,return_to_sitting=len(wanted)==1)
    preview_plans=[dict(r,preview_target=selected['key']) for r in plans]
    if len(wanted)==2 and preview_remaining:
        second=next(c for c in candidates if c['key']=='box')
        box=json.loads((ROOT/'config/batch_objects.json').read_text())['box']
        if box.get('height_m') is None:raise ValueError('Small box height needed for complete preview')
        second_profile=dict(profile,name='small box',height_m=box['height_m'],closing_gap_m=max(.005,second['width_m']-.004))
        second_start=plans[-1]['plan']['goal']
        second_state={**state,'command':second_start,'q':second_start}
        occupied=copy.deepcopy(basket);occupied['occupied_height_m']=float(profile['height_m'])
        second_plans=build_sequence(model,mapping,plane,second_state,second['region'],basket_config=occupied,target_profile_override=second_profile,return_to_sitting=True)
        preview_plans.extend(dict(r,preview_target='box') for r in second_plans)
    return {'plans':plans,'preview_plans':preview_plans,'target':selected['key'],'basket':basket,'remaining':len(wanted),'region':[np.asarray(v).tolist() for v in selected['region']],'profile':target_profile}
