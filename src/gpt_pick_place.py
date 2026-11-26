"""Compose existing grasp/plate planners using a fresh overhead frame."""
import json
import numpy as np
from command_mapping import ROOT
from basket_tracking import measure_basket
from basket_place import append_basket
from grasp_tail import build_tail
from table_clearance import TableClearance


def current_plate(frame,plane,camera_from_base):
    reference=json.loads((ROOT/'config/basket_reference.json').read_text())
    # Always measure from this request's image, never use config/basket.json's old location.
    try:
        return measure_basket(frame,reference,plane,camera_from_base)
    except ValueError as exc:
        raise ValueError(str(exc).replace('Basket','Plate').replace('basket','plate')) from exc


def build_pick_place_tail(model,mapping,plane,hover,region,profile,plate):
    reports=build_tail(model,mapping,plane,np.asarray(hover),closing_gap=profile['closing_gap_m'],lift_height=0,descent_m=profile.get('descent_m',.025))
    up=plane.base_from_board[:3,2].copy()
    if up[2]<0:up=-up
    checker=TableClearance(model,plane.base_from_board[:3,3],up)
    append_basket(reports,checker,model,mapping,region,up,plate,target_profile=profile,return_to_sitting=True)
    return reports


def accelerate_fixed_playback(reports):
    """Slow all stages by 1.5 relative to the previous approach/other-stage rates."""
    import copy
    result=copy.deepcopy(reports)
    for report in result:
        plan=report['plan']
        rate=(2. if report['stage']=='approach' else 4.)/1.5
        plan['execution_rate']=rate
        plan['playback_duration']=plan.get('playback_duration',plan['duration'])/rate
    return result
