"""Appearance selection for the currently requested grasp target."""
import cv2,numpy as np
from measured_objects import highlighter_proxy

def select_target(frame,contours,regions,profile):
    if profile.get('kind')!='packet':return highlighter_proxy(frame,contours,regions,profile)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);colour=(hsv[:,:,0]>=profile['hsv_hue_min'])&(hsv[:,:,0]<=profile['hsv_hue_max'])&(hsv[:,:,1]>55)&(hsv[:,:,2]>35)
    candidates=[]
    for i,c in enumerate(contours):
        width,length=sorted(2*np.asarray(regions[i][1])[:2])
        if not .025<=width<=.080 or not .06<=length<=.18:continue
        mask=np.zeros(frame.shape[:2],np.uint8);cv2.drawContours(mask,[np.asarray(c,np.int32)],-1,255,-1)
        count=np.count_nonzero(colour&(mask>0))
        if count>=120 and count/max(1,np.count_nonzero(mask))>.25:candidates.append(i)
    return list(regions),candidates[0] if len(candidates)==1 else None,None

def refine_packet_contours(frame,contours,profile):
    """Exclude low-saturation shadows from the packet's metric width estimate."""
    if profile.get('kind')!='packet':return contours
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    colour=((hsv[:,:,0]>=profile['hsv_hue_min'])&(hsv[:,:,0]<=profile['hsv_hue_max'])&(hsv[:,:,1]>55)&(hsv[:,:,2]>35)).astype(np.uint8)
    result=[]
    for c in contours:
        mask=np.zeros(frame.shape[:2],np.uint8);cv2.drawContours(mask,[np.asarray(c,np.int32)],-1,1,-1)
        pixels=cv2.findNonZero(mask&colour)
        if pixels is not None and len(pixels)>=120 and len(pixels)/max(1,np.count_nonzero(mask))>.25:
            result.append(cv2.convexHull(pixels).reshape(-1,2))
        else:result.append(c)
    return result
