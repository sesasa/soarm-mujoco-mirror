"""Measured highlighter visual proxy; colour association is deliberately limited."""
import cv2
import numpy as np

def highlighter_proxy(frame,contours,regions,profile):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    pink=((hsv[:,:,0]<=6)|(hsv[:,:,0]>=170))&(hsv[:,:,1]>100)&(hsv[:,:,2]>100)
    candidates=[]
    for index,contour in enumerate(contours):
        mask=np.zeros(frame.shape[:2],np.uint8)
        cv2.drawContours(mask,[np.asarray(contour,np.int32)],-1,255,-1)
        pixels=mask>0
        count=int(np.count_nonzero(pink&pixels))
        if count>=120 and count/max(int(pixels.sum()),1)>.18:candidates.append(index)
    display=list(regions)
    if len(candidates)!=1:return display,None,None
    index=candidates[0];center,size,rotation=regions[index]
    center=np.asarray(center,float);size=np.asarray(size,float).copy();rotation=np.asarray(rotation,float).copy()
    long_axis=int(np.argmax(size[:2]));short_axis=1-long_axis
    length=float(2*size[long_axis])
    width=float(profile['width_m']);height=float(profile['height_m'])
    if not np.isfinite([length,width,height]).all() or width<=0 or height<=0 or length<1.5*width:
        return display,None,None
    up=rotation[:,2] if rotation[2,2]>0 else -rotation[:,2]
    size[short_axis]=width/2;size[2]=height/2
    display[index]=(center+up*height/2,size,rotation)
    report={'name':'pink_highlighter','height_m':height,'width_m':width,'estimated_length_m':length,
            'surface_center_base_m':center.tolist(),'center_base_m':display[index][0].tolist(),
            'association':'unique pink colour candidate','shape':'rectangular visual proxy',
            'pose_and_length_provisional':True,'collision_validated':False}
    return display,index,report
