"""Colour/ellipse fallback for the yellow plate when interior features change."""
import cv2,numpy as np


def ellipses(image,color="yellow"):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    if color=='red':
        mask=(((hsv[:,:,0]<=15)|(hsv[:,:,0]>=165))&(hsv[:,:,1]>=80)&(hsv[:,:,2]>=100)).astype(np.uint8)*255
    else:mask=((hsv[:,:,0]>=10)&(hsv[:,:,0]<=35)&(hsv[:,:,1]>=190)&(hsv[:,:,2]>=140)).astype(np.uint8)*255
    mask[:70]=0;mask[410:]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    output=[]
    for c in contours:
        if len(c)<5 or cv2.contourArea(c)<2500:continue
        # The deposited object can cut a hole in the colour mask or cover a
        # section of rim. Fit the exterior hull, not the object's inner edges.
        hull=cv2.convexHull(c)
        if len(hull)<5 or not 7000<cv2.contourArea(hull)<26000:continue
        center,axes,angle=cv2.fitEllipse(hull)
        if min(axes)<90 or max(axes)>210 or max(axes)/min(axes)>1.5:continue
        if cv2.contourArea(hull)/(np.pi*np.prod(axes)/4)<.85:continue
        x,y,w,h=cv2.boundingRect(c)
        if x<3 or y<=70 or x+w>image.shape[1]-3 or y+h>=410:continue
        theta=np.deg2rad(angle);r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        basis=r@np.diag(np.asarray(axes)/2)
        normalized=(hull.reshape(-1,2)-np.asarray(center))@np.linalg.inv(basis).T
        on_rim=np.abs(np.linalg.norm(normalized,axis=1)-1)<.10
        angles=np.sort(np.mod(np.arctan2(normalized[on_rim,1],normalized[on_rim,0]),2*np.pi))
        if len(angles)<8:continue
        gap=max(np.diff(np.r_[angles,angles[0]+2*np.pi]))
        if gap>np.deg2rad(140):continue
        output.append((np.asarray(center),basis))
    return output


def locate_coloured_plate(image,frame,polygon,color="yellow"):
    refs=ellipses(image,color);current=ellipses(frame,color)
    if len(current)!=1:return None
    refs=[r for r in refs if np.linalg.norm(r[0]-polygon.mean(axis=0))<35]
    if len(refs)!=1:return None
    origin,basis=refs[0];center,new_basis=current[0]
    transform=new_basis@np.linalg.inv(basis)
    # Perspective changes pixel size across the table. This is only a coarse
    # image gate; measure_basket validates the calibrated physical dimensions.
    if not .60<np.sqrt(abs(np.linalg.det(transform)))<1.60:return None
    result=(polygon-origin)@transform.T+center
    return result,{'method':f'Fresh visible {color} rim arcs; deposited-object interior excluded','center_pixels':center.tolist()}
