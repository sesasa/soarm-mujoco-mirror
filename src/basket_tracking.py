"""Locate the basket anew in a lens-corrected planning frame."""
import copy
import cv2
import numpy as np
from command_mapping import ROOT
from scene_changes import Plane


def locate_basket(frame,reference):
    image=cv2.imread(str(ROOT/reference['source_image']))
    if image is None:raise ValueError('Basket reference image missing')
    image=image[:frame.shape[0],:frame.shape[1]]
    polygon=np.asarray(reference['inner_rim_pixels'],np.float32)
    def features(im):
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray-cv2.GaussianBlur(gray,(0,0),5)
    feature_mask=np.zeros(image.shape[:2],np.uint8)
    cv2.fillPoly(feature_mask,[polygon.astype(np.int32)],255)
    feature_mask=cv2.dilate(feature_mask,np.ones((11,11),np.uint8))
    sift=cv2.SIFT_create(contrastThreshold=.01)
    key_a,desc_a=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),feature_mask)
    key_b,desc_b=sift.detectAndCompute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),None)
    if desc_a is not None and desc_b is not None and len(desc_b)>=2:
        pairs=cv2.BFMatcher().knnMatch(desc_a,desc_b,k=2)
        matches=[a for a,b in pairs if a.distance<.7*b.distance]
        if len(matches)>=8:
            source=np.float32([key_a[m.queryIdx].pt for m in matches])
            target=np.float32([key_b[m.trainIdx].pt for m in matches])
            transform,inliers=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=2)
            if transform is not None and int(inliers.sum())>=8:
                spread=np.ptp(source[inliers.ravel().astype(bool)],axis=0)
                scale=float(np.linalg.norm(transform[:,0]))
                if np.all(spread>np.ptp(polygon,axis=0)*.35) and .75<scale<1.25:
                    pixels=cv2.transform(polygon[None],transform)[0]
                    if np.all(pixels>2) and np.all(pixels[:,0]<frame.shape[1]-3) and np.all(pixels[:,1]<frame.shape[0]-3):
                        return pixels,{'method':'Fresh distributed basket feature matches','inliers':int(inliers.sum()),'image_scale':scale}
    if reference.get('kind')=='plate':
        from plate_detection import locate_coloured_plate
        detected=locate_coloured_plate(image,frame,polygon,reference.get("plate_color","yellow"))
        if detected is not None:return detected
    x,y,w,h=cv2.boundingRect(polygon);pad=12
    x=max(0,x-pad);y=max(0,y-pad);w=min(image.shape[1]-x,w+2*pad);h=min(image.shape[0]-y,h+2*pad)
    support=np.zeros(image.shape[:2],np.uint8);cv2.fillPoly(support,[polygon.astype(np.int32)],1)
    support=cv2.dilate(support,np.ones((11,11),np.uint8))
    template=(features(image)*support)[y:y+h,x:x+w];live=features(frame)
    # Diagnostic images have black bars; prevent their edges entering matching.
    if np.mean(frame[:50])<50:live[:70]=0;live[415:]=0
    # Exclude UI bars in saved diagnostic images; real planning frames have none.
    best=None
    def search(angles,scales):
        nonlocal best
        for angle in angles:
            for scale in scales:
                transform=cv2.getRotationMatrix2D((w/2,h/2),float(angle),float(scale))
                co,si=abs(transform[0,0]),abs(transform[0,1]);nw=int(h*si+w*co);nh=int(h*co+w*si)
                if nw>=frame.shape[1] or nh>=frame.shape[0]:continue
                transform[:,2]+=[nw/2-w/2,nh/2-h/2]
                rotated=cv2.warpAffine(template,transform,(nw,nh))
                scores=cv2.matchTemplate(live,rotated,cv2.TM_CCOEFF_NORMED)
                sums=cv2.integral(live);squares=cv2.integral(live*live)
                def window_sum(integral):return integral[nh:,nw:]-integral[:-nh,nw:]-integral[nh:,:-nw]+integral[:-nh,:-nw]
                variance=window_sum(squares)/(nw*nh)-(window_sum(sums)/(nw*nh))**2
                scores[variance<max(.25,float(np.var(rotated))*.1)]=-1
                _,score,_,location=cv2.minMaxLoc(scores)
                if best is None or score>best[0]:best=(score,angle,scale,location,transform,scores)
    search(range(-90,91,10),(.8,.9,1.,1.1,1.2))
    _,angle,scale,*_=best
    search(np.arange(angle-6,angle+7,2),np.arange(scale-.06,scale+.061,.02))
    score,angle,scale,location,transform,scores=best
    if not np.isfinite(score) or score<.50:raise ValueError(f'Basket not clearly visible in current image (match {score:.2f}, angle {angle}, location {location}); clear its rim')
    competitors=scores.copy();lx,ly=location
    competitors[max(0,ly-50):ly+51,max(0,lx-50):lx+51]=-1
    if np.max(competitors)>score-.08:raise ValueError(f'Basket location ambiguous in current image: {score:.3f} vs {np.max(competitors):.3f} at {location}, competitor {cv2.minMaxLoc(competitors)[3]}')
    current=cv2.transform((polygon-[x,y])[None].astype(np.float32),transform)[0]+location
    if np.any(current<2) or np.any(current[:,0]>frame.shape[1]-3) or np.any(current[:,1]>frame.shape[0]-3):raise ValueError('Basket rim extends out of view')
    return current,{'match_score':float(score),'image_rotation_deg':float(angle),'image_scale':float(scale),'method':'Fresh full-frame appearance match to measured basket reference'}


def measure_basket(frame,reference,plane,camera_from_base):
    pixels,evidence=locate_basket(frame,reference)
    raised=plane.base_from_board.copy();up=raised[:3,2].copy()
    if up[2]<0:up=-up
    measured_height=reference.get('measurement_height_m',reference['rim_height_m'])
    raised[:3,3]+=up*measured_height
    rim=Plane(plane.K,camera_from_base,camera_from_base@raised)
    xy=rim.board_xy(pixels)
    world=(raised@np.c_[xy,np.zeros(len(xy)),np.ones(len(xy))].T).T[:,:3]
    center,size,rotation=rim.footprint(pixels)
    allowance=up*(reference['rim_height_m']-measured_height)
    world+=allowance;center+=allowance
    ratio=np.sort(2*size[:2])/np.sort(reference['opening_size_m'])
    if np.any(ratio<.75) or np.any(ratio>1.25):raise ValueError('Detected basket dimensions differ from the measured reference')
    config=copy.deepcopy(reference)
    config.update(rim_polygon_base_m=world.tolist(),center_base_m=center.tolist(),rotation=rotation.tolist(),opening_size_m=(2*size[:2]).tolist(),inner_rim_pixels=pixels.tolist(),fixed_position=False,detection=evidence)
    return config
