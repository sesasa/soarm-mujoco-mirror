"""Provisional fixed-base registration from manually reviewed plate landmarks."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'calibration/table_registration'
    plane=json.loads((out/'board_plane.json').read_text())
    intr=json.loads((ROOT/'calibration/cameras/overhead_640x480.json').read_text())
    K=np.array(intr['matrix']);dist=np.array(intr['distortion'])
    # Board Z points into the table. Flip Y and Z for a right-handed Z-up frame.
    flip=np.diag([1.,-1.,-1.,1.]);ct=np.array(plane['T_camera_from_board'])@flip
    # Visible plate tip, outer shoulder and rear outer corner, raw 640x480 pixels.
    uv=np.array([[439.5,348.5],[418.5,390.5],[416.,427.]])
    points=np.array([[.064635,.03063,0],[.019435,.0554625,0],[-.022365,.04077,0]])
    ray=cv2.undistortPoints(np.array([[[465.,420.]]]),K,dist,P=K)
    b=cv2.perspectiveTransform(ray,np.array(plane['undistorted_pixel_to_board_xy_m_homography']))[0,0]
    def transform(p):
        x,y,a=p;c,s=np.cos(a),np.sin(a);T=np.array([[c,-s,0,x],[s,c,0,y],[0,0,1,.0024],[0,0,0,1.]])
        return T
    def project(p,xyz):
        T=ct@transform(p);r=cv2.Rodrigues(T[:3,:3])[0]
        return cv2.projectPoints(xyz,r,T[:3,3],K,dist)[0].reshape(-1,2)
    fits=[least_squares(lambda p:(project(p,points)-uv).ravel(),[b[0],-b[1],a]) for a in np.linspace(-np.pi,np.pi,8)]
    fit=min(fits,key=lambda f:f.cost);pred=project(fit.x,points)
    T=ct@transform(fit.x);camera=-T[:3,:3].T@T[:3,3]
    # Sensitivity to 2px manual picking noise, not a full uncertainty estimate.
    rng=np.random.default_rng(101);trials=[]
    for _ in range(100):
        noisy=uv+rng.normal(0,2,uv.shape)
        f=least_squares(lambda p:(project(p,points)-noisy).ravel(),fit.x)
        trials.append(f.x.tolist())
    result={'status':'provisional_manual_plate_fit_not_for_robot_commands','image':'captures/base/registration_raw.png','reference_phone':'Photos-1-001/base/IMG_8803.heic','landmark_names':['plate forward outer tip','plate outer shoulder midpoint','plate rear outer corner'],'model_points_base_m':points.tolist(),'raw_pixels':uv.tolist(),'projected_pixels':pred.tolist(),'rms_px':float(np.sqrt(np.mean(np.sum((pred-uv)**2,axis=1)))),'T_table_from_base':transform(fit.x).tolist(),'T_camera_from_base':T.tolist(),'camera_position_base_m':camera.tolist(),'manual_2px_noise_parameter_std':np.std(trials,axis=0).tolist(),'limitations':['Three manually selected landmarks on only one plate side; independent validation required.','Camera calibration has limited lower-image coverage where base is located.','Assumes CAD plate matches physical print and rests flat on tabletop.','Base origin elevated 2.4 mm so lowest CAD plate surface rests on table.','Does not change joint mapping or authorize motion.']}
    (out/'base_registration_candidate.json').write_text(json.dumps(result,indent=2)+'\n')
    frame=cv2.imread(str(ROOT/'captures/base/registration_raw.png'))
    for i,(a,b) in enumerate(zip(uv,pred)):
        cv2.circle(frame,tuple(np.rint(a).astype(int)),4,(0,255,0),1)
        cv2.drawMarker(frame,tuple(np.rint(b).astype(int)),(0,0,255),cv2.MARKER_CROSS,8,1)
    # Draw the CAD plate's two wings, including the partly hidden side.
    for sign in [1,-1]:
        line=np.array([[-.022365,sign*.04077,0],[.015435,sign*.0554625,0],[.023435,sign*.0554625,0],[.064635,sign*.031,0],[.064635,sign*.025,0]])
        px=project(fit.x,line);cv2.polylines(frame,[np.rint(px).astype(np.int32)],False,(255,180,0),1)
    cv2.drawFrameAxes(frame,K,dist,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],.06)
    cv2.imwrite(str(out/'base_fit_overlay.jpg'),frame)
    print(json.dumps({k:result[k] for k in ['rms_px','camera_position_base_m','manual_2px_noise_parameter_std']}))

if __name__=='__main__':main()
