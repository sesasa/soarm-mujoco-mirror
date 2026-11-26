"""Read-only wrist feature tracking for commissioning local image-based grasping.

Tracks a manually selected rigid object patch, not its semantic identity. Never
converts pixels to robot commands: the image/motion map is not calibrated yet.
"""
import threading,time
import cv2
import numpy as np


class WristView:
    def __init__(self, spec):
        self.cap=cv2.VideoCapture(spec['path'],cv2.CAP_V4L2)
        if not self.cap.isOpened():raise RuntimeError('Wrist camera unavailable')
        self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*spec['fourcc']))
        self.cap.set(3,640);self.cap.set(4,480);self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.lock=threading.Lock();self.sample=None;self.stop=threading.Event()
        self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()
    def _read(self):
        while not self.stop.is_set():
            ok,frame=self.cap.read()
            with self.lock:self.sample=(time.monotonic(),frame if ok else None)
            if not ok:break
    def latest(self):
        with self.lock:return self.sample
    def close(self):
        self.stop.set();self.thread.join(timeout=1)
        self.cap.release()


class PatchTracker:
    """Forward/backward LK plus robust similarity fitting; fails closed on loss."""
    def __init__(self):self.reset()
    def reset(self):
        self.previous=None;self.points=None;self.box=None;self.timestamp=None
        self.reason='Drag a rectangle tightly around the object in the wrist view'
    def select(self,frame,rect,timestamp):
        self.reset()
        x0,y0,x1,y1=map(int,rect)
        x0,x1=sorted((max(0,min(639,x0)),max(0,min(639,x1))))
        y0,y1=sorted((max(0,min(479,y0)),max(0,min(479,y1))))
        if x1-x0<12 or y1-y0<12:return False
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);mask=np.zeros_like(gray)
        mask[y0:y1,x0:x1]=255
        points=cv2.goodFeaturesToTrack(gray,80,.02,5,mask=mask)
        if points is None or len(points)<8:
            self.reason='Too little texture: include printed markings on the object';return False
        self.previous=gray;self.points=points
        self.box=np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        self.timestamp=timestamp;self.reason='Tracking selected patch (observation only)';return True
    def update(self,frame,timestamp,now):
        if self.box is None:return None
        if not 0<=now-timestamp<=.25:return self._lost('Wrist frame stale; select object again')
        if timestamp==self.timestamp:return self.features()
        if timestamp<self.timestamp or timestamp-self.timestamp>.25:return self._lost('Wrist frames interrupted; select again')
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        nxt,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,self.points,None)
        if nxt is None:return self._lost('Object patch lost; select again')
        back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,nxt,None)
        if back is None:return self._lost('Object patch lost; select again')
        good=(status.ravel()!=0)&(back_status.ravel()!=0)&(np.linalg.norm(back-self.points,axis=2).ravel()<1.)
        if good.sum()<8:return self._lost('Too few reliable object features; select again')
        matrix,inliers=cv2.estimateAffinePartial2D(self.points[good],nxt[good],method=cv2.RANSAC,ransacReprojThreshold=2.)
        if matrix is None or inliers.sum()<8 or inliers.mean()<.65:return self._lost('Object motion ambiguous; select again')
        scale=np.linalg.norm(matrix[:2,0])
        if not .85<scale<1.18:return self._lost('Object scale changed too quickly; select again')
        box=cv2.transform(self.box[None],matrix)[0]
        if np.any(box[:,0]<3) or np.any(box[:,0]>636) or np.any(box[:,1]<3) or np.any(box[:,1]>476):return self._lost('Object leaving wrist view; select again')
        self.box=box;self.points=nxt[good][inliers.ravel().astype(bool)].reshape(-1,1,2)
        self.previous=gray;self.timestamp=timestamp
        return self.features()
    def _lost(self,reason):
        self.reset();self.reason=reason;return None
    def features(self):
        edge=self.box[1]-self.box[0]
        return dict(center_px=self.box.mean(axis=0).tolist(),rotation_deg=float(np.degrees(np.arctan2(edge[1],edge[0]))),points=len(self.points))
    def draw(self,frame,tips):
        out=frame.copy()
        if self.box is not None:
            cv2.polylines(out,[np.rint(self.box).astype(np.int32)],True,(0,255,0),2)
            cv2.circle(out,tuple(np.rint(self.box.mean(axis=0)).astype(int)),4,(0,255,0),-1)
        for p in tips:cv2.circle(out,tuple(p),5,(0,220,255),2)
        if len(tips)==2:
            mid=np.rint(np.mean(tips,axis=0)).astype(int)
            cv2.line(out,tuple(tips[0]),tuple(tips[1]),(0,220,255),1)
            cv2.drawMarker(out,tuple(mid),(0,220,255),cv2.MARKER_CROSS,18,2)
        cv2.putText(out,'WRIST SETUP: observation only',(8,85),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,255),1)
        cv2.putText(out,self.reason,(8,108),cv2.FONT_HERSHEY_SIMPLEX,.40,(0,220,255),1)
        cv2.putText(out,'Right-click each fingertip contact point; drag to select object',(8,130),cv2.FONT_HERSHEY_SIMPLEX,.40,(0,220,255),1)
        return out


class WristContinuation:
    """Operator-bound patch visibility lease; does not estimate robot motion."""
    def __init__(self):self.observation=None;self.until=None
    def start(self,observation,features,frame_time,now,remaining_m):
        if features is None or not 0<=now-frame_time<=.25:raise ValueError('Select a fresh wrist object patch first')
        if not np.isfinite(remaining_m) or not 0<=remaining_m<=.06:raise ValueError('Wrist continuation requires the final 6 cm of approach')
        self.observation=observation;self.until=now+20.
    def valid(self,features,frame_time,now):
        return (self.observation is not None and features is not None and self.until is not None
                and now<=self.until and 0<=now-frame_time<=.25)
