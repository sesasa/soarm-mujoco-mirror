"""Sampled visual-geometry/table clearance; not collision certification."""
import mujoco
import numpy as np
from simulated_reach import JOINTS,trajectory
from mesh_support import mesh_hulls

class TableClearance:
    def __init__(self,model,plane_point,normal):
        self.model=model;self.data=mujoco.MjData(model);self.mesh_hulls=mesh_hulls(model)
        self.point=np.asarray(plane_point,float)
        self.normal=np.asarray(normal,float);self.normal/=np.linalg.norm(self.normal)
        if self.normal[2]<0:self.normal=-self.normal
        self.addresses=[model.joint(n).qposadr[0] for n in JOINTS]
        self.geoms=[];self.tool_geoms=set()
        gripper=model.body("gripper").id
        for i in range(model.ngeom):
            body=int(model.geom_bodyid[i]);moving=False;tool=False
            while body:
                moving|=bool(model.body_jntnum[body]);tool|=body==gripper;body=int(model.body_parentid[body])
            if moving and model.geom_group[i]==2 and model.geom_rgba[i,3]>0:
                self.geoms.append(i)
                if tool:self.tool_geoms.add(i)

    def set_pose(self,q):
        self.data.qpos[self.addresses]=q;mujoco.mj_forward(self.model,self.data)

    def at(self,q):
        self.set_pose(q)
        best=(float('inf'),None);tool_min=float('inf')
        for i in self.geoms:
            rotation=self.data.geom_xmat[i].reshape(3,3)
            n=rotation.T@self.normal
            center=float((self.data.geom_xpos[i]-self.point)@self.normal)
            kind=self.model.geom_type[i];size=self.model.geom_size[i]
            if kind==mujoco.mjtGeom.mjGEOM_MESH:
                mesh=int(self.model.geom_dataid[i])
                height=center+float(np.min(self.mesh_hulls[mesh]@n))
            elif kind==mujoco.mjtGeom.mjGEOM_BOX:height=center-float(np.abs(n)@size)
            elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:height=center-size[0]
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:height=center-size[0]-size[1]*abs(n[2])
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:height=center-size[0]*np.linalg.norm(n[:2])-size[1]*abs(n[2])
            elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:height=center-np.linalg.norm(size*n)
            else:raise ValueError(f'Unsupported clearance geometry {kind}')
            if i in self.tool_geoms:tool_min=min(tool_min,float(height))
            if height<best[0]:best=(float(height),i)
        height,i=best
        if i is None:raise ValueError('No moving visual geometry to check')
        name=self.model.geom(i).name
        if not name and self.model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
            name=self.model.mesh(int(self.model.geom_dataid[i])).name
        return {'minimum_m':height,'gripper_holder_minimum_m':tool_min,'geom':name or str(i),'body':self.model.body(int(self.model.geom_bodyid[i])).name}

    def path(self,plan,samples=81):
        results=[self.at(trajectory(plan,t)) for t in np.linspace(0,plan['duration'],samples)]
        index=min(range(samples),key=lambda i:results[i]['minimum_m'])
        return {'method':'sampled moving visual geometry vs infinite tabletop plane',
                'samples':samples,'path_minimum_m':results[index]['minimum_m'],
                'gripper_holder_path_minimum_m':min(r['gripper_holder_minimum_m'] for r in results),'minimum_at_fraction':index/(samples-1),'closest_body':results[index]['body'],
                'closest_geom':results[index]['geom'],'endpoint':results[-1],
                'status':'sampled intersection' if results[index]['minimum_m']<0 else 'no intersection in sampled poses',
                'collision_validated':False,
                'limitations':['Provisional camera/base/table registration and approximate holder shape.',
                              'Fixed base excluded. Table objects, cables and self-collision not checked.',
                              'Discrete samples do not establish continuous path clearance.']}
