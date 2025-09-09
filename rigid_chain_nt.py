
###########################################################################
# Rigid bodies connected by revolute joints (Newton)
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples


class Chain:
    def __init__(self, viewer):
        # simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        # simulation substeps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # number of iterations
        self.iterations = 10

        # viewer
        self.viewer = viewer

        # model builder
        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        builder.add_articulation(key="pendulum")

        hx = 0.3 # length
        hy = 0.02 # width
        hz = 0.02 # height

        #cfg = newton.ModelBuilder.JointDofConfig

        self.chain_length = 15

        # create links
        self.links = []
        for i in range(self.chain_length):
            link = builder.add_body() #body index
            builder.add_shape_box(
                link, 
                hx=hx, 
                hy=hy, 
                hz=hz,
            )
            self.links.append(link)

        # add joints (world to first link, then between consecutive links)
        for i, link in enumerate(self.links):
            if i == 0:
                #continue
                parent = -1
                parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 10.0), q=wp.quat_identity()) # initial height of the first link
                child_xform = wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity())
            else:
                parent = self.links[i - 1]
                parent_xform = wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity())
                child_xform = wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity())
            
            builder.add_joint_revolute(
                parent=parent,
                child=link,
                axis=wp.vec3(0.0,1.0,0.0),
                parent_xform=parent_xform,
                child_xform=child_xform,
                target_ke=0.0,
                target_kd=0.05,
                limit_lower=-2*np.pi,
                limit_upper=2*np.pi,
                limit_ke=50.0,
                limit_kd=2.0,
                armature=0.005,
                friction=0.05
            )

            '''builder.add_joint_d6(
                parent=parent,
                child=link,
                linear_axes=[
                    cfg(axis=newton.Axis.X),
                    cfg(axis=newton.Axis.Y), 
                    cfg(axis=newton.Axis.Z)
                ],
                angular_axes=[
                    cfg(axis=newton.Axis.X),
                    cfg(axis=newton.Axis.Y),
                    cfg(axis=newton.Axis.Z)
                ],
                parent_xform=parent_xform,
                child_xform=child_xform,
            )'''
            

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0) # initialize contacts

        self.viewer.set_model(self.model)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0) # evaluate forward kinematics

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:     # if cuda is available, capture the graph
            with wp.ScopedCapture() as capture:     # capture the graph
                self.simulate() # simulate the model
            self.graph = capture.graph  
        else:
            self.graph = None # if cuda is not available, set the graph to None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces() # clear forces

            # apply forces to the model
            self.viewer.apply_forces(self.state_0) 

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Chain(viewer)

    newton.examples.run(example)
