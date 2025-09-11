
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

        #builder.add_articulation(key="pendulum")

        self.E_init = 1.5e6
        self.nu_init = 0.45
        self.mu = self.E_init / 2 / (1 + self.nu_init)  # 2797
        self.lam = (  # 17182
            self.E_init * self.nu_init / (1 + self.nu_init) / (1 - 2 * self.nu_init)
        )
        rot_x_90 = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5)

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 1, 0.0),
            rot=rot_x_90,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=5,
            dim_y=5,
            dim_z=5,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=200,
            k_mu=self.mu,
            k_lambda=self.lam,
            k_damp=0.0,
            fix_top=True,
        )

        # add ground plane
        builder.add_ground_plane()

        # disable gravity to prevent falling
        builder.gravity = wp.vec3(0.0, 0.0, 0.0)

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
