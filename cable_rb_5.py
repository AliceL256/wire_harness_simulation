###########################################################################
# Deformable cable on a roll ball (warp)
###########################################################################

import os
import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
from warp.sim.model import ModelBuilder
from integrator_euler_custom import SemiImplicitCustomIntegrator,  wrap_fem_state
#import cv2

NAN = wp.constant(-1.0e8)
PARTICLE_FLAG_ACTIVE = wp.constant(wp.uint32(1 << 0))

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

norm = Normalize(vmin=0, vmax=20)
sm = ScalarMappable(norm=norm, cmap="cool")

gl_s = 20
global_size = 20


def visualize_tactile_image(tactile_array):
    resolution = 30
    nrows = tactile_array.shape[0]
    ncols = tactile_array.shape[1]
    # print("tactile_array max", np.max(tactile_array, axis=1))

    # Create image array with uint8 type (required by OpenCV)
    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=np.uint8)

    for row in range(nrows):
        for col in range(ncols):
            # Starting point of arrow
            start_x = col * resolution + resolution // 2
            start_y = row * resolution + resolution // 2

            # Calculate end point based on force vector
            scale = 0.1  # Scale factor for visualization
            offset = tactile_array[row, col] * scale * resolution
            end_x = int(start_x + offset[0] + 1)
            end_y = int(start_y + offset[2] + 1)

            # Calculate color based on normal force
            force_z = tactile_array[row, col][1]
            color = sm.to_rgba(force_z)
            b = color[2] * 255
            g = color[1] * 255
            r = color[0] * 255

            # Draw arrow
            '''cv2.arrowedLine(
                imgs_tactile,
                (start_x, start_y),  # Start point (x,y)
                (end_x, end_y),  # End point (x,y)
                color=(b, g, r),  # BGR color
                thickness=2,
                tipLength=0.3,
            )'''

    return imgs_tactile


@wp.kernel
def acturate_cube(
    particle_q: wp.array(dtype=wp.vec3), #粒子位置数组
    ver_start: wp.int32, # 起始顶点索引
    ind: wp.int32, # 当前时间步索引
    actuator_params: wp.array(dtype=wp.vec3), # 执行器参数数组
    particle_qd: wp.array(dtype=wp.vec3), # 粒子速度数组
):
    tid = wp.tid() # 当前线程ID
    if ind > gl_s * 50: # 如果当前时间步索引大于20*50，则使用第二个控制参数
        vel_control = actuator_params[1] # 控制参数
    else:
        # vel_control = actuator_params[0]
        vel_control = wp.vec3(0.0, -0.5, 0.0)
    particle_qd[tid + ver_start] = wp.vec3(
        vel_control[0], vel_control[1], vel_control[2]
    )


class RollBall:
    def __init__(self, stage_path="example_surface.usd", verbose=False, num_frames=300):
        self.verbose = verbose

        fps = 60
        self.frame_dt = 1.0 / fps
        self.num_frames = num_frames

        self.sim_substeps = 50
        self.sim_dt = self.frame_dt / self.sim_substeps
        print("sim_dt", self.sim_dt)
        self.sim_time = 0.0
        self.render_time = 0.0
        self.profiler = {}
        self.sim_steps = self.num_frames * self.sim_substeps

        builder = ModelBuilder()

        builder.default_particle_radius = 0.01

        body_1 = builder.add_body(
            origin=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        )
        builder.add_shape_sphere(
            body=body_1,
            pos=wp.vec3(0, 0.5, 1.0),
            radius=0.5,
            density=100,
            ke=1.0, # 弹性模量，值越高，球体越不容易变形
            kf=0.1, # 摩擦系数，值越高，摩擦越大
            kd=0.1, # 阻尼系数，值越高，能量损失越快
            mu=0.1, # 摩擦系数
        )

        self.ball_pos = [0.0, 20.0, 0.0]
        self.ball_ori = [-90.0, 0.0, 0.0]
        self.ball_vel = [0.0, 0.0, 0.0]

        act_init = np.zeros((2, 3))

        start_ind = 0
        act_init[0] = [0.0, -0.5, 0.0]
        act_init[1] = [0.0, 0.0, 0.3]

        print("start_ind", start_ind, self.sim_steps)
        self.actuator_params = wp.array(act_init, dtype=wp.vec3, requires_grad=False)

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
            dim_x=1,
            dim_y=20,
            dim_z=1,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=200,
            k_mu=self.mu,
            k_lambda=self.lam,
            k_damp=0.0,
            tri_ke=25.0,
            tri_ka=25.0,
            tri_kd=5.0,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
        )

        self.grid_num = builder.particle_count
        print("grid_num", self.grid_num)

        self.model = builder.finalize(requires_grad=False)
        #self.model.ground = True

        self.model.soft_contact_kf = 1000  # kt
        self.model.soft_contact_margin = 0.001
        self.model.rigid_contact_margin = 0.001

        print("particle mass", len(self.model.particle_mass))

        self.integrator = SemiImplicitCustomIntegrator()

        self.states = []
        for _i in range(self.sim_steps + 1):
            self.states.append(wrap_fem_state(self.model, False))

        stage_path = "cable_rb_5.usd"
        self.renderer = CustomSimRenderer(
            self.model, stage_path, scaling=1.0, draw_marker=True
        )

        self.use_cuda_graph = wp.get_device().is_cuda and False
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for i in range(self.sim_steps):
            wp.sim.collide(self.model, self.states[i])
            self.states[i].clear_forces()
            wp.launch(
                kernel=acturate_cube,
                dim=global_size,
                inputs=[
                    self.states[i].particle_q,
                    0,
                    i,
                    self.actuator_params,
                    self.states[i].particle_qd,
                ],
            )

            self.integrator.simulate(
                self.model, self.states[i], self.states[i + 1], self.sim_dt
            )

    def run(self):
        with wp.ScopedTimer("step", dict=self.profiler):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        os.makedirs("tactile_images", exist_ok=True)
        print("render", self.num_frames)
        with wp.ScopedTimer("render"):
            for i in range(self.num_frames + 1):
                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.states[i * self.sim_substeps])
                self.renderer.end_frame()

                self.render_time += self.frame_dt


class CustomSimRenderer(wp.sim.render.SimRenderer):
    def __init__(self, model, path, draw_marker=False, **kwargs):
        super().__init__(model, path, **kwargs)
        self.show_soft_contact_points = False
        self.soft_contact_points_radius = 0.01
        self.draw_marker = draw_marker
        self.i = 0
        if self.show_soft_contact_points:
            self.soft_contact_points_0 = wp.zeros(model.soft_contact_max, dtype=wp.vec3)
            self.soft_contact_points_1 = wp.zeros(model.soft_contact_max, dtype=wp.vec3)
            self.soft_contact_colors = [(1.0, 0.0, 0.0)] * model.soft_contact_max

            indices_beg = np.arange(model.soft_contact_max - 1)
            indices_end = indices_beg + model.soft_contact_max
            self.soft_contact_vis_indices = np.vstack(
                (indices_beg.flatten(), indices_end.flatten())
            ).T.flatten()

    def render(self, state):
        super().render(state)
        if self.draw_marker:
            forces = state.particle_f_debug.numpy()
            pos = state.particle_q.numpy()[-global_size:, :]
            bottom_layer = forces[-global_size * 1 :, :]
            tactile_forces = bottom_layer.reshape((1, gl_s, 3))
            # print("tactile_forces", tactile_forces)
            tactile_forces_img = visualize_tactile_image(tactile_forces)
            '''cv2.imwrite(
                f"tactile_images/tactile_forces_{self.i:04d}.png",
                (tactile_forces_img).astype(np.uint8),
            )'''
            self.i += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Override the default Warp device."
    )
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_walker.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=200,
        help="Total number of frames per training iteration.",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=1,
        help="Total number of training iterations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out additional status messages during execution.",
    )
    parser.add_argument(
        "--save_path", type=str, default="results", help="Path to save the results."
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="surface_follow",
        help="Name of the saved results.",
    )

    args = parser.parse_known_args()[0]

    if not os.path.exists(args.save_path):
        print(f"save_path {args.save_path} does not exist, creating it")
        os.makedirs(args.save_path, exist_ok=True)

    with wp.ScopedDevice(args.device):
        example = RollBall(num_frames=args.num_frames)

        example.run()
        example.render()

        if example.renderer:
            example.renderer.save()
