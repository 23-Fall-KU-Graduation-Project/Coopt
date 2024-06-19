from manim import *
import numpy as np

RESOLUTION = 10

for i in range(0, RESOLUTION):
    print(i / RESOLUTION, end=" ")

for i in range(0, RESOLUTION):
    print(round(i / RESOLUTION * (RESOLUTION - 1)), end=" ")

class SurfaceWithCircle(ThreeDScene):
    def construct(self):
        # 3D axes
        axes = ThreeDAxes()

        # Define the function for the surface
        def func(x, y):
            return np.sin(x) * np.cos(y)

        # Generate points for the surface
        x_values = np.linspace(-3, 3, RESOLUTION)
        y_values = np.linspace(-3, 3, RESOLUTION)
        
        # Generate surface points using meshgrid
        X, Y = np.meshgrid(x_values, y_values)
        Z = func(X, Y)

        # Define the Surface using precomputed Z values
        surface = Surface(
            lambda u, v: axes.c2p(
                x_values[int(u)],
                y_values[int(v)],
                Z[int(u)][int(v)]
            ),
            resolution=(RESOLUTION - 1, RESOLUTION - 1),
            u_range=[0, RESOLUTION - 1],
            v_range=[0, RESOLUTION - 1],
            fill_opacity=0.8,
            checkerboard_colors=[BLUE, BLUE]
        )

        # Define the circle in the xy-plane
        circle_radius = 1
        circle = ParametricFunction(
            lambda t: axes.c2p(
                circle_radius * np.cos(t) + 1,
                circle_radius * np.sin(t) + 1,
                func(circle_radius * np.cos(t) + 1, circle_radius * np.sin(t) + 1)
            ),
            t_range=[0, 2 * PI],
            color=RED
        )

        # Adding the axes, surface, and circle to the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, surface, circle)

        # Rotate the camera around the scene
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(30)
        self.stop_ambient_camera_rotation()

# To render the scene, use the following command in the terminal:
# manim -pql surface_with_circle.py SurfaceWithCircle
