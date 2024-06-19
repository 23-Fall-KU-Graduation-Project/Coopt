from manim import *
import torch
import torch.nn as nn

# Define the DNN model
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# Load the trained model
model = DNN()
model.load_state_dict(torch.load('dnn_model.pth'))
model.eval()

def func(x, y):
    input_tensor = torch.tensor([[x, y]], dtype=torch.float32)
    with torch.no_grad():
        z = model(input_tensor).item() * 0.5
    return z

class IntroduceObjectiveFunction(ThreeDScene):
    def construct(self):

        ################# Objective Function #################

        objective_function = MathTex(r"\max_{\delta , v} \mathcal{L} (x + \delta, w + v)")
        objective_function_text = Text("Objective Function").scale(0.8)


        # Positioning Objective Function
        objective_function_text.next_to(objective_function, UP)

        # Positioning Objective Function & Objective Function text
        objective_function.shift(UP)
        objective_function_text.shift(UP)

        # Coloring delta and v
        objective_function[0][3].set_color(RED)  # delta
        objective_function[0][5].set_color(BLUE)  # v
        objective_function[0][10].set_color(RED)  # delta
        objective_function[0][14].set_color(BLUE)  # v

        # Write "SAMAT(Ours)"
        self.play(
            Write(objective_function_text)
        )
        
        self.play(
            Write(objective_function)
        )

        # Add brace and text explanation for the second step
        brace_first = Brace(objective_function, DOWN)
        brace_text_first = brace_first.get_text("Finding maximum for input and weight").scale(0.8)

        self.play(
            GrowFromCenter(brace_first),
            FadeIn(brace_text_first)
        )

        self.wait(2)

        self.play(
            FadeOut(brace_first),
            FadeOut(brace_text_first),
        )

        self.wait(1)

        ################# Objective Function #################

        ################# SAMAT #################

        # Creating copies of the original text for splitting
        samat = objective_function.copy()
        samat_text = Text("SAMAT(Ours)").scale(0.8)

        # Splitting text
        samat_objective_function = MathTex(r"\max_{\delta} \max_{v} \mathcal{L} (x + \delta, w + v)")
        
        # Positioning the target texts
        samat_objective_function.shift(DOWN * 1.5 + LEFT * 3.5)
        samat_text.next_to(samat_objective_function, UP)

        # Coloring texts
        samat_objective_function[0][3].set_color(RED)  # delta
        samat_objective_function[0][12].set_color(RED)  # delta
        samat_objective_function[0][7].set_color(BLUE)  # v
        samat_objective_function[0][16].set_color(BLUE)  # v

        # Write "SAMAT(Ours)"
        self.play(
            Write(samat_text)
        )

        # Animate the transformation into dummy texts
        self.play(
            Transform(samat, samat_objective_function),
        )

        self.wait(1)

        # Add brace and text explanation for the second step
        brace_first = Brace(samat_objective_function[0][4:], DOWN)
        brace_text_first = brace_first.get_text("Finding maximum for weight").scale(0.8)

        self.play(
            GrowFromCenter(brace_first),
            FadeIn(brace_text_first)
        )

        self.wait(2)

        self.play(
        FadeOut(brace_first),
        FadeOut(brace_text_first),
        )

        # Add brace and text explanation for the third step
        brace_second = Brace(samat_objective_function, DOWN)
        brace_text_second = brace_second.get_text("Finding maximum for input").scale(0.8)

        self.play(
            GrowFromCenter(brace_second),
            FadeIn(brace_text_second)
        )

        self.wait(2)

        self.play(
            FadeOut(brace_second),
            FadeOut(brace_text_second),
        )

        self.wait(1)

        ################# SAMAT #################

        ################# AWP #################

        # Creating copies of the original text for splitting
        awp = objective_function.copy()
        awp_text = Text("AWP").scale(0.8)

        # Splitting text
        awp_objective_function = MathTex(r"\max_{v} \max_{\delta} \mathcal{L} (x + \delta, w + v)")
        
        # Positioning the target texts
        awp_objective_function.shift(DOWN * 1.5 + RIGHT * 3.5)
        awp_text.next_to(awp_objective_function, UP)

        # Coloring texts
        awp_objective_function[0][7].set_color(RED)  # delta
        awp_objective_function[0][12].set_color(RED)  # delta
        awp_objective_function[0][3].set_color(BLUE)  # v
        awp_objective_function[0][16].set_color(BLUE)  # v

        # Write "AWP"
        self.play(
            Write(awp_text)
        )

        # Animate the transformation into dummy texts
        self.play(
            Transform(awp, awp_objective_function),
        )

        self.wait(1)

        # Add brace and text explanation for the second step
        brace_first = Brace(awp_objective_function[0][4:], DOWN)
        brace_text_first = brace_first.get_text("Finding maximum for input").scale(0.8)

        self.play(
            GrowFromCenter(brace_first),
            FadeIn(brace_text_first)
        )

        self.wait(2)

        self.play(
        FadeOut(brace_first),
        FadeOut(brace_text_first),
        )

        # Add brace and text explanation for the third step
        brace_second = Brace(awp_objective_function, DOWN)
        brace_text_second = brace_second.get_text("Finding maximum for weight").scale(0.8)

        self.play(
            GrowFromCenter(brace_second),
            FadeIn(brace_text_second)
        )

        self.wait(2)

        self.play(
            FadeOut(brace_second),
            FadeOut(brace_text_second),
        )

        self.wait(1)

class VisualizeObjectiveFunction(ThreeDScene):
    def construct(self):
        # 3D axes
        axes = ThreeDAxes()
        
        # Surface plot
        surface = Surface(
            lambda u, v: axes.c2p(u, v, func(u, v)),
            resolution=(30, 30),  # Increase resolution for smoother surface
            v_range=[-10, 10],
            u_range=[-10, 10],
            fill_opacity=0.8,
        )
        
        # Adding the axes and surface to the scene
        self.set_camera_orientation(phi=60 * DEGREES, theta=75 * DEGREES)
        self.add(axes, surface)
        
        # Draw a circle in the XY plane
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
        self.add(circle)

        # Create a point on the circle
        point = Dot(color=RED).move_to(axes.c2p(circle_radius, 0, func(circle_radius, 0)))
        self.add(point)
        
        # Animation to update the point along the circle with the highest z value
        def update_point(point):
            t = ValueTracker(0)
            max_z = -np.inf
            best_coords = None
            for angle in np.linspace(0, 2 * PI, 100):
                x = circle_radius * np.cos(angle)
                y = circle_radius * np.sin(angle)
                z = func(x, y)
                if z > max_z:
                    max_z = z
                    best_coords = (x, y, z)
            
            def update(mob):
                coords = best_coords
                mob.move_to(axes.c2p(coords[0], coords[1], coords[2]))
            
            return UpdateFromFunc(point, update)
        
        self.play(update_point(point))

        # Keep the scene for a few seconds
        self.wait(2)

# To render the scene, use the following command in the terminal:
# manim -pql script_name.py ThreeDGraph
