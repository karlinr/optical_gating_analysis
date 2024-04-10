import numpy as np
import matplotlib.pyplot as plt

class DynamicDrawer():
    def __init__(self, width, height, frames, noise_type, noise_level) -> None:
        """
        Drawer class used to simulate the zebrafish heart for use with open optical gating.
        Takes a model describing our phase progression and then generates frames at a given timestamp.
        Used with the SyntheticOpticalGater class.

        Args:
            width (_type_): _description_
            height (_type_): _description_
            frames (_type_): _description_
            noise_type (_type_): _description_
            noise_level (_type_): _description_
        """

        self.settings = {
            "width" : width,
            "height" : height,
            "frames" : frames,
            "noise_type" : noise_type,
            "noise_amount" : noise_level,
            "trigger_frames" : False
        }

        self.dimensions = (width, height)
        self.offset = 0 # Used for modelling fish twitching / drift
        self.phase_offset = 0

        # Initialise our canvas
        self.canvas = np.zeros((self.settings["width"], self.settings["height"]), dtype = np.uint8)

        # Motion model
        self.reset_motion_model()
        #self.add_random_acceleration(5)
        """self.add_velocity_spike(5, 1, 10000)
        self.add_velocity_spike(6, 1, -10000)
        self.add_velocity_spike(7, 1, 10000)
        self.add_velocity_spike(8, 1, -10000)
        self.add_velocity_spike(9, 1, 10000)
        self.add_velocity_spike(10, 1, -10000)
        self.add_velocity_spike(11, 1, 10000)"""

        # Drift model
        self.drift_model = {
            0.0: 0,
            1000: 0
        }
        """self.drift_model = {
            0.0: 0,
            2.2: 250,
            2.5: -250,
            2.8: 0,
            3.2: 250,
            3.5: -250,
            3.8: 0
        }"""
        self.initial_drift_velocity = 0

        self.image_noise_rng = np.random.default_rng(0)

        #self.plot_motion_model()

        #self.save_video()

    def save_video(self):
        import tifffile as tf
        frames = []
        for i in range(self.settings["frames"]):
            frames.append(self.draw_frame_at_timestamp(self.get_state_at_timestamp(float(i / 200))[0])[0])

        frames = np.array(frames)
        tf.imwrite("test.tif", frames)

    # Motion model helper methods
    def reset_motion_model(self):
        self.initial_velocity = 2 * np.pi * 3 # 3 beats per second matches the zebrafish heart
        # Initialise our motion model
        self.motion_model_rng = np.random.default_rng(0)
        self.motion_model = {
            0.0: 0,
            100.0: 0
        }

    def add_random_acceleration(self, sigma = 0):
        """
        Adds a random acceleration for each timestamp in our motion model
        """
        # Set our acceleration to a random value for all times.
        for i, k in enumerate(np.linspace(0, 50, 5000)):
            self.motion_model[k] = self.motion_model_rng.normal(0, sigma)

        # Ensure our keys are in ascending order
        self.motion_model = dict(sorted(self.motion_model.items()))

    def add_velocity_spike(self, time, duration, magnitude):
        """
        Adds an acceleration spike
        """
        self.motion_model[time] = magnitude
        #self.motion_model[time + duration / 2] = -magnitude
        #self.motion_model[time + duration] = 0

        # Ensure our keys are in ascending order
        self.motion_model = dict(sorted(self.motion_model.items()))

 
    def clear_canvas(self):
        # Reinitialise our canvas
        self.canvas = np.zeros_like(self.canvas)

    def set_motion_model(self, initial_velocity, motion_model,):
        """
        Define how are timestamps is converted to phase.

        Args:
            initial_velocity (float) : Initial phase velocity of the simulation.
            motion_model (dict) : Dictionary of timestamps and accelerations.
        """

        self.initial_velocity = initial_velocity
        self.motion_model = motion_model


    def set_drift_model(self, initial_drift_velocity, drift_model):
        self.initial_drift_velocity = initial_drift_velocity
        self.drift_model = drift_model

    def get_state_at_timestamp(self, timestamp):
        """
        Uses equations of motion to determine the system state at a given timestamp with the
        drawers motion model. The function set_motion_model must be called before using this function.

        Args:
            timestamp (_type_): _description_

        Returns:
            tuple: Tuple of the timestamp, position, velocity, and acceleration
        """

        # Get our phase progression
        times = [*self.motion_model.keys()]
        accelerations = [*self.motion_model.values()]
        velocity = self.initial_velocity
        position = 0

        end = False
        # FIXME: This fails when we have an undefined motion model
        # We should set our acceleration to the last acceleration (or zero) in this case and still
        # return the correct values for acceleration, position, and velocity.
        for i in range(len(times) - 1):
            if end == False:
                # First we check if we are within the time period of interest
                if times[i] <= timestamp and times[i + 1] >= timestamp:
                    delta_time = timestamp - times[i]
                    end = True
                else:
                    delta_time = times[i + 1] - times[i]
                    end = False

                # Next we calculate the velocity and position.
                acceleration = accelerations[i]
                position += velocity * delta_time + 0.5 * acceleration * delta_time**2
                velocity += delta_time * acceleration

        # Get our drift
        drift_times = [*self.drift_model.keys()]
        drift_velocities = [*self.drift_model.values()]

        drift_velocity = self.initial_drift_velocity
        drift_position = 0

        drift_end = False
        for i in range(len(drift_times) - 1):
            if drift_end == False:
                # First we check if we are within the time period of interest
                if drift_times[i] <= timestamp and drift_times[i + 1] >= timestamp:
                    delta_time = timestamp - drift_times[i]
                    drift_end = True
                else:
                    delta_time = drift_times[i + 1] - drift_times[i]
                    drift_end = False

                # Next we calculate the velocity and position.
                drift_velocity = drift_velocities[i]
                drift_position += delta_time * drift_velocity

        self.offset = drift_position

        return timestamp, position, velocity, acceleration

    def set_drawing_method(self, draw_mode):
        """
        Define the method used to draw new pixels to the canvas.
        Can use np.add, np.subtract, etc. or pass a function.
        Function should take two inputs: existing canvas and an array
        to draw.

        Args:
            draw_mode (function): Function to use for drawing
        """        
        self.draw = draw_mode

    def draw_to_canvas(self, new_canvas):
        # move our current drawing canvas to the new canvas
        return self.draw(self.canvas, new_canvas)
    
    def get_canvas(self, add_noise = False, timestamp = 0):
        """
        Get the current canvas. Adds noise and ensures correct bit-depth.

        Returns:
            np.array: Canvas
        """        
        self.canvas[self.canvas < 0] = 0
        self.canvas[self.canvas > 255] = 255

        # Add noise
        # For reproducibility of our synthetic data we set the random seed
        #np.random.seed(int((0 + timestamp) * 100))
        if add_noise:
            if self.settings["noise_type"] == "poisson":
                self.canvas += self.image_noise_rng.poisson(self.canvas, self.canvas.shape)
            elif self.settings["noise_type"] == "normal":
                self.canvas += self.image_noise_rng.normal(0, self.settings["noise_amount"], self.canvas.shape)
        
        self.canvas[self.canvas < 0] = 0
        self.canvas[self.canvas > 255] = 255
        return self.canvas.astype(np.uint8)

    # Gaussian
    def circular_gaussian(self, _x, _y, _mean_x, _mean_y, _sdx, _sdy, _theta, _super):
        # Takes an array of x and y coordinates and returns an image array containing a 2d rotated Gaussian
        _xd = (_x - _mean_x)
        _yd = (_y - _mean_y)
        _xdr = _xd * np.cos(_theta) - _yd * np.sin(_theta)
        _ydr = _xd * np.sin(_theta) + _yd * np.cos(_theta)
        return np.exp(-((_xdr**2 / (2 * _sdx**2)) + (_ydr**2 / (2 * _sdy**2)))**_super)

    def draw_circular_gaussian(self, _mean_x, _mean_y, _sdx, _sdy, _theta, _super, _br):
        """
        Draw a circular Gaussian at coordinates

        Args:
            _mean_x (float): X position
            _mean_y (float): Y-position
            _sdx (float): X standard deviation
            _sdy (float): y standard deviation
            _theta (float): Angle (0-2pi)
            _super (float): Supergaussian exponent (>=0)
            _br (float): Brightness
        """
        # Draw a 2d gaussian
        xx, yy = np.indices(self.dimensions)#np.meshgrid(range(self.canvas.shape[0]), range(self.canvas.shape[1]))
        new_canvas = self.circular_gaussian(xx + self.offset, yy + self.offset, _mean_x, _mean_y, _sdx, _sdy, _theta, _super)
        new_canvas = _br * (new_canvas / np.max(new_canvas))
        self.canvas = self.draw_to_canvas(new_canvas)

    def draw_frame_at_timestamp(self, timestamp, add_noise = True):
        """
        Draws a frame at a given timestamp.

        Args:
            phase (float): Phase to draw the frame at
        """

        phase = self.get_state_at_timestamp(timestamp)[1] % (2 * np.pi)
        self.clear_canvas()
        self.set_drawing_method(np.add)
        self.draw_circular_gaussian(64 + 16 * np.sin(phase), 64 + 16 * np.cos(phase), 32 + 8 * np.cos(phase), 32 + 8 * np.cos(phase), 0, 1.6, 256)
        self.set_drawing_method(np.subtract)
        self.draw_circular_gaussian(64 + 16 * np.sin(phase), 64 + 16 * np.cos(phase), 26 + 8 * np.cos(phase), 26 + 8 * np.cos(phase), 0, 1.6, 256)
        self.set_drawing_method(np.add)
        self.draw_circular_gaussian(128 + 16 * np.cos(phase), 128 + 16 * np.sin(phase), 32 + 8 * np.sin(phase), 32 + 8 * np.sin(phase), 0, 1.6, 256)
        self.set_drawing_method(np.subtract)
        self.draw_circular_gaussian(128 + 16 * np.cos(phase), 128 + 16 * np.sin(phase), 26 + 8 * np.sin(phase), 26 + 8 * np.sin(phase), 0, 1.6, 256)

        return self.get_canvas(add_noise, timestamp), phase + self.phase_offset
    

    def plot_motion_model(self):
        """
        Plot the motion model
        """
        xs = np.linspace(0, 15, 1000)
        positions = []
        velocities = []
        accelerations = []
        for x in xs:
            positions.append(self.get_state_at_timestamp(x)[1])
            velocities.append(self.get_state_at_timestamp(x)[2])
            accelerations.append(self.get_state_at_timestamp(x)[3])

        plt.figure()
        plt.title("Heart phase progression model")
        plt.plot(xs, positions, label = "Phase ($m$)")
        plt.plot(xs, velocities, label = "Phase velocity ($ms^{-1}$)")
        plt.plot(xs, accelerations, label = "Phase acceleration ($ms^{-2}$)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    drawer = DynamicDrawer(256, 256, 1000, "normal", 0.1)
    drawer.plot_motion_model()

    for i in range(1000):
        frame = drawer.draw_frame_at_timestamp(i / 200)
        print(i)

    drawer.save_video()
    print("Done")