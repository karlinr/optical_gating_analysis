import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.signal import find_peaks

def get_rr_interval(oog, average_over = 1, method = "gradient"):
    """
    Takes an instance of the BasicOpticalGating class and returns the RR interval.
    NOTE: currently assumes that the phases are in radians.

    Args:
        oog (BasicOpticalGating): Instance of the BasicOpticalGating class
    Returns:
        np.array: gradients of the linear fit of the phase data
    """    
    i_prev = 0
    beat_indices = []
    for i in range(1, oog.phases.shape[0]):
        if (oog.phases[i] - oog.phases[i - 1]) < -np.pi:
            if abs(i_prev - i) > 50:
                beat_indices.append(i)
                i_prev = i

    gradients = []
    from scipy.optimize import curve_fit
    for i in range(len(beat_indices) - average_over):
        xs = (np.arange(oog.delta_phases.shape[0]) / oog.sequence_manager.framerate)[beat_indices[i]:beat_indices[i+average_over]] # convert to time
        ys = (oog.unwrapped_phases / (2 * np.pi))[beat_indices[i]:beat_indices[i+average_over]]
        if method == "gradient":
            popt, popc = curve_fit(lambda x, a, b: a * x + b, xs, ys)
            gradient = popt[0]
        elif method == "mean":
            gradient = ((np.max(ys) - np.min(ys))/(np.max(xs) - np.min(xs)))
        else:
            print("Invalid method")
        gradients.append(gradient)

    gradients = np.asarray(gradients)

    return gradients

def brownian_motion_with_restoring_force(N, k, d, T, dt, restoring_force_model = None, verbose = True, random_seed = None):
    # Constants
    k_b = 1.38e-23

    # Initialise
    x_exp = restoring_force_model(0)
    x = x_exp
    v = 0
    a = 0

    # Initialise arrays
    x_data = np.zeros(N)
    v_data = np.zeros(N)
    a_data = np.zeros(N)

    # Random numbers
    if random_seed is not None:
        np.random.seed(random_seed)
    random_numbers = np.random.normal(size = N)

    # Loop
    for i in tqdm(range(N), disable = not verbose):
        if restoring_force_model is not None:
            x_exp = restoring_force_model(i*dt)

        # (restoring force) + (damping force) + (random force)
        a = -(k*(x - x_exp)) - (d * v) + (k_b*T*random_numbers[i])

        # Apply velocity change
        v += a*dt

        # Apply position change
        x += v*dt

        # Add to array
        x_data[i] = x
        v_data[i] = v
        a_data[i] = a

    x_data = np.asarray(x_data)
    v_data = np.asarray(v_data)
    a_data = np.asarray(a_data)

    return x_data, v_data, a_data

def poincare_plot(delta_phases):
    from scipy.stats import gaussian_kde

    x = delta_phases[0:-1]
    y = delta_phases[1::]

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([X.ravel(), Y.ravel()])

    values = np.vstack([x, y])

    kernel = gaussian_kde(values)

    Z = np.reshape(kernel(positions).T, X.shape)

    plt.figure(figsize = (9,9))
    plt.title("Poincaré Density Plot")
    plt.imshow(Z, origin = "lower", cmap = "gist_earth", extent = [xmin, xmax, ymin, ymax], interpolation = "none")
    plt.xlabel(r"$\Delta \phi_{n}$")
    plt.ylabel(r"$\Delta \phi_{n+1}$")
    plt.colorbar(label = "Density")
    plt.tight_layout()
    plt.show()

def get_hr_from_sad(sad_curve, height = 0.5, distance = 10, prominence = 0.05, plot_sad = True):
    from optical_gating_analysis import v_fitting

    peaks, _ = find_peaks(sad_curve, height = height, prominence = prominence, distance = distance)

    if plot_sad:
        plt.plot(sad_curve)
        plt.plot(peaks, sad_curve[peaks], "x")
        plt.xlabel("Frame")
        plt.ylabel("SAD")
        plt.show()

    diff = sad_curve[peaks[0]-1:peaks[0] + 2]
    if diff[0] < diff[1] or diff[2] < diff[1]:
        sad_curve = -sad_curve

    subframes = []
    for peak in peaks:
        diff = sad_curve[peak-1:peak + 2]
        subframes.append(v_fitting(diff[0], diff[1], diff[2])[0])

    minima_locs = np.array(peaks) + np.array(subframes)

    hrs = np.diff(minima_locs)

    return hrs + 1

def get_hr_from_folder(source_folder, height = 0.5, distance = 10, prominence = 0.05, plot_sad = True):
    import glob
    import optical_gating_analysis as OG
    import j_py_sad_correlation as jps


    files = glob.glob(source_folder)

    oog = OG.BasicOpticalGating()
    oog.sequence_manager.set_source(files[0])
    oog.run()

    hrs = []
    for sequence_src in tqdm(files):
        data = OG.SequenceManager.load_tif(sequence_src)
        def get_hr_from_sad(sad_curve, height = 0.5, distance = 10, prominence = 0.05, plot_sad = True):

            frames_of_interest, _ = find_peaks(sad_curve, height = height, prominence = prominence, distance = distance)

            print(frames_of_interest)

            heartrates = []
            for frame in frames_of_interest:
                # Get diffs array
                diffs = -jps.sad_with_references(data[frame], data)
                diffs -= np.min(diffs)
                diffs /= np.max(diffs)
                
                peaks, _ = find_peaks(diffs[frame - 5:frame+200], height = height, prominence = prominence, distance = distance)
                peaks += frame - 5
                # Get next peak
                for peak in peaks:
                    if peak > frame:
                        diff = diffs[peak - 1:peak + 2]
                        heartrate = (peak + OG.v_fitting(-diff[0], -diff[1], -diff[2])[0] + 1) - frame
                        heartrates.append(heartrate)
                        break;

            return peaks[0:-1], heartrates

            

        diffs = -jps.sad_with_references(oog.sequence_manager.reference_sequence[2], data)[0::]
        diffs -= np.min(diffs)
        diffs /= np.max(diffs)

        frame_number, hr = get_hr_from_sad(diffs, height = 0.2, distance = 10, prominence = 0.05, plot_sad = True)
        hrs.extend(hr)

    print(hrs)
    return hrs