"""
This is a package was written for analysis of the zebrafish heart beat.
This reimplements open-optical-gating with the goal of being more
flexible for testing different approaches to optical-gating and
testing the Kalman filter method of phase estimation and prediction.

Modules include
BasicOpticalGating - Produces phase estimates of the zebrafish heart
PredictorClass - Takes phase estimates and makes forward predictions
OpticalGatingPlotter - Plots the results of the analysis
plistreader - Used to look at the plist files produced by the C optical gating code
"""

# General imports
import matplotlib.pyplot as plt
import numpy as np
import j_py_sad_correlation as jps
import tifffile as tf
import time
import gc
import glob
import plistlib
from scipy.optimize import curve_fit

# Other
from kalman_filter import KalmanFilter

def v_fitting(y_1, y_2, y_3):
    # Fit using a symmetric 'V' function, to find the interpolated minimum for three datapoints y_1, y_2, y_3,
    # which are considered to be at coordinates x=-1, x=0 and x=+1
    if y_1 < y_2 or y_3 < y_2:
        return 0, 0

    if y_1 > y_3:
        x = 0.5 * (y_1 - y_3) / (y_1 - y_2)
        y = y_2 - x * (y_1 - y_2)
    else:
        x = 0.5 * (y_1 - y_3) / (y_3 - y_2)
        y = y_2 + x * (y_3 - y_2)


    return x, y


class BasicOpticalGating():
    def __init__(self):
        self.settings = {
            "drift_correction": True,
            "padding_frames": 2,
            "normalise_sad": False,
            "pi_space": True,
            "buffer_length": 800,
            "reference_framerate_reduction": 1,
            "include_reference_frames": True,
            "subframe_method": "v_fitting"
        }

        self.sequence_manager = SequenceManager()
        self.sequence_manager.settings = self.settings
        self.predictor = Predictor()

        self.drift = (0, 0)
        self.drifts = [self.drift]


    def get_sads(self):
        print("Getting SADs")

        self.sads = []

        self.frame_means = []
        self.ref_means = []

        while True:
            frame = self.sequence_manager.get_next_frame()
            if frame is None:
                break
            sad, frame_mean = self.get_sad(frame, self.sequence_manager.reference_sequence)
            self.sads.append(sad)
            self.frame_means.append(frame_mean)

    def get_sad(self, frame, reference_sequence):
        """ Get the sum of absolute differences for a single frame and our reference sequence.
            NOTE: In future it might be better to pass this function the frame and reference sequence
            instead of the frame number.

            Args:
                frame_number (int): frame number to get the SAD for
                use_jps (bool, optional): Whether to use Jonny's SAD code which is significantly faster but requires correct dtype. Defaults to True.
                reference_sequence (np.array, optional): The reference sequence. Defaults to None.

            Returns:
                np.array: The sum of absolute differences between the frame and the reference sequence
        """
        # NOTE: Drift correction copied from OOG
        if self.settings["drift_correction"]:
            dx, dy = self.drift
            rectF = [0, frame.shape[0], 0, frame.shape[1]]  # X1,X2,Y1,Y2
            rect = [
                0,
                reference_sequence[0].shape[0],
                0,
                reference_sequence[0].shape[1],
            ]  # X1,X2,Y1,Y2

            if dx <= 0:
                rectF[0] = -dx
                rect[1] = rect[1] + dx
            else:
                rectF[1] = rectF[1] - dx
                rect[0] = dx
            if dy <= 0:
                rectF[2] = -dy
                rect[3] = rect[3] + dy
            else:
                rectF[3] = rectF[3] - dy
                rect[2] = +dy

            frame_cropped = frame[rectF[0] : rectF[1], rectF[2] : rectF[3]]
            reference_frames_cropped = [
                f[rect[0] : rect[1], rect[2] : rect[3]] for f in reference_sequence
            ]
            #reference_frames_cropped = reference_sequence[:, rect[0] : rect[1], rect[2] : rect[3]]
        else:
            frame_cropped = frame
            reference_frames_cropped  = reference_sequence

        SAD = jps.sad_with_references(frame_cropped, reference_frames_cropped)

        from scipy import signal
        #sos = signal.butter(50, 40/100, output='sos', btype = "lowpass")
        sos = signal.butter(10, [55,65], fs = 320, output='sos', btype = "bandstop")


        #SAD = signal.sosfiltfilt(sos, SAD)

        if self.settings["normalise_sad"]:
            SAD = (SAD - np.min(SAD)) / (np.max(SAD) - np.min(SAD))

        if self.settings["drift_correction"]:
            dx, dy = self.drift
            self.drift = update_drift_estimate(frame, reference_sequence[np.argmin(SAD)], (dx, dy))
            self.drifts.append(self.drift)

        frame_mean = np.sum(frame_cropped) / np.product(frame_cropped.shape)

        return SAD, frame_mean
    
    def get_phases(self):
        """ Get the phase estimates for our sequence""" 
        print("Getting phases")

        self.phases = []
        self.frame_minimas = []
        
        # Get the frame estimates
        for i, sad in enumerate(self.sads):
            phase = self.get_phase(sad)
            self.phases.append(phase[0])
            self.frame_minimas.append(phase[1])

        self.phases = np.array(self.phases)
        self.frame_minimas = np.array(self.frame_minimas)

        # Get delta phases
        if self.settings["pi_space"]:
            period = 2 * np.pi
        else:
            period = self.sequence_manager.reference_period
        self.delta_phases = np.diff(self.phases)
        self.delta_phases[self.delta_phases < - period / 2] += period
        self.delta_phases[self.delta_phases > period / 2] -= period

        # Get unwrapped phases
        self.unwrapped_phases = np.unwrap(self.phases, period = period)
        self.unwrapped_phases = self.unwrapped_phases - self.unwrapped_phases[0]

    def get_phase(self, sad):
        frame_minima = np.argmin(sad[self.settings["padding_frames"]:-self.settings["padding_frames"]]) + self.settings["padding_frames"]

        y_1 = sad[frame_minima - 1]
        y_2 = sad[frame_minima]
        y_3 = sad[frame_minima + 1]
        
        subframe_minima = v_fitting(y_1, y_2, y_3)[0]

        
        if self.settings["pi_space"] == True:
            phase = 2 * np.pi * ((frame_minima - self.settings["padding_frames"] + subframe_minima) / self.sequence_manager.reference_period)
        else:
            phase = frame_minima - self.settings["padding_frames"] + subframe_minima

        if self.settings["subframe_method"] == "parabola":
            # fit a parabola using curve_fit to find minima
            def u_fit(x, a, b, c):
                return a * x**2 + b * x + c
            
            popt, pcov = curve_fit(u_fit, np.arange(frame_minima - 2, frame_minima + 3), sad[frame_minima - 2:frame_minima + 3], maxfev = 10000)
            phase = - popt[1] / (2 * popt[0]) - self.settings["padding_frames"]


        return phase, frame_minima


    def run(self):
        self.sequence_manager.reset()
        if self.sequence_manager.reference_sequence is None:
            self.sequence_manager.get_reference_sequence()
        drift_correct = 32
        self.drift = get_drift_estimate(self.sequence_manager.get_next_frame(), self.sequence_manager.reference_sequence, dxRange=range(-drift_correct,drift_correct+1,1), dyRange=range(-drift_correct,drift_correct+1,1))
        self.drifts = [self.drift]
        self.get_sads()
        self.get_phases()

    # Run with some default data
    @classmethod
    def default(cls):
        oog = cls()
        oog.sequence_manager.set_source(r"D:\Data\2012-06-20 13.34.11 vid 2x2 multi phase single plane\brightfield\*tif")
        oog.sequence_manager.set_reference_sequence(r"D:\Data\2012-06-20 13.34.11 vid 2x2 multi phase single plane\ref_seq.tif")
        oog.sequence_manager.reference_period = 3.577851226661945105e+01
        return oog

    def plot_summary(self):
        frame_rate = self.sequence_manager.frame_rate

        plt.figure(figsize = (16,16))
        plt.subplot(221)
        plt.title("Phases")
        plt.plot(self.phases)
        plt.xlabel("Frame number")
        plt.ylabel("Phase (rad)")

        plt.subplot(222)
        plt.title("Phases against delta-phases")
        plt.scatter(self.phases[1::],radsperframe_to_bps(self.delta_phases, frame_rate), s = 5)
        plt.xlabel("Phase (rad)")
        plt.ylabel("Beat velocity (beats/s)")

        plt.subplot(223)
        plt.title("Drifts")
        plt.plot(np.array(self.drifts)[:,0], label = "x")
        plt.plot(np.array(self.drifts)[:,1], label = "y")
        plt.legend()
        plt.xlabel("Frame number")
        plt.ylabel("Drift (pixels)")

        plt.subplot(224)
        plt.title("Delta-phases against frame number")
        plt.scatter(range(self.delta_phases.shape[0]), radsperframe_to_bps(self.delta_phases, frame_rate), s = 5)
        plt.xlabel("Frame number")
        plt.ylabel("Beat velocity (beats/s)")

        plt.show()

class SequenceManager():
    def __init__(self):
        self.frame_history = []
        self.period_history = []
        self.reference_sequence = None
        self.reference_indices = None
        self.current_image_array = None
        self.max_frames = None
        self.skip_frames = None
        self.total_frame_index = 0

        self.frame_rate = 80

    def reset(self):
        self.file_index = 0
        self.frame_index = 0

    def set_source(self, sequence_src):
        print(f"Setting source to {sequence_src}")
        self.sequence_src = sequence_src
        if type(sequence_src) is list:
            self.file_list = glob.glob(sequence_src[0])
            for i in range(1, len(sequence_src)):
                self.file_list.extend(glob.glob(sequence_src[i]))
            print(self.file_list)
            self.file_list = sorted(self.file_list)
            print(self.file_list)
        else:
            self.file_list = sorted(glob.glob(sequence_src))

        self.reset_frame_loader()


    def get_next_frame(self):
        if self.max_frames is not None and self.total_frame_index > self.max_frames:
            self.current_image_array = None
            self.current_frame = None
            return None
        
        if (self.current_image_array is None) and (self.file_index < len(self.file_list)):
            self.current_image_array = self.load_tif(self.file_list[self.file_index])
            self.frame_index = 0

        if self.skip_frames is not None:
            while self.total_frame_index < self.skip_frames:
                self.frame_index += 1
                self.total_frame_index += 1

                if self.frame_index == self.current_image_array.shape[0]:
                    self.current_image_array = self.load_tif(self.file_list[self.file_index])
                    self.file_index += 1
                    self.frame_index = 0
                
        if self.file_index >= len(self.file_list):
            self.current_image_array = None
            self.current_frame = None
            return None
        
        if self.current_image_array is None:
            self.current_frame = None
            return None
        else:
            self.current_frame = self.current_image_array[self.frame_index]
            if self.frame_index == self.current_image_array.shape[0] - 1:
                self.frame_index = 0
                self.file_index += 1
                if self.file_index < len(self.file_list):
                    self.current_image_array = self.load_tif(self.file_list[self.file_index])
            else:
                self.frame_index += 1
            self.total_frame_index += 1
            return self.current_frame
        
    def set_reference_sequence(self, data_src):
        print(f"Loading reference sequence from {data_src}")
        self.reference_sequence = self.load_tif(data_src)

    def set_reference_sequence_by_indices(self, indices):
        frame_history = []

        for i in range(0, indices[1]):
            frame = self.get_next_frame()
            if i >= indices[0]:
                frame_history.append(frame)

        self.reference_period = indices[1] - indices[0]
        self.reference_sequence = np.array(frame_history)
        self.reference_indices = indices
        self.reset_frame_loader()

    def get_reference_sequence(self):
        print("Getting reference sequence")

        i = 0
        while True:
            i += 1
            frame = self.get_next_frame()
            refget = self.establish_period_from_frames(frame)
            if refget[0] != None:
                break
        self.reference_sequence = np.array(refget[0])
        self.reference_period = refget[1]
        self.reference_indices = refget[2]

        if self.settings["include_reference_frames"]:
            self.reset_frame_loader()

        print(f"Reference period: {self.reference_period}; Reference indices: {self.reference_indices}")

    def reset_frame_loader(self):
        self.frame_index = 0
        self.file_index = 0
        self.total_frame_index = 0
        self.current_image_array = None
            
    def establish_period_from_frames(self, pixel_array):
        """ Attempt to establish a period from the frame history,
            including the new frame represented by 'pixel_array'.

            Returns: True/False depending on if we have successfully identified a one-heartbeat reference sequence
        """
        # Add the new frame to our history buffer
        self.frame_history.append(pixel_array)

        # Impose an upper limit on the buffer length, to protect against performance degradation
        # in cases where we are not succeeding in identifying a period.
        # That limit is defined in terms of how many seconds of frame data we have,
        # relative to the minimum heart rate (in Hz) that we are configured to expect.
        # Note that this logic should work when running in real time, and with file_optical_gater
        # in force_framerate=False mode. With force_framerate=True (or indeed in real time) we will
        # have problems if we can't keep up with the framerate frames are arriving at.
        # We should probably be monitoring for that situation...
        ref_buffer_duration = len(self.frame_history)
        while (ref_buffer_duration > self.settings["buffer_length"]):
            # I have coded this as a while loop, but we would normally expect to only trim one frame at a time
            del self.frame_history[0]
            ref_buffer_duration = len(self.frame_history)

        return self.establish(self.frame_history, self.period_history)

    def establish(self, sequence, period_history, require_stable_history=True):
        """ Attempt to establish a reference period from a sequence of recently-received frames.
            Parameters:
                sequence        list of PixelArray objects  Sequence of recently-received frame pixel arrays (in chronological order)
                period_history  list of float               Values of period calculated for previous frames (which we will append to)
                ref_settings    dict                        Parameters controlling the sync algorithms
                require_stable_history  bool                Do we require a stable history of similar periods before we consider accepting this one?
            Returns:
                List of frame pixel arrays that form the reference sequence (or None).
                Exact (noninteger) period for the reference sequence
        """
        start, stop, periodToUse = self.establish_indices(sequence, period_history, require_stable_history)
        if (start is not None) and (stop is not None):
            referenceFrames = sequence[start:stop]
        else:
            referenceFrames = None

        return referenceFrames, periodToUse, [start, stop]

    def establish_indices(self, sequence, period_history, require_stable_history=True):
        """ Establish the list indices representing a reference period, from a given input sequence.
            Parameters: see header comment for establish(), above
            Returns:
                List of indices that form the reference sequence (or None).
        """
        if len(sequence) > 1:
            frame = sequence[-1]
            pastFrames = sequence[:-1]

            # Calculate Diffs between this frame and previous frames in the sequence
            diffs = jps.sad_with_references(frame, pastFrames)

            # Calculate Period based on these Diffs
            period = self.calculate_period_length(diffs, 5, 0.5, 0.75)
            if period != -1:
                period_history.append(period)

            # If we have a valid period, extract the frame indices associated with this period, and return them
            # The conditions here are empirical ones to protect against glitches where the heuristic
            # period-determination algorithm finds an anomalously short period.
            # JT TODO: The three conditions on the period history seem to be pretty similar/redundant. I wrote these many years ago,
            #  and have just left them as they "ain't broke". They should really be tidied up though.
            #  One thing I can say is that the reason for the *two* tests for >6 have to do with the fact that
            #  we are establishing the period based on looking back from the *most recent* frame, but then actually
            #  and up taking a period from a few frames earlier, since we also need to incorporate some extra padding frames.
            #  That logic could definitely be improved and tidied up - we should probably just
            #  look for a period starting numExtraRefFrames from the end of the sequence...
            # TODO: JT writes: logically these tests should probably be in calculate_period_length, rather than here
            history_stable = (len(period_history) >= (5 + (2 * self.settings["padding_frames"]))
                                and (len(period_history) - 1 - self.settings["padding_frames"]) > 0
                                and (period_history[-1 - self.settings["padding_frames"]]) > 6)
            if (
                period != -1
                and period > 6
                and ((require_stable_history == False) or (history_stable))
            ):
                # We pick out a recent period from period_history.
                # Note that we don't use the very most recent value, because when we pick our reference frames
                # we will pad them with numExtraRefFrames at either end. We pick the period value that
                # pertains to the range of frames that we will actually use
                # for the central "unpadded" range of our reference frames.
                periodToUse = period_history[-1 - self.settings["padding_frames"]]

                # DevNote: int(x+1) is the same as np.ceil(x).astype(np.int)
                numRefs = int(periodToUse + 1) + (2 * self.settings["padding_frames"])

                # return start, stop, period
                return len(pastFrames) - numRefs, len(pastFrames), periodToUse

        return None, None, None
    
    def calculate_period_length(self, diffs, minPeriod=5, lowerThresholdFactor=0.5, upperThresholdFactor=0.75):
        """ Attempt to determine the period of one heartbeat, from the diffs array provided. The period will be measured backwards from the most recent frame in the array
            Parameters:
                diffs    ndarray    Diffs between latest frame and previously-received frames
            Returns:
                Period, or -1 if no period found
        """

        # Calculate the heart period (with sub-frame interpolation) based on a provided list of comparisons between the current frame and previous frames.
        bestMatchPeriod = None

        # Unlike JTs codes, the following currently only supports determining the period for a *one* beat sequence.
        # It therefore also only supports determining a period which ends with the final frame in the diffs sequence.
        if diffs.size < 2:
            return -1

        # initialise search parameters for last diff
        score = diffs[diffs.size - 1]
        minScore = score
        maxScore = score
        totalScore = score
        meanScore = score
        minSinceMax = score
        deltaForMinSinceMax = 0
        stage = 1
        numScores = 1
        got = False

        for d in range(minPeriod, diffs.size+1):
            score = diffs[diffs.size - d]
            # got, values = gotScoreForDelta(score, d, values)

            totalScore += score
            numScores += 1

            lowerThresholdScore = minScore + (maxScore - minScore) * lowerThresholdFactor
            upperThresholdScore = minScore + (maxScore - minScore) * upperThresholdFactor

            if score < lowerThresholdScore and stage == 1:
                stage = 2

            if score > upperThresholdScore and stage == 2:
                # TODO: speak to JT about the 'final condition'
                stage = 3
                got = True
                break

            if score > maxScore:
                maxScore = score
                minSinceMax = score
                deltaForMinSinceMax = d
                stage = 1
            elif score != 0 and (minScore == 0 or score < minScore):
                minScore = score

            if score < minSinceMax:
                minSinceMax = score
                deltaForMinSinceMax = d

            # Note this is only updated AFTER we have done the other processing (i.e. the mean score used does NOT include the current delta)
            meanScore = totalScore / numScores

        if got:
            bestMatchPeriod = deltaForMinSinceMax

        if bestMatchPeriod is None:
            return -1

        bestMatchEntry = diffs.size - bestMatchPeriod

        interpolatedMatchEntry = (bestMatchEntry + v_fitting(diffs[bestMatchEntry - 1], diffs[bestMatchEntry], diffs[bestMatchEntry + 1])[0])

        return diffs.size - interpolatedMatchEntry
            
    @staticmethod
    def load_tif(data_src, frames = None):
        """
        Load data file
        Adapted from open-optical-gating
        TODO: Add support to only use specified frames
        """
        import os
        # Load
        data = None
        # We accumulate the individual files as a list of arrays, and then concatenate them all together
        # This copes with the wildcard case where there is more than one image being loaded,
        # and this chosen strategy performs much better than np.append when we have lots of individual images.
        if isinstance(data_src, str):
            imageList = []
            for fn in sorted(glob.glob(data_src)):
                imageData = tf.imread(fn)
                if len(imageData.shape) == 2:
                    # Cope with loading a single image - convert it to a 1xMxN array
                    # We have a performance risk here: np.append is inefficient so we can't just append each image individually
                    # Instead we accumulate a list and then do a single np.array() call at the end.
                    imageData = imageData[np.newaxis,:,:]
                if (((imageData.shape[-1] == 3) or (imageData.shape[-1] == 4))
                    and (imageData.strides[-1] != 1)):
                    # skimage.io.imread() seems to automatically reinterpret a 3xMxN array as a colour array,
                    # and reorder it as MxNx3. We don't want that! I can't find a way to tell imread not to
                    # do that (as_grayscale does *not* do what I want...). For now I just detect it empirically
                    # and undo it.
                    # The test of 'strides' is an empirical one - clearly imread tweaks that to
                    # reinterpret the original data in a different way to what was intended, but that
                    # makes it easy to spot
                    imageData = np.moveaxis(imageData, -1, 0)
                imageList.append(imageData)
            if len(imageList) > 0:
                data = np.concatenate(imageList)
        elif isinstance(data_src, np.ndarray):
            data = data_src

        return data

def update_drift_estimate(frame0, bestMatch0, drift0):
    """ Determine an updated estimate of the sample drift.
        We try changing the drift value by ±1 in x and y.
        This just calls through to the more general function get_drift_estimate()

        Parameters:
            frame0         array-like      2D frame pixel data for our most recently-received frame
            bestMatch0     array-like      2D frame pixel data for the best match within our reference sequence
            drift0         (int,int)       Previously-estimated drift parameters
        Returns
            new_drift      (int,int)       New drift parameters
        """
    return get_drift_estimate(frame0, [bestMatch0], dxRange=range(drift0[0]-1, drift0[0]+2), dyRange=range(drift0[1]-1, drift0[1]+2))

def get_drift_estimate(frame, refs, matching_frame=None, dxRange=range(-30,31,3), dyRange=range(-30,31,3)):
    """ Determine an initial estimate of the sample drift.
        We do this by trying a range of variations on the relative shift between frame0 and the best-matching frame in the reference sequence.

        Parameters:
            frame          array-like      2D frame pixel data for the frame we should use
            refs           list of arrays  List of 2D reference frame pixel data that we should search within
            matching_frame int             Entry within reference frames that is the best match to 'frame',
                                        or None if we don't know what the best match is yet
            dxRange        list of int     Candidate x shifts to consider
            dyRange        list of int     Candidate y shifts to consider

        Returns:
            new_drift      (int,int)       New drift parameters
        """
    # frame0 and the images in 'refs' must be numpy arrays of the same size
    assert frame.shape == refs[0].shape

    # Identify region within bestMatch that we will use for comparison.
    # The logic here basically follows that in phase_matching, but allows for extra slop space
    # since we will be evaluating various different candidate drifts
    inset = np.maximum(np.max(np.abs(dxRange)), np.max(np.abs(dyRange)))
    rect = [
            inset,
            frame.shape[0] - inset,
            inset,
            frame.shape[1] - inset,
            ]  # X1,X2,Y1,Y2

    candidateShifts = []
    for _dx in dxRange:
        for _dy in dyRange:
            candidateShifts += [(_dx,_dy)]

    if matching_frame is None:
        ref_index_to_consider = range(0, len(refs))
    else:
        ref_index_to_consider = [matching_frame]

    # Build up a list of frames, each representing a window into frame with slightly different drift offsets
    frames = []
    for shft in candidateShifts:
        dxp = shft[0]
        dyp = shft[1]

        # Adjust for drift and shift
        rectF = np.copy(rect)
        rectF[0] -= dxp
        rectF[1] -= dxp
        rectF[2] -= dyp
        rectF[3] -= dyp
        frames.append(frame[rectF[0] : rectF[1], rectF[2] : rectF[3]])

    frames = np.array(frames)

    # Compare all these candidate shifted images against each of the candidate reference frame(s) in turn
    # Our aim is to find the best-matching shift from within the search space
    best = 1e200
    for r in ref_index_to_consider:
        sad = jps.sad_with_references(refs[r][rect[0] : rect[1], rect[2] : rect[3]], frames)
        smallest = np.min(sad)
        if (smallest < best):
            bestShiftPos = np.argmin(sad)
            best = smallest

    return (candidateShifts[bestShiftPos][0],
            candidateShifts[bestShiftPos][1])

class Predictor():
    """
    Takes a list of phases and returns a list of predicted phases
    """
    def __init__(self, prediction_method = None, unwrapped_phases = None):
        self.unwrapped_phases = np.asarray(unwrapped_phases)

        self.set_prediction_method(prediction_method)

    def set_data(self, unwrapped_phases = None):
        self.unwrapped_phases = np.asarray(unwrapped_phases)
        return None
    
    def set_prediction_method(self, prediction_method):
        self.prediction_method = prediction_method
        if self.prediction_method == "kalman":
            self.initialise_kalman_filter()
        elif self.prediction_method == "linear":
            self.initialise_linear_predictor()
        return None
    
    def initialise_linear_predictor(self):
        self.prediction_points = 30
        print("initialising linear predictor")
        return None
    
    def initialise_kalman_filter(self):
        self.kalman_filter = KalmanFilter.constant_velocity_2(1, 1, 0.1, np.array([0, 1]), np.array([[10, 0], [0, 10]]))
        self.kalman_filter.data = self.unwrapped_phases
    
    def run_prediction(self):
        self.xs = []

        if self.prediction_method == "kalman":
            self.kalman_filter.run()
            return self.kalman_filter.xs
        elif self.prediction_method == "linear":
            # Get a moving forward prediction using a linear predictor
            xs = []
            for i in range(2,self.unwrapped_phases.shape[0]):
                if i < self.prediction_points:
                    x = np.arange(0, i)
                    y = self.unwrapped_phases[0:i]
                    fit =  np.polyfit(x, y, 1)
                    self.xs.append(fit[0] * (i + 1) + fit[1])
                else:
                    x = np.arange(i - self.prediction_points, i)
                    y = self.unwrapped_phases[i - self.prediction_points:i]
                    fit =  np.polyfit(x, y, 1)
                    self.xs.append(fit[0] * (i + 1) + fit[1])

class linear_predictor():
    def __init__(self, xcoords, ycoords):
        self.set_data(xcoords, ycoords)

    def set_data(self, xcoords, ycoords):
        self.xcoords = xcoords
        self.ycoords = ycoords

    def get_prediction(self, index, forward_prediction = 3):
        x = np.arange(index[0], index[1])
        fit = np.polyfit(x, self.ycoords[index[0]:index[1]], 1)
        fitfunc = np.poly1d(fit)
        return fitfunc(index[1] + forward_prediction)
    
    def get_predictions(self, prediction_points = 30, forward_prediction = 3):
        predictions = []
        for index in range(prediction_points, self.ycoords.shape[0] - forward_prediction - prediction_points):
            predictions.append(self.get_prediction([index, index + prediction_points], forward_prediction))
        return predictions
        
if __name__ == "__main__":
    """
    Code to test optical gating analysis
    """

    print("Running optical_gating_analysis.py as main")

    BOG = BasicOpticalGating()
    BOG.sequence_manager.set_source(r"D:\Data\2012-06-20 13.34.11 vid 2x2 multi phase single plane\brightfield\*tif")
    BOG.sequence_manager.set_reference_sequence(r"D:\Data\2012-06-20 13.34.11 vid 2x2 multi phase single plane\ref_seq.tif")
    BOG.sequence_manager.reference_period = 3.577851226661945105e+01
    BOG.run()

    predictor = Predictor(unwrapped_phases = BOG.unwrapped_phases)
    predictor.set_prediction_method("linear")
    predictor.run_prediction()

    plt.scatter(BOG.phases[3:-1], predictor.xs[0:-2] - BOG.unwrapped_phases[3:-1])
    plt.show()

    plt.scatter(range(len(BOG.phases[3:-1])), np.asarray(predictor.xs[0:-2]) - (2 * np.pi / 35.25) * np.arange(3, BOG.phases.shape[0] - 1))
    plt.show()

    """plt.plot(BOG.unwrapped_phases)
    plt.show()"""

class analyser():
    def __init__(self, oog):
        self.oog = oog

        self.flags = {
            "sequence_loaded": False
        }

    def get_indices_of_beats(self, start_frame, height, distance, prominence):
        # Get the SADs
        pass

    def get_sequence(self):
        self.sequence = SequenceManager.load_tif(self.oog.sequence_manager.sequence_src)
        self.flags["sequence_loaded"] = True

    def check_if_sequence_loaded(self):
        if self.flags["sequence_loaded"] == False:
            self.get_sequence()

    def get_heartrate_from_delta_phases(self):
        self.check_if_sequence_loaded()
        self.heartrate = 1 / np.mean(self.oog.delta_phases)
        return self.heartrate
    


def radsperframe_to_bps(radsperframe, framerate):
    return (radsperframe * framerate) / (2 * np.pi)