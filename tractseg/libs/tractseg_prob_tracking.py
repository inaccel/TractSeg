import time 
import inaccel.coral as inaccel

import psutil
import numpy as np
import multiprocessing
from functools import partial

from dipy.tracking.streamline import transform_streamlines
from scipy.ndimage.morphology import binary_dilation
from dipy.tracking.streamline import Streamlines

from tractseg.libs import fiber_utils
from tractseg.libs import img_utils

global _PEAKS
_PEAKS = None

global _BUNDLE_MASK
_BUNDLE_MASK = None

global _START_MASK
_START_MASK = None

global _END_MASK
_END_MASK = None

global _TRACKING_UNCERTAINTIES
_TRACKING_UNCERTAINTIES = None

def seed_generator(mask_coords, nr_seeds):
    """
    Randomly select #nr_seeds voxels from mask.
    """
    nr_voxels = mask_coords.shape[0]
    random_indices = np.random.choice(nr_voxels, nr_seeds, replace=True)
    res = np.take(mask_coords, random_indices, axis=0)
    res_1d = res.reshape(nr_seeds*3)
    return res_1d

def pool_process(seeds_in,spacing,peaks_in,bundle_mask_in,start_mask_in,end_mask_in,random_normal_in):
    NUM_SEEDS = 5000

    with inaccel.allocator:
        seeds = np.array(seeds_in,  dtype=np.float32)
        streamlines = np.ndarray(NUM_SEEDS*250*3, dtype=np.float32)
        num_points_each = np.ndarray(NUM_SEEDS+1, dtype=np.uint32)


    pool = inaccel.request('com.inaccel.tractseq.tracking.pool')

    pool.arg(seeds)
    pool.arg(spacing)
    pool.arg(peaks_in)
    pool.arg(bundle_mask_in)
    pool.arg(start_mask_in)
    pool.arg(end_mask_in)
    pool.arg(random_normal_in)
    pool.arg(streamlines)
    pool.arg(num_points_each)

    #print(pool)

    response = inaccel.submit(pool)

    response.result()
    result = []
    for i in range(NUM_SEEDS):
        if(num_points_each[i] > 0):
            if(num_points_each[i] > 83):
                num_points_each[i] = 83
            offset = i*250
            end_of_points = offset + num_points_each[i]*3
            points = streamlines[offset:end_of_points]
            points = np.array(points).reshape((num_points_each[i], 3))
            if(not( (0.0,0.0,0.0) in points)):
                result.append(points)
    return result



def track(peaks, max_nr_fibers=2000, smooth=None, compress=0.1, bundle_mask=None,
          start_mask=None, end_mask=None, tracking_uncertainties=None, dilation=0,
          next_step_displacement_std=0.15, nr_cpus=-1, affine=None, spacing=None, verbose=True):
    """
    Generate streamlines.

    Great speedup was archived by:
    - only seeding in bundle_mask instead of entire image (seeding took very long)
    - calculating fiber length on the fly instead of using extra function which has to iterate over entire fiber a
    second time
    """

    peaks[:, :, :, 0] *= -1  # have to flip along x axis to work properly
    # Add +1 dilation for start and end mask to be more robust
    start_mask = binary_dilation(start_mask, iterations=dilation + 1).astype(np.uint8)
    end_mask = binary_dilation(end_mask, iterations=dilation + 1).astype(np.uint8)
    if dilation > 0:
        bundle_mask = binary_dilation(bundle_mask, iterations=dilation).astype(np.uint8)

    if tracking_uncertainties is not None:
        tracking_uncertainties = img_utils.scale_to_range(tracking_uncertainties, range=(0, 1))

    global _PEAKS
    _PEAKS = peaks
    global _BUNDLE_MASK
    _BUNDLE_MASK = bundle_mask
    global _START_MASK
    _START_MASK = start_mask
    global _END_MASK
    _END_MASK = end_mask
    global _TRACKING_UNCERTAINTIES
    _TRACKING_UNCERTAINTIES = tracking_uncertainties

    # Get list of coordinates of each voxel in mask to seed from those
    mask_coords = np.array(np.where(bundle_mask == 1)).transpose()

    max_nr_seeds = 100 * max_nr_fibers  # after how many seeds to abort (to avoid endless runtime)
    # How many seeds to process in each pool.map iteration
    seeds_per_batch = 5000

    streamlines = []
    fiber_ctr = 0
    seed_ctr = 0
    # Processing seeds in batches so we can stop after we reached desired nr of streamlines. Not ideal. Could be
    #   optimised by more multiprocessing fanciness.
    peaks_1d = peaks.reshape(73*87*73*3)
    bundle_mask_1d = bundle_mask.reshape(73*87*73)
    start_mask_1d = start_mask.reshape(73*87*73)
    end_mask_1d = end_mask.reshape(73*87*73)

    #random_normal = np.loadtxt("/home/data/random_normal.txt", delimiter=',')

    with inaccel.allocator:
        peaks_in = np.array(peaks_1d, dtype=np.float32)
        bundle_mask_in = np.array(bundle_mask_1d, dtype=np.uint32)
        start_mask_in = np.array(start_mask_1d, dtype=np.uint32)
        end_mask_in = np.array(end_mask_1d, dtype=np.uint32)
        #random_normal_in = np.array(random_normal, dtype=np.float32)
    start = time.time()
    while fiber_ctr < max_nr_fibers:
        seeds_1d = seed_generator(mask_coords, seeds_per_batch)
        with inaccel.allocator:
            random_normal_in = np.random.normal(0, 0.15,2000)
            random_normal_in = np.array(random_normal_in, dtype=np.float32)

        streamlines_tmp = pool_process(seeds_1d,spacing,peaks_in,bundle_mask_in,start_mask_in,end_mask_in,random_normal_in)
        streamlines += streamlines_tmp
        fiber_ctr = len(streamlines)
        if verbose:
            print("nr_fibs: {}".format(fiber_ctr))
        seed_ctr += seeds_per_batch
        if seed_ctr > max_nr_seeds:
            if verbose:
                print("Early stopping because max nr of seeds reached.")
            break
    #print("time of while loop {} sec".format(round(time.time()-start,5)))
    #print("final nr streamlines: {}".format(len(streamlines)))

    if verbose:
        print("final nr streamlines: {}".format(len(streamlines)))

    streamlines = streamlines[:max_nr_fibers]   # remove surplus of fibers (comes from multiprocessing)
    streamlines = Streamlines(streamlines)  # Generate streamlines object

    # Move from convention "0mm is in voxel corner" to convention "0mm is in voxel center". Most toolkits use the
    # convention "0mm is in voxel center".
    # We have to add 0.5 before applying affine otherwise 0.5 is not half a voxel anymore. Then we would have to add
    # half of the spacing and consider the sign of the affine (not needed here).
    streamlines = fiber_utils.add_to_each_streamline(streamlines, -0.5)

    # move streamlines to coordinate space
    #  This is doing: streamlines(coordinate_space) = affine * streamlines(voxel_space)
    streamlines = list(transform_streamlines(streamlines, affine))

    # If the original image was not in MNI space we have to flip back to the original space
    # before saving the streamlines
    flip_axes = img_utils.get_flip_axis_to_match_MNI_space(affine)
    for axis in flip_axes:
        streamlines = fiber_utils.invert_streamlines(streamlines, bundle_mask, affine, axis=axis)
    
    # Smoothing does not change overall results at all because is just little smoothing. Just removes small unevenness.
    if smooth:
        streamlines = fiber_utils.smooth_streamlines(streamlines, smoothing_factor=smooth)

    if compress:
        streamlines = fiber_utils.compress_streamlines(streamlines, error_threshold=0.1, nr_cpus=nr_cpus)

    return streamlines
