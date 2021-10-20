from skimage import measure, filters
import numpy as np

def is_edge_zero(a):
    assert a.ndim == 2, 'must be 2 dimension, got %s' % a.ndim
    return (np.count_nonzero(a[0, :]) == 0) and \
           (np.count_nonzero(a[-1,:]) == 0) and \
           (np.count_nonzero(a[:, 0]) == 0) and \
           (np.count_nonzero(a[:,-1]) == 0)

# local max of the connected component as center for crop
def frame_peak_patches_maxcenter(frame, psz):
    h, w = frame.shape
    patches, peak_ori = [], []
    labels = measure.label(frame > 0) # 17ms
    
    for lbl in range(1, labels.max()+1):
        mask = labels == lbl
        if mask.sum() <= 1: continue # remove all single pixel peak
        _filtered = frame * mask
        _pr, _pc  = np.unravel_index(_filtered.argmax(), _filtered.shape)

        _patch = _filtered[max(0, _pr - psz // 2) : (_pr + psz // 2 + psz%2), \
                           max(0, _pc - psz // 2) : (_pc + psz // 2 + psz%2)]

        if _patch.size != psz**2: continue # ignore patch at frame edge
        if not is_edge_zero(_patch):
            # print("patch size of %d cannot hold peak %d" % (psz, lbl))
            continue
            
        _pr_o, _pc_o = _pr - psz//2, _pc - psz//2
        peak_ori.append((_pr_o, _pc_o))
        patches.append(_patch)
        
    # min-max norm all patches
    patches = np.array(patches)
    _min = patches.min(axis=(1, 2))[:, np.newaxis, np.newaxis]
    _max = patches.max(axis=(1, 2))[:, np.newaxis, np.newaxis]
    patches = (patches - _min) / (_max - _min)

    return patches, np.array(peak_ori)

# geometric center connected component as center for crop
def frame_peak_patches_gcenter(frame, psz, min_intensity=0):
    h, w = frame.shape
    patches, peak_ori = [], []
    labels = measure.label(frame > min_intensity) # 17ms
    
    big_peaks = 0
    for c in measure.regionprops(labels):
        if c.bbox_area == 1: continue # remove all single pixel peak
        _patch = frame[c.bbox[0]:c.bbox[2], c.bbox[1]:c.bbox[3]]
        if _patch.shape[0] > psz or _patch.shape[1] > psz:
            # print("patch size of %d cannot hold peak %d" % (psz, c.label))
            big_peaks += 1
            continue
        _patch = _patch * c.filled_image
        h, w = _patch.shape
        _lp = (psz - w) // 2
        _rp = (psz - w) - _lp
        _tp = (psz - h) // 2
        _bp = (psz - h) - _tp
        _patch = np.pad(_patch, ((_tp, _bp), (_lp, _rp)), mode='constant', constant_values=0)
            
        _pr_o = c.bbox[0] - _tp
        _pc_o = c.bbox[1] - _lp
        peak_ori.append((_pr_o, _pc_o))
        patches.append(_patch)

    # min-max norm all patches
    patches = np.array(patches)
    if patches.shape[0] == 0:
        return patches, None, big_peaks
    _min = patches.min(axis=(1, 2))[:, np.newaxis, np.newaxis]
    _max = patches.max(axis=(1, 2))[:, np.newaxis, np.newaxis]
    patches = (patches - _min) / (_max - _min)

    return patches, np.array(peak_ori), big_peaks