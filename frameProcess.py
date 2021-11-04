from skimage import measure
import numpy as np
import cv2

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
    # labels = measure.label(frame > 0) 
    ccs, labels = cv2.connectedComponents((frame > 0).astype(np.uint8))
    
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

    return patches.astype(np.float32), np.array(peak_ori)

# geometric center connected component as center for crop
def frame_peak_patches_gcenter(frame, psz, min_intensity=0):
    h, w = frame.shape
    patches, peak_ori = [], []
    labels = measure.label(frame > min_intensity) 

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

    return patches.astype(np.float32), np.array(peak_ori), big_peaks

# cv2 based geometric center connected component as center for crop
def frame_peak_patches_cv2(frame, psz, min_intensity=0):
    fh, fw = frame.shape
    patches, peak_ori = [], []
    mask = (frame > 0).astype(np.uint8)
    comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    big_peaks = 0
    single_pixel_peak = 0
    for comp in range(1, comps):
        area = stats[comp, cv2.CC_STAT_AREA]
        if area == 1: 
            single_pixel_peak += 1
            continue # ignore all single pixel peak
            
        cw, ch = stats[comp, cv2.CC_STAT_WIDTH], stats[comp, cv2.CC_STAT_HEIGHT]
        if cw > psz or ch > psz:
            big_peaks += 1
            continue
            
        c, r = round(centroids[comp, 0]), round(centroids[comp, 1])

        cs = max(0, c-psz//2)
        ce = min(fw, c+psz//2+1)
        rs = max(0, r-psz//2)
        re = min(fh, r+psz//2+1)
        _patch =  frame[rs:re, cs:ce]
        _mask  = output[rs:re, cs:ce] == comp
        _patch = _patch * _mask
        
        if _patch.size != psz * psz:
            h, w = _patch.shape
            _lp = (psz - w) // 2
            _rp = (psz - w) - _lp
            _tp = (psz - h) // 2
            _bp = (psz - h) - _tp
            _patch = np.pad(_patch, ((_tp, _bp), (_lp, _rp)), mode='constant', constant_values=0)
        else:
            _tp, _lp = 0, 0
        
        _min, _max = _patch.min(), _patch.max()
        if _min == _max:
            _patch /= _max
        else:
            try:
                _patch = (_patch - _min) / (_max - _min)
            except:
                print(_min, _max)
        
        _pr_o = rs - _tp
        _pc_o = cs - _lp
        peak_ori.append((_pr_o, _pc_o))
        patches.append(_patch.astype(np.float32))

    if len(patches) == 0:
        return patches, None, big_peaks
    
    return patches, np.array(peak_ori), big_peaks

