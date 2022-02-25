import numpy as np
import cv2, logging, multiprocessing, time
from codecAD import CodecAD

# cv2 based geometric center connected component as center for crop
def frame_peak_patches_cv2(frame, psz, angle, min_intensity=0, max_r=None):
    import cv2
    fh, fw = frame.shape
    patches, peak_ori = [], []
    mask = (frame > min_intensity).astype(np.uint8)
    comps, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    big_peaks = 0
    single_pixel_peak = 0
    for comp in range(1, comps):
        # ignore single-pixel peak
        area = stats[comp, cv2.CC_STAT_AREA]
        if area == 1: 
            single_pixel_peak += 1
            continue 
            
        # ignore component that is bigger than patch size
        if stats[comp, cv2.CC_STAT_WIDTH] > psz or stats[comp, cv2.CC_STAT_HEIGHT] > psz:
            big_peaks += 1
            continue
        
        # check if the component is within the max radius
        c, r = centroids[comp, 0], centroids[comp, 1]
        if max_r is not None and max_r**2 < ((c - fw/2)**2 + ( r - fh/2)**2):
            continue
                    
        col_s = stats[comp, cv2.CC_STAT_LEFT]
        col_e = col_s + stats[comp, cv2.CC_STAT_WIDTH]
        
        row_s = stats[comp, cv2.CC_STAT_TOP]
        row_e = row_s + stats[comp, cv2.CC_STAT_HEIGHT]

        _patch = frame[row_s:row_e, col_s:col_e]
        
        # mask out other labels in the patch
        _mask  = cc_labels[row_s:row_e, col_s:col_e] == comp
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
        if _min == _max: continue

        _pr_o = row_s - _tp
        _pc_o = col_s - _lp
        peak_ori.append((angle, _pr_o, _pc_o))
        patches.append(_patch)

    return patches, peak_ori, big_peaks


def frame_process_worker_func(frame_tq, psz, patch_tq, mbsz, offset_recover, min_intensity, max_r=None, frame_writer=None):
    logging.info(f"frame process worker {multiprocessing.current_process().name} starting now")
    codecAD = CodecAD()
    patch_list = []
    patch_ori_list = []
    while True:
        try:
            frm_id, data_codec, compressed, uncompressed, codec, rows, cols = frame_tq.get()
        except queue.Empty:
            continue
        except:
            logging.error("Something else of the Queue went wrong")
            continue

        if frm_id < 0: break

        if compressed is None:
            data = data_codec 
        else:
            dec_tick = time.time()
            codecAD.decompress(data_codec, codec, compressed, uncompressed)
            data = codecAD.getData()
            dec_time = 1000 * (time.time() - dec_tick)
            logging.info(f"frame %d has been decoded in %.2f ms using {codec['name']}, compress ratio is %.1f" % (\
                         frm_id, dec_time, codecAD.getCompressRatio()))

        frame = data.reshape((rows, cols))
        if offset_recover != 0:
            frame[frame > 0] += offset_recover

        tick = time.time()
        patches, patch_ori, big_peaks = frame_peak_patches_cv2(frame=frame, angle=frm_id, psz=psz, \
                                                               min_intensity=min_intensity, max_r=max_r)
        patch_list.extend(patches)
        patch_ori_list.extend(patch_ori)

        while len(patch_list) >= mbsz:
            batch_task = (np.array(patch_list[:mbsz])[:,np.newaxis], \
                          np.array(patch_ori_list[:mbsz]).astype(np.float32))
            patch_tq.put(batch_task)
            patch_list = patch_list[mbsz:]
            patch_ori_list = patch_ori_list[mbsz:]
        
        elapse = 1000 * (time.time() - tick)
        logging.info("%d patches cropped from frame %d, %.3fms/frame, %d peaks are too big; "\
                     "%d patches pending infer" % (\
                     len(patch_ori), frm_id, elapse, big_peaks, mbsz*patch_tq.qsize()))
        # back-up raw frames
        if frame_writer is not None:
            frame_writer.append2write({"angle":np.array([frm_id])[None], "frame":frame[None]})
    logging.info(f"worker {multiprocessing.current_process().name} exiting now")