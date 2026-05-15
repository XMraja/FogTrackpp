import numpy as np
from collections import deque
from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_reid
from .matching import *

# base class
from .basetracker import BaseTracker

class SDETracker(BaseTracker):
    def __init__(self, args, frame_rate=120): 
        super().__init__(args, frame_rate=frame_rate)
        self.with_reid = args.reid
        self.reid_model = None
        BaseTrack.clear_count()

    def update(self, output_results, img, ori_img):
        self.frame_id += 1
        activated_tracklets = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        categories = output_results[:, -1]

        remain_inds = scores > self.args.conf_thresh
        inds_low = scores > self.args.conf_thresh_low
        inds_high = scores < self.args.conf_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]

        cates = categories[remain_inds]
        cates_second = categories[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        detections = [Tracklet(tlwh, s, cate, motion='sde') for
                      (tlwh, s, cate) in zip(dets, scores_keep, cates)]

        unconfirmed = []
        tracked_tracklets = []  
        for track in self.tracked_tracklets:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracklets.append(track)

        tracklet_pool = BaseTracker.joint_tracklets(tracked_tracklets, self.lost_tracklets)
        for tracklet in tracklet_pool:
            tracklet.predict() 
        for tracklet in unconfirmed:
            tracklet.predict() 

        dists = igda_distance(tracklet_pool, detections)
        if getattr(self.args, 'fuse_detection_score', False):
            dists = fuse_det_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.8) 

        for itracked, idet in matches:
            track = tracklet_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        detections_second = [Tracklet(tlwh, s, cate, motion='sde') for
                             (tlwh, s, cate) in zip(dets_second, scores_second, cates_second)]
        
        r_tracked_tracklets = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        dists_second = igda_distance(r_tracked_tracklets, detections_second)
        matches_second, u_track_second, u_detection_second = linear_assignment(dists_second, thresh=0.5)

        for itracked, idet in matches_second:
            track = r_tracked_tracklets[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        for it in u_track_second:
            track = r_tracked_tracklets[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)

        detections_for_unconfirmed = [detections[i] for i in u_detection]
        dists_unconfirmed = igda_distance(unconfirmed, detections_for_unconfirmed)

        matches_u, u_unconfirmed, u_detection_u = linear_assignment(dists_unconfirmed, thresh=0.9)

        for itracked, idet in matches_u:
            unconfirmed[itracked].update(detections_for_unconfirmed[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)


        for inew in u_detection_u:
            track = detections_for_unconfirmed[inew]
            if track.score < self.init_thresh:
                continue
            track.activate(self.frame_id)
            activated_tracklets.append(track)


        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.merge_tracklets(activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets)
        
        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]
        return output_tracklets