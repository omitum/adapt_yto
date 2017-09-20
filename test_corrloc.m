function test_corrloc()
setenv('LC_ALL','C')
dataset = load('model_result.mat');

%unique_videos = unique(dataset.video_id);
%n_videos = size(unique_videos, 1);
n_frames = size(dataset.label);

vid_corrloc_correct = 0;
vid_frames = 0;
%for vid=1:n_videos
    for ix=1: n_frames
        if (dataset.label(ix) == 1)
            selproposal = dataset.selproposal(ix);
            %preparing bb from [x y w h] to [x1 y1 x2 y2]
            bb = squeeze(dataset.proposals(ix, selproposal, :));
            bb(3) = bb(1) + bb(3);
            bb(4) = bb(2) + bb(4);
            
            bbgt = dataset.gt(ix, :);
            
            %compute bb intersection
            bbi = zeros(1,4);
            bbi(1) = max(bb(1), bbgt(1));
            bbi(2) = max(bb(2), bbgt(2));
            bbi(3) = min(bb(3), bbgt(3));
            bbi(4) = min(bb(4), bbgt(4));
            
            iw = bbi(3)-bbi(1)+1;
            ih = bbi(4)-bbi(2)+1;
            if iw>0 && ih>0
                bb_w = (bb(3)-bb(1)+1);
                bb_h = (bb(4)-bb(2)+1);
                bbgt_w = (bbgt(3)-bbgt(1)+1);
                bbgt_h = (bbgt(4)-bbgt(2)+1);
                intersect_area = iw * ih;
                union_area = (bb_w*bb_h) + (bbgt_w*bbgt_h) - intersect_area;
                iou = intersect_area/union_area;
                if (iou > 0.5)
                    vid_corrloc_correct = vid_corrloc_correct + 1;
                end
            end

            vid_frames = vid_frames + 1;
        end
    end
%end

avg_vid_correct = vid_corrloc_correct / vid_frames

if (~exist('local_result.mat'))
    local_result = [];
    save('local_result.mat', 'local_result');
end
load('local_result.mat', 'local_result');
%local_result = avg_vid_correct;
local_result = [local_result avg_vid_correct];
save('local_result.mat', 'local_result');

