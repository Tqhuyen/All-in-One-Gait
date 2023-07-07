import os
import os.path as osp
import time
import sys
sys.path.append(os.path.abspath('.') + "/demo/libs/")
from track import *
from segment import *
from recognise import *
# current folder = OpenGait
from config import *
import json
from glob import glob
import pickle

human_info = {
    "id": "01",
    "name": "both"
}

data_info = {
    "view": "front"
}




def new_id_prepare(human_info, data_info):
    print(human_info["id"])
    view_dir = osp.join(DATA_REFERENCE, human_info["id"], 'data', data_info['view'])
    feat_dir = osp.join(view_dir, 'GaitFeatures')
    sil_dir = osp.join(view_dir, 'GaitSil')
    track_dir = osp.join(view_dir, 'Track')
    if not osp.exists(view_dir):
        os.makedirs(feat_dir)
        os.makedirs(sil_dir)
        os.makedirs(track_dir)
    else:
        print("dir already created, choose another id or check data")
    json_dir = osp.join(DATA_REFERENCE, human_info["id"], 'info.json')
    with open(json_dir, "w") as fp:
        json.dump(human_info , fp)
        
        
        
def add_data(vid_path, human_info, data_info):
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    # video_path = osp.join(DATA_REFERENCE, human_info["id"], "data", data_info["view"],"data_raw", vid_name)
    
    video_save = osp.join(DATA_REFERENCE, human_info["id"], "data", data_info["view"], "Track")
    track_result = track(video_path=vid_path, video_save_folder=video_save, save_res=True, save_vid=False)
    
    sil_path = osp.join(DATA_REFERENCE, human_info["id"], "data", data_info["view"], "GaitSil")
    print(sil_path)
    data_silhouette = seg(video_path=vid_path, track_result=track_result, sil_save_path=sil_path)
    
    gait_feat_dir = osp.join(DATA_REFERENCE, human_info["id"], "data", data_info["view"], "GaitFeatures")
    print(gait_feat_dir)
    gait_feat = extract_sil(data_silhouette, gait_feat_dir)
    print(gait_feat)
    feat = gait_feat_dir + "/huyen_train/001/undefined/undefined.pkl"
    my_dict = pickle.load(open(feat, 'rb'))
    print(my_dict)
    
    
    
def compare_data(video_path):
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    vid_name = video_path.split("/")[-1]
    # video_path = osp.join(DATA_TEST, human_info["id"], "data", data_info["view"],"data_raw", vid_name)
    video_save = osp.join(DATA_TEST, human_info["id"], "data", data_info["view"], "Track")
    track_result = track(video_path=video_path, video_save_folder=video_save, save_res=False, save_vid=True)
    
    sil_path = osp.join(DATA_TEST, human_info["id"], "data", data_info["view"], "GaitSil")
    print(sil_path)
    data_silhouette = seg(video_path=video_path, track_result=track_result, sil_save_path=sil_path)
    
    gait_feat_dir = osp.join(DATA_TEST, human_info["id"], "data", data_info["view"], "GaitFeatures/")
    print(gait_feat_dir)
    gait_feat = extract_sil(data_silhouette, gait_feat_dir)
    print(gait_feat)
    feat_dir = glob(DATA_REFERENCE + "/*/*/*/GaitFeatures/*/001/undefined/undefined.pkl")
    compare_list = []
    for feat in feat_dir:
        my_dict = pickle.load(open(feat, 'rb'))
        feat_path = feat.split("/")[-8]
        temp_dict = {feat_path: {'undefined': my_dict}}
        compare_list.append(temp_dict)
    compare_dict = {"data_ref": compare_list}
    res = compare(gait_feat, compare_dict)
    print(res)
    for val in list(res.values()):
        current_id = val.split("-")[-1]
        json_dir = osp.join(DATA_REFERENCE, current_id, "info.json")
        f = open(json_dir)
        data = json.load(f)
        print(data)

def main():
    # new_id_prepare(human_info=human_info, data_info=data_info)
    # add_data("/root/All-in-One-Gait/data_pool/huyen_train.mp4", human_info=human_info, data_info=data_info)
    compare_data("/root/All-in-One-Gait/data_pool/two_front_train.mp4")
    # output_dir = "./demo/output/OutputVideos/"
    # os.makedirs(output_dir, exist_ok=True)
    # current_time = time.localtime()
    # timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    # video_save_folder = osp.join(output_dir, timestamp)
    
    # save_root = './demo/output/'
    
    # gallery_video_path = "./demo/output/InputVideos/gallery.mp4"
    # probe1_video_path  = "./demo/output/InputVideos/probe1.mp4"
    # probe2_video_path  = "./demo/output/InputVideos/probe2.mp4"
    # probe3_video_path  = "./demo/output/InputVideos/probe3.mp4"
    # probe4_video_path  = "./demo/output/InputVideos/probe4.mp4"

    # # tracking
    # gallery_track_result = track(gallery_video_path, video_save_folder)
    # probe1_track_result  = track(probe1_video_path, video_save_folder)
    # probe2_track_result  = track(probe2_video_path, video_save_folder)
    # probe3_track_result  = track(probe3_video_path, video_save_folder)
    # probe4_track_result  = track(probe4_video_path, video_save_folder)
    

    # gallery_video_name = gallery_video_path.split("/")[-1]
    # print(gallery_video_name)
    # gallery_video_name = save_root+'/GaitSilhouette/'+gallery_video_name.split(".")[0]
    # # print(gallery_video_name)
    # probe1_video_name  = probe1_video_path.split("/")[-1]
    # probe1_video_name  = save_root+'/GaitSilhouette/'+probe1_video_name.split(".")[0]
    # probe2_video_name  = probe2_video_path.split("/")[-1]
    # probe2_video_name  = save_root+'/GaitSilhouette/'+probe2_video_name.split(".")[0]
    # probe3_video_name  = probe3_video_path.split("/")[-1]
    # probe3_video_name  = save_root+'/GaitSilhouette/'+probe3_video_name.split(".")[0]
    # probe4_video_name  = probe4_video_path.split("/")[-1]
    # probe4_video_name  = save_root+'/GaitSilhouette/'+probe4_video_name.split(".")[0]
    # exist = os.path.exists(gallery_video_name)  and os.path.exists(probe1_video_name) \
    #         and os.path.exists(probe2_video_name) and os.path.exists(probe3_video_name) \
    #         and os.path.exists(probe4_video_name)
    # print(exist)
    # if exist:
    #     gallery_silhouette = getsil(gallery_video_path, save_root+'/GaitSilhouette/')
    #     probe1_silhouette  = getsil(probe1_video_path , save_root+'/GaitSilhouette/')
    #     probe2_silhouette  = getsil(probe2_video_path , save_root+'/GaitSilhouette/')
    #     probe3_silhouette  = getsil(probe3_video_path , save_root+'/GaitSilhouette/')
    #     probe4_silhouette  = getsil(probe4_video_path , save_root+'/GaitSilhouette/')
    # else:
    #     gallery_silhouette = seg(gallery_video_path, gallery_track_result, save_root+'/GaitSilhouette/')
    #     probe1_silhouette  = seg(probe1_video_path , probe1_track_result , save_root+'/GaitSilhouette/')
    #     probe2_silhouette  = seg(probe2_video_path , probe2_track_result , save_root+'/GaitSilhouette/')
    #     probe3_silhouette  = seg(probe3_video_path , probe3_track_result , save_root+'/GaitSilhouette/')
    #     probe4_silhouette  = seg(probe4_video_path , probe4_track_result , save_root+'/GaitSilhouette/')

    # # # recognise
    # gallery_feat = extract_sil(gallery_silhouette, save_root+'/GaitFeatures/')
    # probe1_feat  = extract_sil(probe1_silhouette , save_root+'/GaitFeatures/')
    # probe2_feat  = extract_sil(probe2_silhouette , save_root+'/GaitFeatures/')
    # probe3_feat  = extract_sil(probe3_silhouette , save_root+'/GaitFeatures/')
    # probe4_feat  = extract_sil(probe4_silhouette , save_root+'/GaitFeatures/')

    # gallery_probe1_result = compare(probe1_feat, gallery_feat)
    # gallery_probe2_result = compare(probe2_feat, gallery_feat)
    # gallery_probe3_result = compare(probe3_feat, gallery_feat)
    # gallery_probe4_result = compare(probe4_feat, gallery_feat)

    # # write the result back to the video
    # writeresult(gallery_probe1_result, probe1_video_path, video_save_folder)
    # writeresult(gallery_probe2_result, probe2_video_path, video_save_folder)
    # writeresult(gallery_probe3_result, probe3_video_path, video_save_folder)
    # writeresult(gallery_probe4_result, probe4_video_path, video_save_folder)


if __name__ == "__main__":
    main()
