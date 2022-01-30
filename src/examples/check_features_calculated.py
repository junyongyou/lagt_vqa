import os
from pickle import dump, load


def video_frame_features_PHIQNet_folder(video_folder, target_folder):
    video_types = ('.mp4')
    video_paths = [f for f in os.listdir(video_folder) if f.endswith(video_types)]
    # video_paths = video_paths[120000:130000]
    numb_videos = len(video_paths)
    unfinished = []

    # f = open(r'', 'w+')
    t = 0
    for i, video_path in enumerate(video_paths):
        ext = os.path.splitext(video_path)
        np_file = os.path.join(target_folder, 'frame_features_flipped', '{}.npy'.format(ext[0]))
        if not os.path.exists(np_file):
            # f.write('{}\n'.format(video_path))
            unfinished.append(video_path)
            t += 1
        else:
            print('{} out of {}, {} already exists'.format(i, numb_videos, video_path))
    # f.flush()
    # f.close()
    print(t)
    # with open('unfinished.pickle', 'wb') as handle:
    #     dump(unfinished, handle)


if __name__ == '__main__':
    # video_path = r'.\\sample_data\example_video (mos=3.24).mp4'
    video_folder = r'K:\Faglitteratur\VQA\k150ka'
    # video_folder = r'D:\VQ_datasets\ugc'
    # video_folder = r'K:\Faglitteratur\VQA\k150kb\k150kb'
    # with open('unfinished.pickle', 'rb') as handle:
    #     b = load(handle)
    t = 0

    target_folder = r'D:\vid150k_phiqnet'
    features = video_frame_features_PHIQNet_folder(video_folder, target_folder)

    # Use None that ResNet50 will download ImageNet Pretrained weights or specify the weight path
    # resnet50_imagenet_weights = r'C:\pretrained_weights_files\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # features_resnet50 = video_frame_features_ResNet50(resnet50_imagenet_weights, video_path)

    # target_folder = r'C:\vq_datasets\Resnet50_features\frame_features\ugc'
    # video_frame_features_ResNet50_folder(resnet50_imagenet_weights, video_folder, target_folder)
    t = 0