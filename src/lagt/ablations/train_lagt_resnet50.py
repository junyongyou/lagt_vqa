from lagt.train.train import train_main


#
# Train script of LAGT-Resnet50 on all the three databases, it is same as LAGT-PHIQNet training, but using Resnet50 features to replace PHIQNet features.
# Use lagt\ablations\frame_features_video_folders_resnet50.py to calculate Resnet50 features
#
if __name__ == '__main__':
    args = {}
    args['result_folder'] = r'C:\vq_datasets\results\lagt'
    args['vids_meta'] = r'..\\meta_data\all_vids.pkl'

    # The feature file paths must be changed to Resnet50 feature files
    args['meta_file'] = r'..\\meta_data\all_video_mos.csv'

    # if ugc_chunk_pickle is used, then the folders containing PHIQNet features of UGC chunks must be specified
    args['ugc_chunk_pickle'] = None # r'..\\meta_data\ugc_chunks.pkl'
    args['ugc_chunk_folder'] = r'.\frame_features_resnet50\ugc_chunks'
    args['ugc_chunk_folder_flipped'] = r'.\frame_features_flipped_resnet50\ugc_chunks'

    args['database'] = ['live', 'konvid', 'ugc']

    args['model_name'] = 'lagt'

    args['transformer_params'] = [2, 64, 8, 256]
    args['dropout_rate'] = 0.1
    args['clip_length'] = 32

    args['batch_size'] = 32

    args['lr_base'] = 1e-3
    args['epochs'] = 400

    args['multi_gpu'] = 1
    args['gpu'] = 1

    args['validation'] = 'validation'

    args['do_finetune'] = False

    train_main(args)
