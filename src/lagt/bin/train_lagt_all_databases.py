from lagt.train.train import train_main
from pickle import load
from sklearn.model_selection import train_test_split


"""
General train script of LAGT-PHIQNet on all the three databases
"""
if __name__ == '__main__':
    args = {}

    args['vids_meta'] = r'..\meta\all_vids.pkl'
    args['meta_file'] = r'..\meta_data\all_video_mos.csv'

    # if ugc_chunk_pickle is used, then the folders containing PHIQNet features of UGC chunks must be specified
    args['ugc_chunk_pickle'] = None # r'..\\meta_data\ugc_chunks.pkl'
    args['ugc_chunk_folder'] = r'.\frame_features\ugc_chunks'
    args['ugc_chunk_folder_flipped'] = r'.\frame_features_flipped\ugc_chunks'

    # args['database'] = ['live', 'konvid', 'ugc']
    args['database'] = ['konvid']

    args['model_name'] = 'lagt'

    args['transformer_params'] = [2, 64, 8, 256]
    args['result_folder'] = r'C:\vq_datasets\results\lagt_konviq_swin'
    args['dropout_rate'] = 0.1

    args['batch_size'] = 64
    args['clip_length'] = 32

    args['lr_base'] = 1e-3
    args['epochs'] = 300

    args['multi_gpu'] = 0
    args['gpu'] = 1

    args['do_finetune'] = True

    for i in range(10):
        with open(r'C:\vq_datasets\random_splits\split_{}.pkl'.format(i), 'rb') as f:
            train_vids, val_vids = load(f)

        val_vids, test_vids = train_test_split(val_vids, test_size=0.5, random_state=42)
        train_main(args, train_vids=train_vids, val_ids=val_vids, test_ids=test_vids)
