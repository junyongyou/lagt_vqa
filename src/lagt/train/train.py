import os
import tensorflow as tf
import numpy as np
import scipy
import glob
from tensorflow.keras.optimizers import Adam, SGD

from callbacks.callbacks import create_callbacks
from callbacks.warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
from lagt.train.video_clip_feature_generator import VideoClipFeatureGenerator
from lagt.utils.gather_video_ids import gather_all_vids
from callbacks.evaluation_vq_generator import ModelEvaluationGeneratorVQ
from lagt.models.lagt_model import create_model


def check_args(args):
    if 'result_folder' not in args:
        exit('Result folder must be specified')
    if 'meta_file' not in args:
        exit('Meta file of videos and MOS must be specified')
    if 'vids_meta' not in args:
        args['vids_meta'] = None

    if 'model_name' not in args:
        args['model_name'] = 'lagt'
    if 'ugc_chunk_pickle' not in args or 'ugc_chunk_folder' not in args or '' not in args:
        args['ugc_chunk_pickle'] = None
        args['ugc_chunk_folder'] = None
        args['ugc_chunk_folder_flipped'] = None
    if 'database' not in args:
        args['database'] = ['live', 'konvid', 'ugc']

    if 'transformer_params' not in args:
        args['transformer_params'] = [2, 64, 8, 256]
    if 'dropout_rate' not in args:
        args['dropout_rate'] = 0.1

    if 'clip_length' not in args:
        args['clip_length'] = 32

    if 'epochs' not in args:
        args['epochs'] = 400
    if 'lr_base' not in args:
        args['lr_base'] = 1e-3
    if 'batch_size' not in args:
        args['batch_size'] = 32
    if 'lr_schedule' not in args:
        args['lr_schedule'] = True
    if 'multi_gpu' not in args:
        args['multi_gpu'] = 0
    if 'gpu' not in args:
        args['gpu'] = 0

    if 'do_finetune' not in args:
        args['do_finetune'] = True

    return args


def identify_best_weights(result_folder, history, best_plcc):
    best_weights = np.where(history['plcc'] == best_plcc)
    if not all(best_weights):
        return None
    pos = best_weights[0][0]

    pos_loss = '{}_{:.4f}'.format(pos + 1, history['loss'][pos])
    all_weights_files = glob.glob(os.path.join(result_folder, '*.h5'))
    for all_weights_file in all_weights_files:
        weight_file = os.path.basename(all_weights_file)
        if weight_file.startswith(pos_loss):
            best_weights_file = all_weights_file
            return best_weights_file
    return None


def remove_non_best_weights(result_folder, best_weights_files):
    all_weights_files = glob.glob(os.path.join(result_folder, '*.h5'))
    for all_weights_file in all_weights_files:
        if all_weights_file not in best_weights_files:
            os.remove(all_weights_file)


def evaluation_on_testset(model, vq_generator):
    predictions = []
    mos_scores = []

    for i in range(vq_generator.__len__()):
        features, score = vq_generator.__getitem__(i)
        mos_scores.extend(score)
        prediction = model(features)
        predictions.extend(np.squeeze(prediction, 1))

    PLCC = scipy.stats.pearsonr(mos_scores, predictions)[0]
    SROCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
    RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
    MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
    return PLCC, SROCC, RMSE, MAD

def train_main(args, train_vids=None, val_ids=None, test_ids=None):
    """
    Main function to train LAGT-PHIQNet
    :param args: arguments for training
    :return: Max PLCC from the training
    """
    args = check_args(args)
    result_folder = args['result_folder']
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_file = os.path.join(result_folder, 'result.csv')
    if os.path.exists(result_file):
        rf = open(result_file, 'a')
    else:
        rf = open(result_file, 'w+')

    model_name = args['model_name']

    if train_vids == None or val_ids == None:
        # train and val videos will be randomly split based on random seed
        train_vids, val_ids = gather_all_vids(all_vids_pkl=args['vids_meta'])

    clip_length = args['clip_length']
    model_name += '_clip_{}'.format(clip_length)

    epochs = args['epochs']

    # Model parameters
    transformer_params = args['transformer_params']
    dropout_rates = args['dropout_rate']

    feature_length = 1280

    if args['multi_gpu'] == 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[args['gpu']], 'GPU')
        model = create_model(clip_length,
                             feature_length=feature_length,
                             transformer_params=transformer_params,
                             dropout_rate=dropout_rates)
    else:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        with strategy.scope():
            model = create_model(clip_length,
                                 feature_length=feature_length,
                                 transformer_params=transformer_params,
                                 dropout_rate=dropout_rates)
    model.summary()

    optimizer = Adam(args['lr_base'])
    loss = 'mse'
    metrics = 'mae'
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    # model.run_eagerly = True

    train_generator = VideoClipFeatureGenerator(args['meta_file'],
                                                train_vids,
                                                batch_size=args['batch_size'],
                                                clip_length=args['clip_length'],
                                                random_ratio=0.25,
                                                training=True,
                                                ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                                ugc_chunk_folder=args['ugc_chunk_folder'],
                                                ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                                database=args['database'])
    val_generator = VideoClipFeatureGenerator(args['meta_file'],
                                               val_ids,
                                               batch_size=args['batch_size'],
                                               clip_length=args['clip_length'],
                                               random_ratio=0,
                                               training=False,
                                               ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                               ugc_chunk_folder=args['ugc_chunk_folder'],
                                               ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                               database=args['database'])
    test_generator = VideoClipFeatureGenerator(args['meta_file'],
                                              test_ids,
                                              batch_size=1,
                                              clip_length=args['clip_length'],
                                              random_ratio=0,
                                              training=False,
                                              ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                              ugc_chunk_folder=args['ugc_chunk_folder'],
                                              ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                              database=args['database'])

    evaluation_callback = ModelEvaluationGeneratorVQ(val_generator, None)
    callbacks = create_callbacks(model_name,
                                 result_folder,
                                 evaluation_callback,
                                 checkpoint=True,
                                 early_stop=True,
                                 metrics=metrics)

    train_steps = train_generator.__len__()
    if args['lr_schedule']:
        warmup_epochs = 10
        total_train_steps = epochs * train_steps
        warmup_steps = warmup_epochs * train_steps
        warmup_lr = WarmUpCosineDecayScheduler(learning_rate_base=args['lr_base'],
                                               total_steps=total_train_steps,
                                               warmup_learning_rate=0.0,
                                               warmup_steps=warmup_steps,
                                               # hold_base_rate_steps=0,
                                               hold_base_rate_steps=10 * train_steps,
                                               verbose=1)
        callbacks.append(warmup_lr)

    model_history = model.fit(
        x=train_generator,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=val_generator,
        validation_steps=val_generator.__len__(),
        verbose=1,
        shuffle=False,
        callbacks=callbacks,
    )

    max_plcc_pretrain = np.max(model_history.history['plcc'])
    info = 'Pretrain: epochs: {}, MAX PLCC: {}\n'.format(len(model_history.history['plcc']), max_plcc_pretrain)
    rf.write(info)
    print(info)

    best_weights_file = identify_best_weights(result_folder, model_history.history, callbacks[3].best)
    remove_non_best_weights(result_folder, [best_weights_file])

    if not best_weights_file:
        return max_plcc_pretrain

    # do fine-tuning
    if args['do_finetune'] and best_weights_file:
        del (callbacks[-1])
        model.load_weights(best_weights_file)
        finetune_lr = 1e-5
        if args['lr_schedule']:
            warmup_lr_finetune = WarmUpCosineDecayScheduler(learning_rate_base=finetune_lr,
                                                            total_steps=total_train_steps,
                                                            warmup_learning_rate=0.0,
                                                            warmup_steps=warmup_steps,
                                                            hold_base_rate_steps=10 * train_steps,
                                                            verbose=1)
            callbacks.append(warmup_lr_finetune)
        finetune_optimizer = SGD(learning_rate=finetune_lr, momentum=0.9)
        model.compile(loss=loss, optimizer=finetune_optimizer, metrics=[metrics])

        finetune_model_history = model.fit(
            x=train_generator,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_data=val_generator,
            validation_steps=val_generator.__len__(),
            verbose=1,
            shuffle=False,
            callbacks=callbacks,
        )

        max_plcc_finetune = np.max(finetune_model_history.history['plcc'])
        info = 'Finetune: epochs: {}, MAX PLCC: {}\n'.format(len(finetune_model_history.history['plcc']),
                                                             max_plcc_finetune)
        rf.write(info)
        print(info)

        best_weights_file_finetune = identify_best_weights(result_folder, finetune_model_history.history, callbacks[3].best)

    if args['do_finetune']:
        best_weights = best_weights_file_finetune if best_weights_file_finetune is not None else best_weights_file
    else:
        best_weights = best_weights_file
    model.load_weights(best_weights)
    plcc, srocc, rmse, mad = evaluation_on_testset(model, test_generator)
    rf.write('Results on testset: PLCC: {}, SROCC: {}, RMSE: {}, MAD: {}\n'.format(plcc, srocc, rmse, mad))

    rf.flush()
    rf.close()

    if args['do_finetune']:
        return max([max_plcc_pretrain, max_plcc_finetune])
    return max_plcc_pretrain
