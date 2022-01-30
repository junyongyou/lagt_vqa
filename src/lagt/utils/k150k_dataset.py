import os


def check_mos_video():
    meta = r'C:\vq_datasets\k150k\k150ka_scores.csv'

    min_score = 100
    max_score = -1

    video_mos_file = open(r'C:\vq_datasets\k150k\video_mos.csv', 'w+')

    with open(meta, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            content = line.strip().split(',')
            video_name = content[0].replace('.mp4', '')
            video_score = float(content[1])

            if video_score > max_score:
                max_score = video_score
            if video_score < min_score:
                min_score = video_score

            if os.path.exists(os.path.join(r'F:\k150k_features', '{}.npy'.format(video_name))) or os.path.exists(os.path.join(r'K:\Faglitteratur\VQA\k150k_features', '{}.npy'.format(video_name))):
                if os.path.exists(os.path.join(r'F:\k150k_features', '{}.npy'.format(video_name))):
                    video_mos_file.write('F:\k150k_features\{}.npy,{}\n'.format(video_name, content[1]))
                else:
                    video_mos_file.write('K:\Faglitteratur\VQA\k150k_features\{}.npy,{}\n'.format(video_name, content[1]))
            print(line)

    video_mos_file.flush()
    video_mos_file.close()
    print('Min score: {}'.format(min_score))
    print('Max score: {}'.format(max_score))


def split_train_val():
    video_mos_file = r'C:\vq_datasets\k150k\video_mos.csv'
    train_file = open(r'C:\vq_datasets\k150k\train.csv', 'w+')
    val_file = open(r'C:\vq_datasets\k150k\val.csv', 'w+')
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    with open(video_mos_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            content = line.split(',')
            score = float(content[1])
            if score < 2:
                s1.append(line)
            elif score < 3:
                s2.append(line)
            elif score < 4:
                s3.append(line)
            else:
                s4.append(line)
            print(line)

    for k, l in enumerate(s1):
        if k % 10 == 0:
            val_file.write('{}\n'.format(l))
        else:
            train_file.write('{}\n'.format(l))
    for k, l in enumerate(s2):
        if k % 10 == 0:
            val_file.write('{}\n'.format(l))
        else:
            train_file.write('{}\n'.format(l))
    for k, l in enumerate(s3):
        if k % 10 == 0:
            val_file.write('{}\n'.format(l))
        else:
            train_file.write('{}\n'.format(l))
    for k, l in enumerate(s4):
        if k % 10 == 0:
            val_file.write('{}\n'.format(l))
        else:
            train_file.write('{}\n'.format(l))

    print('{}, {}, {}, {}'.format(len(s1), len(s2), len(s3), len(s4)))

    train_file.flush()
    train_file.close()

    val_file.flush()
    val_file.close()


if __name__ == '__main__':
    check_mos_video()
    split_train_val()