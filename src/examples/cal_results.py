import numpy as np
import os
import glob

# folder = r'C:\vq_datasets\results\lagt_ugc'
folder = r'C:\vq_datasets\results\lsat_koniq_128_64_8_256_new_model'
# folder = r'C:\vq_datasets\results\lsat_koniq_32_new_model'

log_file = glob.glob(os.path.join(folder, '*.log'))[0]
with open(log_file, 'r+') as lf:
    log_infos = lf.readlines()

result_file = os.path.join(folder, 'result.csv')

plcc = []
srocc = []
rmse = []
with open(result_file, 'r+') as f:
    lines = f.readlines()
    evaluation = []
    run = 0
    for line in lines:
        if 'Finetune' in line:
            contents = line.strip().split(',')
            for content in contents:
                c = content.split(':')
                if len(c) == 3:
                    epoch = int(c[-1])
                else:
                    max_plcc = '{:.4f}'.format(float(c[-1]))

            for log_info in log_infos[1:]:
                content = log_info.strip().split(';')
                if int(content[0]) == (epoch - 41) and float(content[3]) == float(max_plcc):
                    criterion = log_info.strip()
                    evaluation.append(criterion)
                    plcc.append(float(content[3]))
                    rmse.append(float(content[4]))
                    srocc.append(float(content[5]))

            run += 1

    print('PLCC: {}, SROCC: {}, RMSE: {}'.format(np.mean(plcc), np.mean(srocc), np.mean(rmse)))
    t = 0


