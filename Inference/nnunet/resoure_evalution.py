import glob
import os
import shutil
import time
import torch
from pathlib import Path
join = os.path.join
# from logger import add_file_handler_to_logger


def check_dir(file_path):
    file_path = Path(file_path)
    files = [f for f in file_path.iterdir() if ".nii.gz" in str(f)]
    if len(files) != 0:
        return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_img_path', required=True, help="All test files") # /root/imagesTs
    parser.add_argument('-temp_in', required=True, help="Inputfolder for individual test files") # /root/FLARE23_ValResults/input/
    parser.add_argument('-temp_out', required=True, help="Outputfolder for test files") # /root/FLARE23_ValResults/outputs
    
    parser.add_argument('-net_roi', required=True, help="Mi") # /root/FLARE23_ValResults/outputs
    parser.add_argument('-net_organs', required=True, help="Mo") # /root/AMOTS/Ours_tumor_model/model_best.model
    parser.add_argument('-net_tumor', required=True, help="Mt") # /root/AMOTS/Ours_organs_model/model_best.model
    
    args = parser.parse_args()
    test_img_path = args.test_img_path
    temp_in = args.temp_in
    temp_out = args.temp_out
    net_roi_path = args.net_roi
    net_organs_path = args.net_organs
    net_tumor_path = args.net_tumor
    
    os.makedirs(temp_in, exist_ok=True)
    os.makedirs(temp_out, exist_ok=True)
    test_cases = sorted(os.listdir(test_img_path))

    try:
        print('loading this model')
        print(test_cases)
        for case in test_cases:
            if case[-1] == 'z':
                if not check_dir(temp_in):
                        print("please check inputs folder "+temp_in)
                shutil.copy(join(test_img_path, case), temp_in)
                start_time = time.time()
                os.system('python /root/AMOTS/Inference/nnunet/run_inference.py -i {} -o {} -net_roi {} -net_organs {} -net_tumor {}'.format(temp_in, temp_out, net_roi_path, net_organs_path, net_tumor_path))
                print(case+" finished!")
                os.remove(join(temp_in, case))
        torch.cuda.empty_cache()
    except Exception as e:
        print(e)