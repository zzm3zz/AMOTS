import argparse
import torch
import numpy as np
import torch.nn.functional as f

from threading import Thread
from SimpleITK import WriteImage, ReadImage, GetArrayFromImage, GetImageFromArray
import yaml
from medpy import metric
import nibabel as nb
from nnunet.utilities.utilities_stuff import subfiles, resample_patient, keep_largest_connected_area, slice_argmax
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network.AMOTS import AMOTS
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from collections import OrderedDict

net_pars_fine = {'mean': 83.560104, 'std': 136.6215, 'lower_bound1': -967.0,
                 'upper_bound1': 291.0, 'patch_size': np.array([32, 128, 192]),
                 'target_spacing': np.array([4.0, 1.2, 1.2])}
net_pars_fine2 = {'mean': 79.64796, 'std': 139.33594, 'lower_bound1': -965.0,
                  'upper_bound1': 276.0, 'patch_size': np.array([64, 128, 224]),
                  'target_spacing': np.array([2.5, 0.81835938, 0.81835938])}

ROI_SLIDING_THRESHOLD = 240
ROI_THICKNESS_THRESHOLD = 400
FINE_SIZE_THRESHOLD = 400


def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.
    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int
    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) == 1, print('mask label error!')
    z_index = np.where(organ_mask > 0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)

    return z_lower, z_upper


def get_data(**kwargs):
    data_files = kwargs["data_files"]
    label_file = data_files[0]
    data_itk = [ReadImage(f) for f in data_files]
    label_file = label_file.replace('imagesTs', 'labelsTs').replace('_0000.nii.gz', '.nii.gz') 
    label1 = ReadImage(label_file)
    label = GetArrayFromImage(label1)
    data = np.vstack([GetArrayFromImage(d)[None] for d in data_itk])
    direc = data_itk[0].GetDirection()
    if direc[-1] < 0:
        data = data[:, ::-1]
    data = data.astype(np.float32)

    original_spacing = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    original_size = data[0].shape
    crop_box = [[0, data.shape[1] + 1], [0, data.shape[2] + 1], [0, data.shape[3] + 1]]

    itk_origin = data_itk[0].GetOrigin()
    itk_spacing = data_itk[0].GetSpacing()
    itk_direction = data_itk[0].GetDirection()

    step = [int(np.round(5.0 / original_spacing[0])), 4, 4]
    data_r = data[:, ::step[0], ::step[1], ::step[2]].copy()

    data_r = np.clip(data_r, -1024, 1024)
    data_r = (data_r - data_r.mean()) / (data_r.std() + 1e-8)

    return data, data_r, label, itk_direction, itk_spacing, itk_origin, crop_box, original_size, original_spacing, direc


def get_model_roi(**kwargs):
    with torch.no_grad():
        input_folder = kwargs["input_folder"]
        output_folder = input_folder.replace('input', 'output_roi')
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)
        os.system('python /root/FLARE2023/nnunet/inference/predict_simple.py -i {} -o {} -t 9  -tr nnUNetTrainerV2_Attention  -m 2d  -p nnUNetPlansAttention  --disable_tta'.format(input_folder, output_roi))
        
        # One test at a time, so there is only one prediction file in the folder
        roi_path = os.listdir(folder_path)[0]
        roi = ReadImage(roi_path)
        roi_array = GetArrayFromImage(roi)
        net_roi = torch.from_numpy(numpy_array).float()
        return net_roi


def get_model_fine(**kwargs):
    with torch.no_grad():
        model_fine = kwargs["model_fine"]
        num_input_channels = 1
        num_classes = 15
        net_fine = AMOTS(
            in_channels = num_input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r= 2,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=False,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2],
            ).cuda()
        if torch.cuda.is_available():
            net_fine.cuda()
        net_fine.inference_apply_nonlin = softmax_helper

        net_fine.load_state_dict(torch.load(model_fine)['state_dict'], strict=False)

        net_fine.eval()
    return net_fine
    
    
def get_model_fine2(**kwargs):
    with torch.no_grad():
        model_fine = kwargs["model_fine"]
        
        num_input_channels = 1
        num_classes = 15
        net_fine = AMOTS(
            in_channels = num_input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r= 2,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=False,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2],
            ).cuda()
        if torch.cuda.is_available():
            net_fine.cuda()
        net_fine.load_state_dict(torch.load(model_fine)['state_dict'], strict=False)
        net_fine.cuda()
        net_fine.eval()
        
        print(original_spacing)
        data = kwargs["data"]
        data = resample_patient(data, original_spacing, net_pars_fine2['target_spacing'])

        mean_intensity2 = net_pars_fine2['mean']
        std_intensity2 = net_pars_fine2['std']
        lower_bound2 = net_pars_fine2['lower_bound1']
        upper_bound2 = net_pars_fine2['upper_bound1']
        data = np.clip(data, lower_bound2, upper_bound2)
        data = (data - mean_intensity2) / std_intensity2
    
    return net_fine, data

class MyThread(Thread):
    def __init__(self, *args, **kwargs):
        super(MyThread, self).__init__(*args, **kwargs)
        self.result = None

    def run(self) -> None:
        self.result = self._target(**self._kwargs)

    def get_result(self):
        return self.result

if __name__ == "__main__":
    """ We predict roi in fastest mode for saving time. The step-size is 1 in roi extracting"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    
    parser.add_argument('-net_roi', required=True, help="Mi") # /root/FLARE23_ValResults/outputs
    parser.add_argument('-net_organs', required=True, help="Mo") # /root/AMOTS/Ours_tumor_model/model_best.model
    parser.add_argument('-net_tumor', required=True, help="Mt") # /root/AMOTS/Ours_organs_model/model_best.model
    
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = subfiles(input_folder, suffix=".nii.gz")
    case_ids = np.unique([f[:-12] for f in files])

    output_files = [os.path.join(output_folder, cid + ".nii.gz") for cid in case_ids]
    list_of_lists = [[os.path.join(input_folder, f) for f in files if f[:len(j)].startswith(j) and
                      len(f) == (len(j) + 12)] for j in case_ids]

    seg_metrics = OrderedDict()
    seg_metrics['Name'] = list()
    label_tolerance = OrderedDict({'Liver': 5, 'RK': 3, 'Spleen': 3, 'Pancreas': 5,
                                   'Aorta': 2, 'IVC': 2, 'RAG': 2, 'LAG': 2, 'Gallbladder': 2,
                                   'Esophagus': 3, 'Stomach': 5, 'Duodenum': 7, 'LK': 3, 'Tumor': 2})
    for organ in label_tolerance.keys():
        seg_metrics['{}_DSC'.format(organ)] = list()
    for organ in label_tolerance.keys():
        seg_metrics['{}_NSD'.format(organ)] = list()
    
    torch.cuda.empty_cache()
    for idx in range(len(list_of_lists)):
        data_files = list_of_lists[idx]
        task1 = MyThread(target=get_data, kwargs={"data_files": data_files})
        task2 = MyThread(target=get_model_roi, kwargs={"input_folder": input_folder})
        task3 = MyThread(target=get_model_fine, kwargs={"model_fine": model_fine})
        task1.start()
        task2.start()
        task3.start()
        task1.join()
        task2.join()
        with torch.no_grad():
            data, data_r, label, itk_direction, itk_spacing, itk_origin, crop_box, original_size, original_spacing, direc = task1.get_result()
            net_roi = task2.get_result()
            net_roi.cuda()
            net_roi.eval()
            if data_r.shape[1] > ROI_SLIDING_THRESHOLD:
                seg = net_roi.predict_3D(data_r, step_size=1, patch_size=[80, 128, 128])
            else:
                seg = net_roi.predict_3D(data_r, step_size=1, sliding=False, patch_size=None)
        del data_r, net_roi, task2
        torch.cuda.empty_cache()
 
        label = np.expand_dims(label, axis=0)
        crop_box[0][0] = 0
        crop_box[0][1] = data.shape[1]
        crop_box[1][0] = 0
        crop_box[1][1] = data.shape[2]
        crop_box[2][0] = 0
        crop_box[2][1] = data.shape[3]

        data = data[:, crop_box[0][0]:crop_box[0][1], crop_box[1][0]:crop_box[1][1], crop_box[2][0]:crop_box[2][1]]
        size_ac = list(data.shape[1:])

        task4 = MyThread(target=get_model_fine2, kwargs={"model_fine": model_fine2, "data": data})
        task4.start()
        
        data1 = resample_patient(data, original_spacing, net_pars_fine['target_spacing'])

        mean_intensity = net_pars_fine['mean']
        std_intensity = net_pars_fine['std']
        lower_bound = net_pars_fine['lower_bound1']
        upper_bound = net_pars_fine['upper_bound1']
        data1 = np.clip(data1, lower_bound, upper_bound)
        data1 = (data1 - mean_intensity) / std_intensity

        task3.join()
        with torch.no_grad():
            net_fine = task3.get_result()
            softmax = net_fine.predict_3D(data1, step_size=0.5, patch_size=net_pars_fine['patch_size'])[None][0]

            del data1, net_fine, task3
            torch.cuda.empty_cache()

            current_shape = softmax.shape
  
            if size_ac[0] > FINE_SIZE_THRESHOLD:
                softmax = softmax.detach().cpu()
                torch.cuda.empty_cache()
                if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(size_ac))]):
                    step = size_ac[0] // 150 + 1
                    seg_old_spacing = np.zeros(size_ac)
                    z = current_shape[1]
                    stride = int(z / step)
                    step1 = [i * stride for i in range(step)] + [z]
                    z = size_ac[0]
                    stride = int(z / step)
                    step2 = [i * stride for i in range(step)] + [z]
                    for i in range(step):
                        size = list(size_ac)
                        size[0] = step2[i + 1] - step2[i]
                        slicer = softmax[:, step1[i]:step1[i + 1]][None].half()
                        slicer = f.interpolate(slicer.cuda(), mode='trilinear', size=size, align_corners=True)[0]
                        seg_old_spacing[step2[i]:step2[i + 1]] = slice_argmax(slicer)
                        del slicer
                        torch.cuda.empty_cache()
                else:
                    seg_old_spacing = slice_argmax(softmax)
            else:
                if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(size_ac))]):
                    softmax = f.interpolate(softmax[None].half(), mode='trilinear', size=size_ac, align_corners=True)[0]
                    seg_old_spacing = torch.argmax(softmax, 0).cpu().numpy()
                else:
                    seg_old_spacing = torch.argmax(softmax, 0).cpu().numpy()
            del softmax
            torch.cuda.empty_cache()

            task4.join()
            net_fine2, data = task4.get_result()
            softmax = net_fine2.predict_3D(data, step_size=0.8, patch_size=net_pars_fine2['patch_size'])[None][0]
            del data, net_fine2, task4
            torch.cuda.empty_cache()

            current_shape = softmax.shape
            if size_ac[0] > FINE_SIZE_THRESHOLD:
                softmax = softmax.detach().cpu()
                torch.cuda.empty_cache()
                if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(size_ac))]):
                    step = size_ac[0] // 150 + 1
                    seg_old_spacing2 = np.zeros(size_ac)
                    z = current_shape[1]
                    stride = int(z / step)
                    step1 = [i * stride for i in range(step)] + [z]
                    z = size_ac[0]
                    stride = int(z / step)
                    step2 = [i * stride for i in range(step)] + [z]
                    for i in range(step):
                        size = size_ac
                        size[0] = step2[i + 1] - step2[i]
                        slicer = softmax[:, step1[i]:step1[i + 1]][None].half()
                        slicer = f.interpolate(slicer.cuda(), mode='trilinear', size=size, align_corners=True)[0]
                        seg_old_spacing2[step2[i]:step2[i + 1]] = slice_argmax(slicer)
                        del slicer
                        torch.cuda.empty_cache()
                else:
                    seg_old_spacing2 = slice_argmax(softmax)
            else:
                if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(size_ac))]):
                    softmax = f.interpolate(softmax[None].half(), mode='trilinear', size=size_ac, align_corners=True)[0]
                    seg_old_spacing2 = torch.argmax(softmax, 0).cpu().numpy()
                else:
                    seg_old_spacing2 = torch.argmax(softmax, 0).cpu().numpy()
            del softmax

            seg_old_spacing2[seg_old_spacing > 13] = 14

            if size_ac[0] < FINE_SIZE_THRESHOLD:
                seg_old_spacing2 = keep_largest_connected_area(seg_old_spacing2.astype(np.uint8))

            seg_old_size = np.zeros(original_size)
            for c in range(3):
                crop_box[c][1] = np.min((crop_box[c][0] + seg_old_spacing2.shape[c], original_size[c]))
            seg_old_size[crop_box[0][0]:crop_box[0][1], crop_box[1][0]:crop_box[1][1], crop_box[2][0]:crop_box[2][1]] = \
                seg_old_spacing2
            del seg_old_spacing2

            if direc[-1] < 0:
                seg_old_size = seg_old_size[::-1]
            seg_resized_itk = GetImageFromArray(seg_old_size.astype(np.uint8))
            seg_resized_itk.SetSpacing(itk_spacing)
            seg_resized_itk.SetOrigin(itk_origin)
            seg_resized_itk.SetDirection(itk_direction)
            WriteImage(seg_resized_itk, output_files[idx])

        torch.cuda.empty_cache()
      
        print("Starting with volume {}".format(output_files[idx]))
        submission_volume_path = output_files[idx]
        reference_volume_fn = data_files[0].replace('imagesTs', 'labelsTs').replace('_0000.nii.gz', '.nii.gz')
        print(submission_volume_path)
        print(reference_volume_fn)
        if not os.path.exists(submission_volume_path):
            raise ValueError("Submission volume not found - terminating!\n"
                             "Missing volume: {}".format(submission_volume_path))

        print("Found corresponding submission file {} for reference file {}"
              "".format(reference_volume_fn, submission_volume_path))

        # Load reference and submission volumes with Nibabel.
        reference_volume = nb.load(reference_volume_fn)
        submission_volume = nb.load(submission_volume_path)
        # Get the current voxel spacing.
        voxel_spacing = reference_volume.header.get_zooms()[:3]

        # Get Numpy data and compress to int8.
        reference_volume = (reference_volume.get_fdata()).astype(np.int8)
        submission_volume = (submission_volume.get_fdata()).astype(np.int8)
        # Ensure that the shapes of the masks match.
        if submission_volume.shape != reference_volume.shape:
            raise AttributeError("Shapes do not match! Prediction mask {}, "
                                 "ground truth mask {}"
                                 "".format(submission_volume.shape,
                                           reference_volume.shape))
        # print("Done loading files ({:.2f} seconds)".format(t()))

        # ----------------------- flare metrics
        seg_metrics['Name'].append(os.path.basename(reference_volume_fn))
        for i, organ in enumerate(label_tolerance.keys(), 1):
            if np.sum(reference_volume == i) == 0 and np.sum(submission_volume == i) == 0:
                DSC_i = 1
                NSD_i = 1
            elif np.sum(reference_volume == i) == 0 and np.sum(submission_volume == i) > 0:
                DSC_i = 0
                NSD_i = 0
            elif np.sum(reference_volume == i) > 0 and np.sum(submission_volume == i) == 0:
                DSC_i = 0
                NSD_i = 0
            else:
                if i == 5 or i == 6 or i == 10:  # for Aorta, IVC, and Esophagus, only evaluate the labelled slices in ground truth
                    z_lower, z_upper = find_lower_upper_zbound(reference_volume == i)
                    organ_i_gt, organ_i_seg = reference_volume[:, :, z_lower:z_upper] == i, submission_volume[:, :,
                                                                                            z_lower:z_upper] == i
                else:
                    organ_i_gt, organ_i_seg = reference_volume == i, submission_volume == i
                DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
                if DSC_i < 0.1:
                    NSD_i = 0
                else:
                    surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, voxel_spacing)
                    NSD_i = compute_surface_dice_at_tolerance(surface_distances, label_tolerance[organ])
            seg_metrics['{}_DSC'.format(organ)].append(round(DSC_i, 4))
            print(seg_metrics['{}_DSC'.format(organ)])
            seg_metrics['{}_NSD'.format(organ)].append(round(NSD_i, 4))
    
    overall_metrics = {}
    stand_metrics = {}
    for key, value in seg_metrics.items():
        print(key)
        print(value)
        if 'Name' not in key:
            overall_metrics[key] = round(np.mean(value), 4)
            stand_metrics[key] = round(np.std(value), 4)

    organ_dsc = []
    organ_nsd = []
    for key, value in  stand_metrics.items():
        print(key)
        print(value)

    for key, value in overall_metrics.items():
        if 'Tumor' not in key:
            if 'DSC' in key:
                organ_dsc.append(value)
            if 'NSD' in key:
                organ_nsd.append(value)
    overall_metrics['Organ_DSC'] = round(np.mean(organ_dsc), 4)
    overall_metrics['Organ_NSD'] = round(np.mean(organ_nsd), 4)

    print("Computed metrics:")
    for key, value in overall_metrics.items():
        print("{}: {:.4f}".format(key, float(value)))
            
            
