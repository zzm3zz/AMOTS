#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Tuple
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from nnunet.network.AMOTS import AACNet
from nnunet.network_architecture.generic_modular_residual_UNet import FabiansUNet, get_default_network_config
from nnunet.training.loss_functions.compound_losses_cutmix import DC_CE_Partial_MergeProb_loss, DC_CE_Partial_MergeProb_loss_ours
from nnunet.training.loss_functions.dice_loss import MemoryEfficientSoftDiceLoss
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.deep_loss import DeepSupervisionWrapper
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader3D_partial, DataLoader2D, unpack_dataset
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation_partial
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import torch.backends.cudnn as cudnn
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetTrainerV2_Organs(nnUNetTrainerV2):
    def initialize_network(self):
        self.network = AACNet(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2                 ,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )
        if torch.cuda.is_available():
            self.network.cuda()

    def process_plans(self, plans):
        super().process_plans(plans)
        # Please don't do this for nnunet. This is only for MedNeXt for all the DS to be used
        num_of_outputs_in_mednext = 5
        self.net_num_pool_op_kernel_sizes = [[2,2,2] for i in range(num_of_outputs_in_mednext+1)] 
        
    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]
   
    def _get_deep_supervision_scales(self):
        self.pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            self.pool_op_kernel_sizes), axis=0))[:-1]
        return deep_supervision_scales

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None
        
    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.begin_partial_epoch = 200
            deep_supervision_scales = self._get_deep_supervision_scales()
            self.different_partial_type_keys_json = '/zhangzengmin/AMOTS/Training/nnUNet/dataset/nnUNet_preprocessed/Task006_Tumor_Crop/different_partial_type_filter.json'
            self.case_to_partial_type_json = '/zhangzengmin/AMOTS/Training/nnUNet/dataset/nnUNet_preprocessed/Task006_Tumor_Crop/case_to_partial_type.json'
            self.case_to_partial_type_dict = load_json(self.case_to_partial_type_json)
            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)
            # net_numpool = net_numpool + 1
            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            # self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            loss = DC_CE_Partial_MergeProb_loss_ours(
            {'batch_dice': self.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': False},
            {}, weight_ce=1, weight_dice=1,ignore_label=255,
            dice_class=MemoryEfficientSoftDiceLoss)


            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            #weights = [weights[0]] + [0]*(len(weights)-1)
            print(f"deep supervision weights:{weights}")
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
            
            self.loss = loss
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr_full, self.dl_tr_part_organs, self.dl_val = self.get_partial_dataloaders()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen_full, self.tr_gen_part, self.val_gen = get_moreDA_augmentation_partial(
                    self.dl_tr_full, self.dl_tr_part_organs, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING FULL_ORGANS KEYS:\n %s" % (str(self.dataset_tr_full.keys())),
                                       also_print_to_console=True)
                self.print_to_log_file("TRAINING PART_ORGANS:\n %s" % (str(self.dataset_tr_part_organs.keys())),
                                       also_print_to_console=True)
                self.print_to_log_file("TRAINING VALIDATE:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=True)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def get_tr_and_val_datasets(self):
        # create dataset split
        self.dataset = load_dataset(self.folder_with_preprocessed_data)
        self.do_split() 
        
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, is_val=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        
        self.device = torch.device('cuda')
        if is_val:
            partial_type = [0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14]
        else:
            partial_type = [int(item) for item in data_dict['partial_type'][0].split("_")]
#             if len(partial_type) == 14:
#                 partial_type = [14]
        partial_type = [torch.tensor(partial_type).to(self.device)]*len(target)
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        # print(data_dict['partial_type'])
        self.optimizer.zero_grad()
        from torch.cuda.amp import GradScaler, autocast
        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target, partial_type)
                
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.validate(self, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                                     step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
                                     overwrite=overwrite, validation_folder_name=validation_folder_name,
                                     debug=debug, all_in_gpu=all_in_gpu,
                                     segmentation_export_kwargs=segmentation_export_kwargs,
                                     run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.decoder.deep_supervision = ds
        return ret
    
    
    def get_partial_dataloaders(self):
        self.get_tr_and_val_datasets()
        
        dl_tr_full = DataLoader3D_partial(self.dataset_tr_full, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', different_partial_type_keys=self.different_partial_type_keys, is_partial=False)
        dl_tr_part_organs = DataLoader3D_partial(self.dataset_tr_part_organs, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', different_partial_type_keys=self.different_partial_type_keys, is_partial=True)
        dl_val = DataLoader3D_partial(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r', different_partial_type_keys=self.different_partial_type_keys, is_partial=False)
        return dl_tr_full, dl_tr_part_organs, dl_val
    
    def _get_different_partial_keys(self):
        self.different_partial_type_keys = load_json(self.different_partial_type_keys_json)
        full_organs_label_keys = self.different_partial_type_keys['1_2_3_4_5_6_7_8_9_10_11_12_13']
        # print(full_label_keys)
        return full_organs_label_keys
    
    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = list(self.dataset.keys())
            val_keys = tr_keys
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        
        full_organs_label_keys = self._get_different_partial_keys()
        tr_keys_full = list(set(tr_keys).intersection(set(full_organs_label_keys)))
        tr_keys_part_organs = [item for item in tr_keys if item not in tr_keys_full]
        
        self.dataset_tr_full = OrderedDict()
        for i in tr_keys_full:
            self.dataset_tr_full[i] = self.dataset[i]
        self.dataset_tr_part_organs = OrderedDict()
        for i in tr_keys_part_organs:
            self.dataset_tr_part_organs[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

        
    def run_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen_full.next()
        _ = self.tr_gen_part.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            train_losses_epoch = []

            # train one epoch
            self.network.train()
            if self.epoch >= self.begin_partial_epoch:
                self.num_batches_per_epoch = 120
            l_part = None
            number = 0
            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))
                        number = number + 1
                        l = self.run_iteration(self.tr_gen_full, True)
                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
                       
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen_full, True)
                    train_losses_epoch.append(l)
                    number = number + 1
                    if self.epoch >= self.begin_partial_epoch:
                            l1 = self.run_iteration(self.tr_gen_part, True)
                            train_losses_epoch.append(l1)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, do_backprop=False, run_online_evaluation=True, is_val=True)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            # epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            # self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))
    

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=do_mirroring,
                                                                             mirror_axes=mirror_axes,
                                                                             use_sliding_window=use_sliding_window,
                                                                             step_size=step_size,
                                                                             use_gaussian=use_gaussian,
                                                                             pad_border_mode=pad_border_mode,
                                                                             pad_kwargs=pad_kwargs,
                                                                             all_in_gpu=all_in_gpu,
                                                                             verbose=verbose,
                                                                             mixed_precision=mixed_precision)
        self.network.decoder.deep_supervision = ds
        return ret
