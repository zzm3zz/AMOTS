import torch
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
import torch.nn.functional as F
from torch import nn
import numpy as np

def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=None):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_CE_Partial_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", 
                 weight_ce=1, weight_dice=1,ignore_label=255,ex=True,
                 dice_class=SoftDiceLoss):
        super(DC_CE_Partial_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ignore_label = ignore_label
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = nn.NLLLoss()
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ex_choice = ex
        self.weight_ce = 0
        self.weight_dice = weight_dice
        #self.apply_nonlin = softmax_helper_dim1
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, partial_type):
        new_net_output, new_target = net_output.clone(), target.clone()
        #new_net_output = self.apply_nonlin(new_net_output)
        new_net_output, new_target = merge_prediction(new_net_output,
                                                new_target,
                                                partial_type)
        # filter other class output 
        dc_loss = self.dc(new_net_output, new_target)
        ce_loss = self.ce(new_net_output, new_target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            # reserved for other stuff (later?)
            raise NotImplementedError("nah son")
        return result


class DC_CE_Partial_MergeProb_loss(nn.Module):
    """
    for partial data, this loss first convert logits to prob and 
    merge prob to background class
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", 
                 weight_ce=1, weight_dice=1,ignore_label=255,ex=True,
                 dice_class=SoftDiceLoss):
        super(DC_CE_Partial_MergeProb_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ignore_label = ignore_label
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = nn.NLLLoss()
        #self.dc = dice_class(apply_nonlin=None, **soft_dice_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ex_choice = ex
        self.weight_ce = 0
        self.weight_dice = weight_dice
        self.apply_nonlin = softmax_helper_dim1
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, partial_type):
#         print(net_output.shape)
#         print(target.shape)
#         print(partial_type)
        new_net_output, new_target = net_output.clone(), target.clone()
        #new_net_output_soft = self.apply_nonlin(new_net_output)
        #print(f"dc old: {dc_old}, dc:{self.dc(new_net_output_soft, new_target)}")
        #print(f"ce old: {ce_old}, ce:{self.ce(torch.log(new_net_output_soft), new_target.squeeze().type(torch.cuda.LongTensor))}")
        # if partial_type[0]==14:
        #     new_net_output, new_target = merge_prediction(new_net_output,
        #                                                    new_target,
        #                                                    partial_type)
        # else:
        if len(partial_type) < 13:
            new_net_output, new_target = merge_prediction_max(new_net_output,
                                                           new_target,
                                                           partial_type)
        # filter other class output 
        dc_loss = self.dc(new_net_output, new_target)
        ce_loss = self.ce(new_net_output, new_target)
        # # ce_loss = self.ce(torch.log(new_net_output_soft), 
        #                   new_target.squeeze().type(torch.cuda.LongTensor))
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            # reserved for other stuff (later?)
            raise NotImplementedError("nah son")
        return result
    
    
    
class DC_CE_Partial_MergeProb_loss_ours(nn.Module):
    """
    for partial data, this loss first convert logits to prob and 
    merge prob to background class
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", 
                 weight_ce=1, weight_dice=1,ignore_label=255,ex=True,
                 dice_class=SoftDiceLoss):
        super(DC_CE_Partial_MergeProb_loss_ours, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ignore_label = ignore_label
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = nn.NLLLoss()
        #self.dc = dice_class(apply_nonlin=None, **soft_dice_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ex_choice = ex
        self.weight_ce = 0
        self.weight_dice = weight_dice
        self.apply_nonlin = softmax_helper_dim1
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, partial_type):
#         print(net_output.shape)
#         print(target.shape)
#         print(partial_type)
        if len(partial_type) >= 13:
            dc_loss = self.dc(net_output, target)
            ce_loss = self.ce(net_output, target)
            result = ce_loss + dc_loss
            return result
        new_net_output, new_target = net_output.clone(), target.clone()
        # print(partial_type)
        new_bg = 0
        new_target_bg = 0
        reg_loss = 0
        if len(partial_type) < 13:
            merge_max = merge_prediction_max_ours(new_net_output,
                                                           new_target,
                                                           partial_type)
            reg_loss = -(merge_max * torch.log(merge_max + 1e-6)).sum(dim=1).mean()
        # filter other class output
        # 前景混合损失
#         print(new_net_output.shape)
#         print(new_target.shape)
#         print(new_bg.shape)
#         print(new_target_bg.shape)
#         print(merge_max.shape)
#         print(1/0)
        dc_loss = self.dc(new_net_output, new_target)
        ce_loss = self.ce(new_net_output, new_target)
        
#         # 背景+未标记混合损失+香浓熵损失
#         dc_loss_bg = self.dc(new_bg, new_target_bg)
#         ce_loss_bg = self.ce(new_bg, new_target_bg)
        
        # probs = F.log_softmax(merge_max)
        
        # pseudo_label_one_hot = F.one_hot(merge_max.squeeze(1), num_classes=(15-len(partial_type))).float()
        
        # # ce_loss = self.ce(torch.log(new_net_output_soft), 
        #                   new_target.squeeze().type(torch.cuda.LongTensor))
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            # reserved for other stuff (later?)
            raise NotImplementedError("nah son")
        # p1 = len(partial_type) / 15
        # result = result*p1 + (dc_loss_bg + ce_loss_bg)*(1-p1)*0.5 + reg_loss*(1-p1)*0.5
        return result + reg_loss

    
class DC_CE_Partial_MergeProb_loss_mots(nn.Module):
    """
    for partial data, this loss first convert logits to prob and 
    merge prob to background class
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", 
                 weight_ce=1, weight_dice=1,ignore_label=255,ex=True,
                 dice_class=SoftDiceLoss):
        super(DC_CE_Partial_MergeProb_loss_mots, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ignore_label = ignore_label
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = nn.NLLLoss()
        #self.dc = dice_class(apply_nonlin=None, **soft_dice_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ex_choice = ex
        self.weight_ce = 0
        self.weight_dice = weight_dice
        self.apply_nonlin = softmax_helper_dim1
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, partial_type, begin_ahm):
#         print(net_output.shape)
#         print(target.shape)
#         print(partial_type)
#         if len(partial_type) >= 13:
#             dc_loss = self.dc(net_output, target)
#             ce_loss = self.ce(net_output, target)
#             result = ce_loss + dc_loss
#             return result
        new_net_output, new_target = net_output.clone(), target.clone()
        new_net_output[:,10,:,:][new_net_output[:,10,:,:]!= 0] = 0
        new_target[new_target == 11] = 0
        # print(partial_type)
        new_bg = 0
        new_target_bg = 0
        reg_loss = 0
        # if len(partial_type) < 13:
        # print(begin_ahm)
        # print(new_net_output.shape)
        if begin_ahm == 1:
            new_net_output, new_target, merge_max = merge_prediction_max_mots(new_net_output,
                                                               new_target,
                                                               partial_type)
        # reg_loss = -(merge_max * torch.log(merge_max + 1e-6)).sum(dim=1).mean()
        # filter other class output
        # 前景混合损失
#         print(new_net_output.shape)
#         print(new_target.shape)
#         print(new_bg.shape)
#         print(new_target_bg.shape)
#         print(merge_max.shape)
#         print(1/0)
#         print(new_net_output.shape)
#         print(new_target.shape)
        # print(1/0)
        # print(new_net_output.shape)
        # print(new_target.shape)
        dc_loss = self.dc(new_net_output, new_target)
        ce_loss = self.ce(new_net_output, new_target)
        
#         # 背景+未标记混合损失+香浓熵损失
#         dc_loss_bg = self.dc(new_bg, new_target_bg)
#         ce_loss_bg = self.ce(new_bg, new_target_bg)
        
        # probs = F.log_softmax(merge_max)
        
        # pseudo_label_one_hot = F.one_hot(merge_max.squeeze(1), num_classes=(15-len(partial_type))).float()
        
        # # ce_loss = self.ce(torch.log(new_net_output_soft), 
        #                   new_target.squeeze().type(torch.cuda.LongTensor))
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            # reserved for other stuff (later?)
            raise NotImplementedError("nah son")
        # p1 = len(partial_type) / 15
        # result = result*p1 + (dc_loss_bg + ce_loss_bg)*(1-p1)*0.5 + reg_loss*(1-p1)*0.5
        return result
    
    
class DC_CE_Partial_MergeProb_loss_vessel(nn.Module):
    """
    for partial data, this loss first convert logits to prob and 
    merge prob to background class
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", 
                 weight_ce=1, weight_dice=1,ignore_label=255,ex=True,
                 dice_class=SoftDiceLoss):
        super(DC_CE_Partial_MergeProb_loss_vessel, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ignore_label = ignore_label
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = nn.NLLLoss()
        #self.dc = dice_class(apply_nonlin=None, **soft_dice_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ex_choice = ex
        self.weight_ce = 0
        self.weight_dice = weight_dice
        self.apply_nonlin = softmax_helper_dim1
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, partial_type):
#         print(net_output.shape)
#         print(target.shape)
#         print(partial_type)
#         if len(partial_type) >= 13:
#             dc_loss = self.dc(net_output, target)
#             ce_loss = self.ce(net_output, target)
#             result = ce_loss + dc_loss
#             return result
        new_net_output, new_target = net_output.clone(), target.clone()
        # print(partial_type)
        new_bg = 0
        new_target_bg = 0
        reg_loss = 0
        # if len(partial_type) < 13:
#         new_net_output, new_target, merge_max = merge_prediction_max_vessel(new_net_output,
#                                                            new_target,
#                                                            partial_type)
        # reg_loss = -(merge_max * torch.log(merge_max + 1e-6)).sum(dim=1).mean()
        # filter other class output
        # 前景混合损失
#         print(new_net_output.shape)
#         print(new_target.shape)
#         print(new_bg.shape)
#         print(new_target_bg.shape)
#         print(merge_max.shape)
#         print(1/0)
#         print(new_net_output.shape)
#         print(new_target.shape)
        # print(1/0)
        dc_loss = self.dc(new_net_output, new_target)
        ce_loss = self.ce(new_net_output, new_target)
        
#         # 背景+未标记混合损失+香浓熵损失
#         dc_loss_bg = self.dc(new_bg, new_target_bg)
#         ce_loss_bg = self.ce(new_bg, new_target_bg)
        
        # probs = F.log_softmax(merge_max)
        
        # pseudo_label_one_hot = F.one_hot(merge_max.squeeze(1), num_classes=(15-len(partial_type))).float()
        
        # # ce_loss = self.ce(torch.log(new_net_output_soft), 
        #                   new_target.squeeze().type(torch.cuda.LongTensor))
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            # reserved for other stuff (later?)
            raise NotImplementedError("nah son")
        # p1 = len(partial_type) / 15
        # result = result*p1 + (dc_loss_bg + ce_loss_bg)*(1-p1)*0.5 + reg_loss*(1-p1)*0.5
        return result
    
    
class DC_CE_Partial_MergeProb_loss_avg(nn.Module):
    """
    for partial data, this loss first convert logits to prob and 
    merge prob to background class
    """
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", 
                 weight_ce=1, weight_dice=1,ignore_label=255,ex=True,
                 dice_class=SoftDiceLoss):
        super(DC_CE_Partial_MergeProb_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ignore_label = ignore_label
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = nn.NLLLoss()
        #self.dc = dice_class(apply_nonlin=None, **soft_dice_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ex_choice = ex
        self.weight_ce = 0
        self.weight_dice = weight_dice
        self.apply_nonlin = softmax_helper_dim1
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, partial_type):
#         print(net_output.shape)
#         print(target.shape)
#         print(partial_type)
        new_net_output, new_target = net_output.clone(), target.clone()
        #new_net_output_soft = self.apply_nonlin(new_net_output)
        #print(f"dc old: {dc_old}, dc:{self.dc(new_net_output_soft, new_target)}")
        #print(f"ce old: {ce_old}, ce:{self.ce(torch.log(new_net_output_soft), new_target.squeeze().type(torch.cuda.LongTensor))}")
        # if partial_type[0]==14:
        #     new_net_output, new_target = merge_prediction(new_net_output,
        #                                                    new_target,
        #                                                    partial_type)
        # else:
        if len(partial_type) < 13:
            new_net_output, new_target = merge_prediction_max_avg(new_net_output,
                                                           new_target,
                                                           partial_type)
        # filter other class output 
        dc_loss = self.dc(new_net_output, new_target)
        ce_loss = self.ce(new_net_output, new_target)
        # # ce_loss = self.ce(torch.log(new_net_output_soft), 
        #                   new_target.squeeze().type(torch.cuda.LongTensor))
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            # reserved for other stuff (later?)
            raise NotImplementedError("nah son")
        return result


class DC_CE_Partial_Filter_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", 
                 weight_ce=1, weight_dice=1,ignore_label=255,ex=True,
                 dice_class=SoftDiceLoss):
        super(DC_CE_Partial_Filter_loss, self).__init__()
        print("*"*10,"Using DC_CE_Partial_Filter_loss","*"*10)
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ignore_label = ignore_label
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        #self.ce = nn.NLLLoss()
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ex_choice = ex
        self.weight_ce = 0
        self.weight_dice = weight_dice
        #self.apply_nonlin = softmax_helper_dim1
        print(f"mode:{aggregate}/ weight:[1:1] with exclusion:{ex}")

    def forward(self, net_output, target, partial_type, do_bg_filter):
        new_net_output, new_target = net_output.clone(), target.clone()
        merge_classes = [item for item in range(1,15) if item not in partial_type]
        if do_bg_filter:
            new_net_output_soft = torch.softmax(new_net_output, dim=1)
            max_prob,max_index = torch.max(new_net_output_soft,dim=1)
            mask = torch.logical_not((max_prob>0.8) & torch.isin(max_index, torch.tensor(merge_classes).cuda())).unsqueeze(1)
        else:
            mask = None
        #new_net_output = self.apply_nonlin(new_net_output)
        new_net_output, new_target = merge_prediction(new_net_output,
                                                new_target,
                                                partial_type)
        # filter other class output 
        
        dc_loss = self.dc(new_net_output, new_target,loss_mask=mask)
        new_target[mask==0] = 255
        ce_loss = self.ce(new_net_output, 
                          new_target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        elif self.aggregate == "ce":
            result = ce_loss
        elif self.aggregate == "dc":
            result = dc_loss
        else:
            # reserved for other stuff (later?)
            raise NotImplementedError("nah son")
        # print(dc_loss)
        return result



def merge_prediction(output, target, partial_type):
    '''
        cur_task: GT task
        default_task: net_output task
    '''
    
    try:
        merge_classes = [item for item in range(0,15) if item not in partial_type]
    except:
        print(f"partial error:{partial_type}")
    #print(f"merge prediction partial type: {partial_type}, merge classes: {merge_classes}")
    merge_output_bg = output[:, merge_classes, :, :].sum(dim=1, keepdim=True)
    output_fg = output[:, partial_type, :, :]
    new_output = torch.cat([merge_output_bg, 
                            output_fg], dim=1)
    for i,label in enumerate(partial_type):
        target[target==label] = i+1
    return new_output, target


def merge_prediction_max(output, target, partial_type):
    '''
        cur_task: GT task
        default_task: net_output task
    '''
 #   partial_type = partial_type.append(14)
  #  print(partial_type)
    #if 14 not in partial_type:
    #    partial_type = torch.cat((partial_type.view(-1),14*torch.ones_like(partial_type)[0]))
    #    print(partial_type)
        #partial_type.append(14)
    #print(partial_type)
    try:
        merge_classes = [item for item in range(0,14) if item not in partial_type]
    except:
        print(f"partial error:{partial_type}")
    #print(f"merge prediction partial type: {partial_type}, merge classes: {merge_classes}")
   # merge_classes = 0
    merge_output_bg, _ = output[:, merge_classes, :, :].max(dim=1, keepdim=True)
    output_fg = output[:, partial_type, :, :]
    new_target = torch.zeros_like(target)
    if 14 not in partial_type:
        new_output = torch.cat([merge_output_bg, 
                            output_fg, output[:,14,...].unsqueeze(1)], dim=1)
    else:
        new_output = torch.cat([merge_output_bg,
                            output_fg], dim=1)
   
    for i,label in enumerate(partial_type):
        new_target[target==label] = i+1

    if 14 not in partial_type:
        new_target[target==14] = len(partial_type) +1 
    #print(new_output.shape, torch.unique(target))
   # print(partial_type,merge_classes, new_output.shape, torch.unique(new_target))
    return new_output, new_target


def merge_prediction_max_ours(output, target, partial_type):
    '''
        cur_task: GT task
        default_task: net_output task
    '''
 #   partial_type = partial_type.append(14)
  #  print(partial_type)
    #if 14 not in partial_type:
    #    partial_type = torch.cat((partial_type.view(-1),14*torch.ones_like(partial_type)[0]))
    #    print(partial_type)
        #partial_type.append(14)
    #print(partial_type)
    try:
        merge_classes = [item for item in range(0,15) if item not in partial_type]
    except:
        print(f"partial error:{partial_type}")
    #print(f"merge prediction partial type: {partial_type}, merge classes: {merge_classes}")
   # merge_classes = 0
    output_bg = output[:, merge_classes, :, :]
    
    prob = F.log_softmax(output, dim=1).exp()
    merge_output_bg, _ = prob[:, merge_classes, :, :].max(dim=1, keepdim=True)
    # 计算merge-max香农熵损失
    
    
    # 计算全监督
#     output_fg = output[:, partial_type, :, :]
#     new_target = torch.zeros_like(target)
#     new_target_bg = torch.zeros_like(target)
#     if 14 not in partial_type:
#         new_output = torch.cat([output[:,0,...], 
#                             output_fg, output[:,14,...].unsqueeze(1)], dim=1)
#     else:
#         new_output = torch.cat([output[:,0,...],
#                             output_fg], dim=1)
    #带有真实标签的层 
#     for i,label in enumerate(partial_type):
#         new_target[target==label] = i
        
#     #不带真实标签的层 
#     for i,label in enumerate(merge_classes):
#         new_target_bg[target==label] = i

#     if 14 not in partial_type:
#         new_target[target==14] = len(partial_type) +1 
    #print(new_output.shape, torch.unique(target))
   # print(partial_type,merge_classes, new_output.shape, torch.unique(new_target))
    return merge_output_bg


def merge_prediction_max_mots(output, target, partial_type):
    '''
        cur_task: GT task
        default_task: net_output task
    '''
 #   partial_type = partial_type.append(14)
  #  print(partial_type)
    #if 14 not in partial_type:
    #    partial_type = torch.cat((partial_type.view(-1),14*torch.ones_like(partial_type)[0]))
    #    print(partial_type)
        #partial_type.append(14)
    #print(partial_type)
    try:
        merge_classes = [item for item in range(0,11) if item not in partial_type]
    except:
        print(f"partial error:{partial_type}")
    #print(f"merge prediction partial type: {partial_type}, merge classes: {merge_classes}")
   # merge_classes = 0
    # print(partial_type)
    # print(merge_classes)
    prob = F.log_softmax(output, dim=1).exp()
    
    output_bg = prob[:, merge_classes, :, :].clone()
   #  print(output_bg.shape)
    merge_output_bg, _ = output_bg.max(dim=1, keepdim=True)
    # 计算merge-max香农熵损失
    
    
    # 计算全监督
    output_fg = output[:, partial_type, :, :].clone()
    new_target = torch.zeros_like(target)
    new_target_bg = torch.zeros_like(target)
    
    new_output = torch.cat([merge_output_bg,
                            output_fg], dim=1)
    #带有真实标签的层 
    for i,label in enumerate(partial_type):
        new_target[target==label] = i+1
        
#     #不带真实标签的层 
#     for i,label in enumerate(merge_classes):
#         new_target_bg[target==label] = i
  #   print(new_output.shape)
  #   print(new_target.shape)
#     if 14 not in partial_type:
#         new_target[target==14] = len(partial_type) +1 
    #print(new_output.shape, torch.unique(target))
   # print(partial_type,merge_classes, new_output.shape, torch.unique(new_target))
    return new_output, new_target, output_bg



def merge_prediction_max_vessel(output, target, partial_type):
    '''
        cur_task: GT task
        default_task: net_output task
    '''
 #   partial_type = partial_type.append(14)
  #  print(partial_type)
    #if 14 not in partial_type:
    #    partial_type = torch.cat((partial_type.view(-1),14*torch.ones_like(partial_type)[0]))
    #    print(partial_type)
        #partial_type.append(14)
    #print(partial_type)
    try:
        merge_classes = [item for item in range(0,11) if item not in partial_type]
    except:
        print(f"partial error:{partial_type}")
    #print(f"merge prediction partial type: {partial_type}, merge classes: {merge_classes}")
   # merge_classes = 0
   #  print(partial_type)
   #  print(merge_classes)
    prob = F.log_softmax(output, dim=1).exp()
    
    output_bg = prob[:, merge_classes, :, :].clone()
   #  print(output_bg.shape)
    merge_output_bg, _ = output_bg.max(dim=1, keepdim=True)
    # 计算merge-max香农熵损失
    
    
    # 计算全监督
    output_fg = output[:, partial_type, :, :].clone()
    new_target = torch.zeros_like(target)
    new_target_bg = torch.zeros_like(target)
    
    new_output = torch.cat([merge_output_bg,
                            output_fg], dim=1)
    #带有真实标签的层 
    for i,label in enumerate(partial_type):
        new_target[target==label] = i+1
        
#     #不带真实标签的层 
#     for i,label in enumerate(merge_classes):
#         new_target_bg[target==label] = i
  #   print(new_output.shape)
  #   print(new_target.shape)
#     if 14 not in partial_type:
#         new_target[target==14] = len(partial_type) +1 
    #print(new_output.shape, torch.unique(target))
   # print(partial_type,merge_classes, new_output.shape, torch.unique(new_target))
    return new_output, new_target, output_bg



def merge_prediction_max_avg(output, target, partial_type):
    '''
        cur_task: GT task
        default_task: net_output task
    '''
 #   partial_type = partial_type.append(14)
  #  print(partial_type)
    #if 14 not in partial_type:
    #    partial_type = torch.cat((partial_type.view(-1),14*torch.ones_like(partial_type)[0]))
    #    print(partial_type)
        #partial_type.append(14)
    #print(partial_type)
    try:
        merge_classes = [item for item in range(0,14) if item not in partial_type]
    except:
        print(f"partial error:{partial_type}")
    #print(f"merge prediction partial type: {partial_type}, merge classes: {merge_classes}")
   # merge_classes = 0
    merge_output_bg, _ = output[:, merge_classes, :, :].sum(dim=1, keepdim=True)
    output_fg = output[:, partial_type, :, :]
    new_target = torch.zeros_like(target)
    if 14 not in partial_type:
        new_output = torch.cat([merge_output_bg, 
                            output_fg, output[:,14,...].unsqueeze(1)], dim=1)
    else:
        new_output = torch.cat([merge_output_bg,
                            output_fg], dim=1)
   
    for i,label in enumerate(partial_type):
        new_target[target==label] = i+1

    if 14 not in partial_type:
        new_target[target==14] = len(partial_type) +1 
    #print(new_output.shape, torch.unique(target))
   # print(partial_type,merge_classes, new_output.shape, torch.unique(new_target))
    return new_output, new_target