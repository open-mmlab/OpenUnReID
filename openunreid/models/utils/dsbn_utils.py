# Written by Yixiao Ge

import torch.nn as nn
import copy
from ..layers.domain_specific_bn import DSBN


def convert_dsbn(model, num_domains=2, target_bn_idx=-1):
    '''
    convert all bn layers in the model to domain-specific bn layers
    '''

    for _, (child_name, child) in enumerate(model.named_children()):

        if isinstance(child, nn.BatchNorm2d):
            # BN2d -> DSBN2d
            m = DSBN(child.num_features, num_domains, nn.BatchNorm2d, target_bn_idx,
                        child.weight.requires_grad, child.bias.requires_grad)
            m.to(next(child.parameters()).device)

            for idx in range(num_domains):
                m.dsbn[idx].load_state_dict(child.state_dict())

            setattr(model, child_name, m)

        elif isinstance(child, nn.BatchNorm1d):
            # BN1d -> DSBN1d
            m = DSBN(child.num_features, num_domains, nn.BatchNorm1d, target_bn_idx,
                        child.weight.requires_grad, child.bias.requires_grad)
            m.to(next(child.parameters()).device)

            for idx in range(num_domains):
                m.dsbn[idx].load_state_dict(child.state_dict())

            setattr(model, child_name, m)

        else:
            # recursive searching
            convert_dsbn(child, num_domains=num_domains, target_bn_idx=target_bn_idx)


def convert_bn(model, target_bn_idx=-1):
    '''
    convert all domain-specific bn layers in the model back to normal bn layers
    you need to do convert_sync_bn again after this function, if you use sync bn in the model
    '''

    for _, (child_name, child) in enumerate(model.named_children()):

        if isinstance(child, DSBN):
            # DSBN 1d/2d -> BN 1d/2d
            m = child.batchnorm_layer(child.num_features)
            m.weight.requires_grad_(child.weight_requires_grad)
            m.bias.requires_grad_(child.bias_requires_grad)
            m.to(next(child.parameters()).device)

            m.load_state_dict(child.dsbn[target_bn_idx].state_dict())

            setattr(model, child_name, m)

        else:
            # recursive searching
            convert_bn(child, target_bn_idx=target_bn_idx)


def extract_single_bn_model(model, target_bn_idx=-1):
    '''
    extract a model with normal bn layers from the domain-specific bn models
    '''
    model_cp = copy.deepcopy(model)
    convert_bn(model_cp, target_bn_idx=target_bn_idx)
    return model_cp


def switch_target_bn(model, target_bn_idx=-1):
    '''
    switch the target_bn_idx of all domain-specific bn layers
    '''

    for _, (child_name, child) in enumerate(model.named_children()):

        if isinstance(child, DSBN):
            child.target_bn_idx = target_bn_idx

        else:
            # recursive searching
            switch_target_bn(child, target_bn_idx=target_bn_idx)
