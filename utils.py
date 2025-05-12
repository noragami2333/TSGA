import os
import random
import numpy as np
import torch
import time
from datetime import datetime
import csv
import sys
from torch import nn, autograd
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict
from torchmeta.modules import MetaModule
from torchmeta.datasets.helpers import omniglot, cifar_fs, fc100, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from data import create_miniimagenet_task_distribution, create_omniglot_allcharacters_task_distribution, create_cifarfs_task_distribution


class GradientDesent(nn.Module):
    def __init__(self, params, n_inner_steps, plus=True, lr=1e-2, wd=5e-4):
        super(GradientDesent, self).__init__()
        # self.device = device
        self.lr = torch.ones(1) * lr
        # self.lr.to(self.device)
        self.wd = torch.ones(1) * lr * wd
        # self.wd.to(self.device)
        self.alpha = nn.ParameterDict()
        self.beta = nn.ParameterDict()
        for k, v in params.items():
            # print(k, v.shape)
            # KeyError: 'parameter name can\'t contain "."'
            # We had to add this line of code because of the above error :(
            k = k.replace('.', '-')
            self.alpha[k] = nn.Parameter(torch.ones(n_inner_steps) * self.lr, requires_grad=plus)
            self.beta[k] = nn.Parameter(torch.ones(n_inner_steps) * self.wd, requires_grad=plus)
    
    def update(self, model, loss, step, params=None, adapter=None, first_order=False):
        if not isinstance(model, MetaModule):
            raise ValueError('The model must be an instance of `torchmeta.modules.MetaModule`, got `{0}`'.format(type(model)))

        if params is None:
            params = OrderedDict(model.meta_named_parameters())

        grads = autograd.grad(loss, params.values(), create_graph=not first_order)

        if adapter is not None:
            task_embedding = []
            for v in params.values():
                task_embedding.append(v.mean())
            for grad in grads:
                task_embedding.append(grad.mean())
            task_embedding = torch.stack(task_embedding)
            factors = adapter(task_embedding)
            alpha, beta = torch.chunk(factors, 2)
            alpha_params = {k: alpha[i] for i, k in enumerate(params.keys())}
            beta_params = {k: beta[i] for i, k in enumerate(params.keys())}

        updated_params = OrderedDict()

        for (name, param), grad in zip(params.items(), grads):
            k = name.replace('.', '-')
            if adapter is not None:
                updated_params[name] = (1 - beta_params[name] * self.beta[k][step]) * param - alpha_params[name] * self.alpha[k][step] * grad
            else:
                updated_params[name] = (1 - self.beta[k][step]) * param - self.alpha[k][step] * grad

        return updated_params


def gradient_update_parameters(model, loss, params=None, lr=0.01, wd=5e-4, first_order=False):
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = autograd.grad(loss, params.values(), create_graph=not first_order)

    updated_params = OrderedDict()

    if isinstance(lr, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = (1 - lr[name] * wd) * param - lr[name] * grad
    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = (1 - lr * wd) * param - lr * grad

    return updated_params


def gradient_update_parameters_with_adapter(model, loss, adapter, params=None, step_size=0.5, first_order=False):
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = autograd.grad(loss, params.values(), create_graph=not first_order)
    
    task_embedding = []
    for v in params.values():
        task_embedding.append(v.mean())
    for grad in grads:
        task_embedding.append(grad.mean())
    task_embedding = torch.stack(task_embedding)
    factors = adapter(task_embedding)
    alpha, beta = torch.chunk(factors, 2)
    alpha_params = {k: alpha[i] for i, k in enumerate(params.keys())}
    beta_params = {k: beta[i] for i, k in enumerate(params.keys())}
    
    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = (1 - alpha_params[name] * 1e-2 * beta_params[name] * 5e-4) * param - alpha_params[name] * 1e-2 * grad
    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = (1 - alpha_params[name] * 1e-2 * beta_params[name] * 5e-4) * param - alpha_params[name] * 1e-2 * grad

    return updated_params


class OCCData_v2:
    def __init__(self, args):
        if args.dataset == 'MIN':
            self.train_dataloader = BatchMetaDataLoader(miniimagenet(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='train', 
                transform=transforms.Compose([
                    transforms.RandomCrop(84, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            ), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            self.val_dataloader = BatchMetaDataLoader(miniimagenet(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='val', 
            ), batch_size=1, shuffle=True, num_workers=args.num_workers)
            self.val_dataloader = BatchMetaDataLoader(miniimagenet(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='test', 
            ), batch_size=1, shuffle=True, num_workers=args.num_workers)
        elif args.dataset == 'CIFAR':
            self.train_dataloader = BatchMetaDataLoader(cifar_fs(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='train', 
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            ), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            self.val_dataloader = BatchMetaDataLoader(cifar_fs(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='val', 
            ), batch_size=1, shuffle=True, num_workers=args.num_workers)
            self.val_dataloader = BatchMetaDataLoader(cifar_fs(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='test', 
            ), batch_size=1, shuffle=True, num_workers=args.num_workers)
        elif args.dataset == 'FC':
            self.train_dataloader = BatchMetaDataLoader(fc100(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='train', 
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            ), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            self.val_dataloader = BatchMetaDataLoader(fc100(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='val', 
            ), batch_size=1, shuffle=True, num_workers=args.num_workers)
            self.val_dataloader = BatchMetaDataLoader(fc100(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='test', 
            ), batch_size=1, shuffle=True, num_workers=args.num_workers)
        elif args.dataset == 'OMN':
            self.train_dataloader = BatchMetaDataLoader(omniglot(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='train', 
            ), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            self.val_dataloader = BatchMetaDataLoader(omniglot(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='val', 
            ), batch_size=1, shuffle=True, num_workers=args.num_workers)
            self.val_dataloader = BatchMetaDataLoader(omniglot(
                args.folder, shots=args.n_support, ways=args.n_query+1, shuffle=True, test_shots=args.n_query, meta_split='test', 
            ), batch_size=1, shuffle=True, num_workers=args.num_workers)
        else:
            print('Wrong datasets !')
            exit()


class OCCData:
    def __init__(self, args):

        if args.dataset == 'MIN':
            self.metatrain_task_distribution, self.metaval_task_distribution, self.metatest_task_distribution = create_miniimagenet_task_distribution(
                "/data/lyl/FS-OCC/miniimagenet/miniimagenet.pkl",
                train_occ=True,
                test_occ=True,
                num_training_samples_per_class=args.n_support, 
                num_test_samples_per_class=int(args.n_query / 2),
                num_training_classes=2,
                meta_batch_size=args.batch_size,
                seq_length=0
            )
        elif args.dataset == 'CIFAR':
            self.metatrain_task_distribution, self.metaval_task_distribution, self.metatest_task_distribution = create_cifarfs_task_distribution(
                "/data/lyl/FS-OCC/cifarfs/cifarfs.pkl",
                train_occ=True,
                test_occ=True,
                num_training_samples_per_class=args.n_support, 
                num_test_samples_per_class=int(args.n_query / 2),
                num_training_classes=2,
                meta_batch_size=args.batch_size,
                seq_length=0
            )
        elif args.dataset == 'OMN':
            self.metatrain_task_distribution, self.metaval_task_distribution, self.metatest_task_distribution = create_omniglot_allcharacters_task_distribution(
                "/data/lyl/FS-OCC/omniglot/omniglot.pkl",
                train_occ=True,
                test_occ=True,
                num_training_samples_per_class=args.n_support,
                num_test_samples_per_class=int(args.n_query / 2),
                num_training_classes=2,
                meta_batch_size=args.batch_size,
                seq_length=0
            )

    def get_episode(self, meta_batch):
        support_X, support_y, query_X, query_y = [], [], [], []
        for task in meta_batch:  # 8
            support, query = task.get_train_set(), task.get_test_set()
            support_X.append(support[0])  # support set
            support_y.append(support[1])
            query_X.append(query[0])  # query set
            query_y.append(query[1])
        support_X = torch.from_numpy(np.array(support_X))  # [B, K, 3, H, W]
        support_y = torch.from_numpy(np.array(support_y))  # [B, K, 1]
        query_X = torch.from_numpy(np.array(query_X))  # [B, Q, 3, H, W]
        query_y = torch.from_numpy(np.array(query_y))  # [B, Q, 1]
        return support_X, support_y, query_X, query_y

    def get_train_episode(self):
        meta_batch = self.metatrain_task_distribution.sample_batch()
        return self.get_episode(meta_batch)

    def get_val_episode(self):
        meta_batch = self.metaval_task_distribution.sample_batch(1)
        return self.get_episode(meta_batch)

    def get_test_episode(self, n=1):
        meta_batch = self.metatest_task_distribution.sample_batch(n)
        return self.get_episode(meta_batch)


def set_device(gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # # speed up
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.cudnn_enabled = False


def format_runtime(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}分")
    parts.append(f"{seconds}秒")
    return "".join(parts)


def time_monitor(func):
    def wrapper(*args, **kwargs):
        tic = time.time()
        print("🚀 Program startup time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"❌ 执行出错：{str(e)}")
            raise
        finally:
            toc = time.time()
            runtime = toc - tic
            print("🕒 Program runtime: {}".format(format_runtime(runtime)))
            print("⏰ Program end time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return result
    return wrapper


def save_statistics(line_to_add, filename="summary_statistics.csv", create=False):
    with open(filename, 'w' if create else 'a') as f:
        writer = csv.writer(f)
        writer.writerow(line_to_add)
    return filename


class LossNet(nn.Module):
    def __init__(self, args):
        super(LossNet, self).__init__()
        self.args = args
        self.ffn = nn.Sequential(
            nn.Linear(2, self.args.dim_feedforward_loss),  # self.classifier.get_fea_dim() + 
            nn.ELU(inplace=True),
            nn.Linear(self.args.dim_feedforward_loss, self.args.dim_feedforward_loss), 
            nn.ELU(inplace=True),
            nn.Linear(self.args.dim_feedforward_loss, self.args.dim_feedforward_loss), 
            nn.ELU(inplace=True),
            nn.Linear(self.args.dim_feedforward_loss, 1), 
        )
    
    def forward(self, p):
        return self.ffn(p).mean()


class AdaptNet(nn.Module):
    def __init__(self, args):
        super(AdaptNet, self).__init__()
        self.args = args
        self.trans = nn.TransformerEncoder(
            TransformerEncoderLayer(d_model=1, nhead=1, dim_feedforward=self.args.dim_feedforward_adapter), 
            self.args.num_layers_adapter
        )
    
    def forward(self, a):
        return torch.sigmoid(self.trans(a.reshape(-1, 1, 1)).squeeze())


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


def get_state(net, log=False, flg=False):
    state = []
    params = net.named_parameters()
    for k, v in params.items():
        # print(k, v.shape)
        dim = 0 if flg and 'fc' in k else -1
        state.append(torch.mean(v.reshape(v.shape[0], -1), dim))
    sz = [x.shape[0] for x in state]
    if log: print(sz)
    num_layer = len(state)
    state = torch.cat(state, dim=0)
    if log: print(state.shape)
    # state = (state - state.mean()) / (state.std() + 1e-12)
    return state, num_layer, params, sz


class AAN(nn.Module):
    def __init__(self, sz):
        super().__init__()
        dim = sum(sz)
        # print(sz, dim, min(sz))
        dim = dim // min(sz)
        # print(dim, dim//2, dim//4, dim//8)
        flg = [(dim // i) % 2 for i in (1, 2, 4, 8)]
        # print(flg)
        self.dw_conv1 = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.dw_bn1 = nn.BatchNorm2d(16)
        self.dw_conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=False)
        self.dw_bn2 = nn.BatchNorm2d(32)
        self.dw_conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.dw_bn3 = nn.BatchNorm2d(64)
        self.dw_conv4 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.dw_bn4 = nn.BatchNorm2d(128)
        self.up_conv3 = nn.ConvTranspose2d(128, 64, (2+flg[2], 2), 2, bias=False)
        self.up_bn3 = nn.BatchNorm2d(64)
        self.up_samp3 = nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.up_samp_bn3 = nn.BatchNorm2d(64)
        self.up_conv2 = nn.ConvTranspose2d(64, 32, (2+flg[1], 2), 2, bias=False)
        self.up_bn2 = nn.BatchNorm2d(32)
        self.up_samp2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.up_samp_bn2 = nn.BatchNorm2d(32)
        self.up_conv1 = nn.ConvTranspose2d(32, 16, (2+flg[0], 2), 2, bias=False)
        self.up_bn1 = nn.BatchNorm2d(16)
        self.up_samp1 = nn.Conv2d(32, 16, 3, 1, 1, bias=False)
        self.up_samp_bn1 = nn.BatchNorm2d(16)
        self.up_conv0 = nn.Conv2d(16, 1, 3, 1, 1, bias=False)
        self.up_bn0 = nn.BatchNorm2d(1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, params, mode='mean'):
        state = []
        for k, v in params.items():
            if 'fc' in k:
                if mode == 'mean':
                    state.append(torch.mean(v.reshape(v.shape[0], -1), dim=0))
                elif mode == 'max':
                    state.append(torch.max(v.reshape(v.shape[0], -1), dim=0)[0])
            else:
                if mode == 'mean':
                    state.append(torch.mean(v.reshape(v.shape[0], -1), dim=-1))
                elif mode == 'max':
                    state.append(torch.max(v.reshape(v.shape[0], -1), dim=-1)[0])
        sz = [x.shape[0] for x in state]
        # print(sz)
        state = torch.cat(state, dim=0)
        state = state.view(1, 1, -1, min(sz))
        # state = (state - state.mean()) / (state.std() + 1e-12)
        # 1, H, W
        a10 = F.relu_(self.dw_bn1(self.dw_conv1(state)))
        # 16, H, W
        a11 = self.maxpool(a10)
        # 16, H/2, W/2
        a20 = F.relu_(self.dw_bn2(self.dw_conv2(a11)))
        # 32, H/2, W/2
        a21 = self.maxpool(a20)
        # 32, H/4, W/4
        a30 = F.relu_(self.dw_bn3(self.dw_conv3(a21)))
        # 64, H/4, W/4
        a31 = self.maxpool(a30)
        # 64, H/8, W/8
        b31 = F.relu_(self.dw_bn4(self.dw_conv4(a31)))
        # 128, H/8, W/8
        b30 = F.relu_(self.up_bn3(self.up_conv3(b31)))
        # 64, H/4, W/4
        b21 = F.relu_(self.up_samp_bn3(self.up_samp3(torch.cat((a30, b30), dim=1))))
        # 64, H/4, W/4
        b20 = F.relu_(self.up_bn2(self.up_conv2(b21)))
        # 32, H/2, W/2
        b11 = F.relu_(self.up_samp_bn2(self.up_samp2(torch.cat((a20, b20), dim=1))))
        # 32, H/2, W/2
        b10 = F.relu_(self.up_bn1(self.up_conv1(b11)))
        # 16, H, W
        b01 = F.relu_(self.up_samp_bn1(self.up_samp1(torch.cat((a10, b10), dim=1))))
        # 16, H, W
        b00 = F.relu_(self.up_bn0(self.up_conv0(b01)))
        # 1, H, W
        out = torch.sigmoid(b00).view(-1)
        generated_multiplier = torch.split(out, split_size_or_sections=sz, dim=-1)
        # multiplier_bias = torch.split(self.multiplier_bias, split_size_or_sections=sz, dim=-1)
        updated_params = dict()
        for i, (key, val) in enumerate(params.items()):
            if 'fc' in key:
                dim = [1 for _ in range(len(list(val.shape)) - 1)] + [-1]
            else:
                dim = [-1] + [1 for _ in range(len(list(val.shape)) - 1)]
            # updated_params[key] = (1 + self.multiplier_bias[i] * generated_multiplier[i]) * val + self.offset_bias[i] * generated_offset[i]
            updated_params[key] = torch.mul(val, generated_multiplier[i].view(dim))
        return updated_params
