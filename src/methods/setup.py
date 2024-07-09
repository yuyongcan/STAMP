import math
import torch
from torch import optim, nn
import os
from src.data import load_dataset
from .OWTTT import get_source_features_labels, Prototype_Pool, OWTTT
from .T3A import T3A
from .adacontrast import AdaContrast
from .stamp import STAMP
from .bn import AlphaBatchNorm
from .cotta import CoTTA
from .eata import EATA
from .lame import LAME
from .memo import MEMO
from .norm import Norm
from .sar import SAR
from .sotta import SoTTA
from .tent import Tent
from ..models import BaseModel
from ..utils.sam import SAM
from ..utils.utils import split_up_model, get_source_features_labels, Prototype_Pool
from .rotta import RoTTA
from .odin import ODIN
from src.models import OfficeHome_Shot
from src.models import DomainNet126_Shot
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def setup_optimizer(params, cfg):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                          lr=cfg.OPTIM.LR,
                          betas=(cfg.OPTIM.BETA, 0.999),
                          weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                         lr=cfg.OPTIM.LR,
                         momentum=cfg.OPTIM.MOMENTUM,
                         dampening=cfg.OPTIM.DAMPENING,
                         weight_decay=cfg.OPTIM.WD,
                         nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def setup_adacontrast_optimizer(model, cfg):
    backbone_params, extra_params = (
        model.src_model.get_params()
        if hasattr(model, "src_model")
        else model.get_params()
    )

    if cfg.OPTIM.METHOD == "SGD":
        optimizer = optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": cfg.OPTIM.LR,
                    "momentum": cfg.OPTIM.MOMENTUM,
                    "weight_decay": cfg.OPTIM.WD,
                    "nesterov": cfg.OPTIM.NESTEROV,
                },
                {
                    "params": extra_params,
                    "lr": cfg.OPTIM.LR * 10,
                    "momentum": cfg.OPTIM.MOMENTUM,
                    "weight_decay": cfg.OPTIM.WD,
                    "nesterov": cfg.OPTIM.NESTEROV,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{cfg.OPTIM.METHOD} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer


def setup_NRC_optimizer(model, cfg):
    if isinstance(model, (OfficeHome_Shot, DomainNet126_Shot)):
        param_group = []
        param_group_c = []
        for k, v in model.netF.named_parameters():
            if True:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 0.1}]

        for k, v in model.netB.named_parameters():
            if True:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]
        for k, v in model.netC.named_parameters():
            param_group_c += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]
        optimizer = optim.SGD(param_group)
        optimizer_c = optim.SGD(param_group_c)
        return op_copy(optimizer), op_copy(optimizer_c)
    else:
        encoder, classifier = split_up_model(model)
        param_group = []
        param_group_c = []
        for k, v in encoder.named_parameters():
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 0.1}]
        for k, v in classifier.named_parameters():
            param_group_c += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]
        optimizer = optim.SGD(param_group)
        optimizer_c = optim.SGD(param_group_c)
        return op_copy(optimizer), op_copy(optimizer_c)


def setup_shot_optimizer(model, cfg):
    if isinstance(model, (OfficeHome_Shot, DomainNet126_Shot)):
        param_group = []
        for k, v in model.netF.named_parameters():
            if True:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 0.1}]

        for k, v in model.netB.named_parameters():
            if True:
                param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]
        for k, v in model.netC.named_parameters():
            v.requires_grad = False
        optimizer = optim.SGD(param_group)
        return op_copy(optimizer)
    else:
        encoder, classifier = split_up_model(model)
        param_group = []
        for k, v in encoder.named_parameters():
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 0.1}]
        for k, v in classifier.named_parameters():
            v.requires_grad = False
        optimizer = optim.SGD(param_group)
        return op_copy(optimizer)


def setup_source(model, cfg=None):
    """Set up BN--0 which uses the source model without any adaptation."""
    model.eval()
    return model, None


def setup_t3a(model, cfg=None):
    model.eval()
    T3A_model = T3A(model, filter_k=cfg.T3A.FILTER_K, cached_loader=False)
    return T3A_model, None


def setup_norm_test(model, cfg):
    """Set up BN--1 (test-time normalization adaptation).
    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    model.eval()
    for m in model.modules():
        # Re-activate batchnorm layer
        if (isinstance(m, nn.BatchNorm1d) and cfg.TEST.BATCH_SIZE > 1) or isinstance(m,
                                                                                     (nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()

    # Wrap test normalization into Norm class to enable sliding window approach
    norm_model = Norm(model)
    return norm_model, None


def setup_tent(model, cfg):
    model = Tent.configure_model(model)
    params, param_names = Tent.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    tent_model = Tent(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      )
    return tent_model, param_names


def setup_cotta(model, cfg):
    model = CoTTA.configure_model(model)
    params, param_names = CoTTA.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    cotta_model = CoTTA(model, optimizer,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC,
                        dataset_name=cfg.CORRUPTION.ID_DATASET,
                        mt_alpha=cfg.M_TEACHER.MOMENTUM,
                        rst_m=cfg.COTTA.RST,
                        ap=cfg.COTTA.AP)
    return cotta_model, param_names


def setup_owttt(model, cfg):
    model = BaseModel(model, cfg.MODEL.ARCH)
    model.eval()
    source_dataset, source_loader = load_dataset(dataset=cfg.CORRUPTION.SOURCE_DATASET,
                                                 split='val' if cfg.CORRUPTION.SOURCE_DATASET == 'imagenet' else 'train',
                                                 root=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                                 batch_size=cfg.TEST.BATCH_SIZE,
                                                 )

    source_feature, source_labels = get_source_features_labels(model, source_loader,
                                                               ckpt_dir=os.path.join(cfg.CKPT_DIR, 'features',
                                                                                     cfg.CORRUPTION.SOURCE_DATASET+'_'+cfg.MODEL.ARCH))
    source_prototypes = []
    for i in range(cfg.num_classes):
        source_prototypes.append(source_feature[source_labels == i].mean(0))
    source_prototypes = torch.stack(source_prototypes)
    source_prototypes = F.normalize(source_prototypes)
    source_memory = Prototype_Pool(max=cfg.num_classes, memory=source_prototypes)

    source_distribution = {}
    source_distribution['mu'] = source_prototypes.mean(0)
    source_distribution['cov'] = torch.cov(source_feature.t())

    optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIM.LR, momentum=0.9)
    # optimizer = setup_optimizer(model.parameters(), cfg)

    model = OWTTT(model, optimizer, ce_scale=cfg.OWTTT.CE_SCALE, delta=cfg.OWTTT.DELTA, class_num=cfg.num_classes,
                  source_memory=source_memory,
                  queue_length=cfg.OWTTT.QUEUE_LENGTH, max_prototypes=cfg.OWTTT.MAX_PROTOTYPES,
                  source_distribution=source_distribution)


    return model, None


def setup_eata(model, cfg):
    # compute fisher informatrix
    batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
    fisher_dataset, fisher_loader = load_dataset(dataset=cfg.CORRUPTION.SOURCE_DATASET,
                                                 split='val' if cfg.CORRUPTION.SOURCE_DATASET == 'imagenet' else 'train',
                                                 root=cfg.DATA_DIR, adaptation=cfg.MODEL.ADAPTATION,
                                                 batch_size=batch_size_src,
                                                 domain=cfg.CORRUPTION.SOURCE_DOMAIN)
    # sub dataset
    fisher_dataset = torch.utils.data.Subset(fisher_dataset, range(cfg.EATA.NUM_SAMPLES))
    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=batch_size_src, shuffle=True,
                                                num_workers=cfg.TEST.NUM_WORKERS, pin_memory=True)
    model = EATA.configure_model(model)
    params, param_names = EATA.collect_params(model)
    ewc_optimizer = optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    for iter_, batch in enumerate(fisher_loader, start=1):
        images = batch[0].cuda(non_blocking=True)
        outputs = model(images)
        _, targets = outputs.max(1)
        loss = train_loss_fn(outputs, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    # logger.info("compute fisher matrices finished")
    del ewc_optimizer

    optimizer = setup_optimizer(params, cfg)
    eta_model = EATA(model, optimizer,
                     steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC,
                     fishers=fishers,
                     fisher_alpha=cfg.EATA.FISHER_ALPHA,
                     e_margin=math.log(cfg.num_classes) * (
                         0.40 if cfg.EATA.E_MARGIN_COE is None else cfg.EATA.E_MARGIN_COE),
                     d_margin=cfg.EATA.D_MARGIN
                     )

    return eta_model, param_names


def setup_lame(model, cfg):
    model = LAME.configure_model(model)
    model = BaseModel(model, cfg.MODEL.ARCH)
    lame_model = LAME(model,
                      affinity=cfg.LAME.AFFINITY,
                      knn=cfg.LAME.KNN,
                      sigma=cfg.LAME.SIGMA,
                      force_symmetry=cfg.LAME.FORCE_SYMMETRY)
    return lame_model, None


def setup_memo(model, cfg):
    model, param_names = setup_alpha_norm(model, cfg)
    model = model.cuda()
    # model.eval()
    params, param_names = MEMO.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    memo_model = MEMO(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      n_augmentations=cfg.TEST.N_AUGMENTATIONS,
                      dataset_name=cfg.CORRUPTION.ID_DATASET)
    return memo_model, param_names


def setup_alpha_norm(model, cfg):
    """Set up BN--0.1 (test-time normalization adaptation with source prior).
    Normalize features by combining the source moving statistics and the test batch statistics.
    """
    model.eval()
    norm_model = AlphaBatchNorm.adapt_model(model,
                                            alpha=cfg.BN.ALPHA).cuda()  # (1-alpha) * src_stats + alpha * test_stats
    return norm_model, None


def setup_adacontrast(model, cfg):
    model = AdaContrast.configure_model(model)
    params, param_names = AdaContrast.collect_params(model)

    optimizer = setup_optimizer(params, cfg)

    adacontrast_model = AdaContrast(model, optimizer,
                                    steps=cfg.OPTIM.STEPS,
                                    episodic=cfg.MODEL.EPISODIC,
                                    arch_name=cfg.MODEL.ARCH,
                                    queue_size=cfg.ADACONTRAST.QUEUE_SIZE,
                                    momentum=cfg.M_TEACHER.MOMENTUM,
                                    temperature=cfg.CONTRAST.TEMPERATURE,
                                    contrast_type=cfg.ADACONTRAST.CONTRAST_TYPE,
                                    ce_type=cfg.ADACONTRAST.CE_TYPE,
                                    alpha=cfg.ADACONTRAST.ALPHA,
                                    beta=cfg.ADACONTRAST.BETA,
                                    eta=cfg.ADACONTRAST.ETA,
                                    dist_type=cfg.ADACONTRAST.DIST_TYPE,
                                    ce_sup_type=cfg.ADACONTRAST.CE_SUP_TYPE,
                                    refine_method=cfg.ADACONTRAST.REFINE_METHOD,
                                    num_neighbors=cfg.ADACONTRAST.NUM_NEIGHBORS)
    return adacontrast_model, param_names


def setup_sar(model, cfg):
    sar_model = SAR(model, lr=cfg.OPTIM.LR, batch_size=cfg.TEST.BATCH_SIZE, steps=cfg.OPTIM.STEPS,
                    num_classes=cfg.num_classes, episodic=cfg.MODEL.EPISODIC, reset_constant=cfg.SAR.RESET_CONSTANT,
                    e_margin=math.log(cfg.num_classes) * (
                        0.40 if cfg.SAR.E_MARGIN_COE is None else cfg.SAR.E_MARGIN_COE))

    return sar_model


def setup_rotta(model, cfg):
    model = RoTTA.configure_model(model, cfg.ROTTA.ALPHA)
    params, param_names = RoTTA.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    rotta_model = RoTTA(model, optimizer, cfg.CORRUPTION.ID_DATASET, cfg.ROTTA.MEMORY_SIZE, cfg.num_classes,
                        LAMBDA_T=cfg.ROTTA.LAMBDA_T,
                        LAMBDA_U=cfg.ROTTA.LAMBDA_U, NU=cfg.ROTTA.NU, UPDATE_FREQUENCY=cfg.ROTTA.UPDATE_FREQUENCY,
                        ALPHA=cfg.ROTTA.ALPHA,
                        steps=cfg.OPTIM.STEPS)

    return rotta_model


def setup_sotta(model, cfg):
    model = SoTTA.configure_model(model)
    params, param_names = SoTTA.collect_params(model)
    base_optimizer = optim.Adam
    optimizer = SAM(params, base_optimizer, lr=cfg.OPTIM.LR, rho=0.5)
    model = SoTTA(model, optimizer, cfg.SOTTA.MEMORY_SIZE, cfg.SOTTA.THRESHOLD, cfg.num_classes)
    return model


def setup_odin(base_model, cfg):
    model = ODIN(base_model, cfg.ODIN.TEMP, cfg.ODIN.MAG)
    return model


def setup_stamp(base_model, cfg):
    params, param_names = STAMP.collect_params(base_model)
    # optimizer = setup_optimizer(params, cfg)
    base_optimizer = optim.SGD
    optimizer = SAM(params, base_optimizer, lr=cfg.OPTIM.LR, rho=0.05)
    model = STAMP(base_model, optimizer, cfg.STAMP.ALPHA, cfg.num_classes)
    return model


def setup_model(cfg):
    from src.models.load_model import load_model
    from src.utils.conf import get_num_classes
    cfg.num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.ID_DATASET)
    base_model = load_model(model_name=cfg.MODEL.ARCH, checkpoint_dir=os.path.join(cfg.CKPT_DIR, 'models'),
                            domain=cfg.CORRUPTION.SOURCE_DOMAIN)
    base_model = base_model.cuda()

    logger.info(f"Setting up test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")
    if cfg.MODEL.ADAPTATION == "source":  # BN--0
        model, param_names = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "t3a":
        model, param_names = setup_t3a(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "norm_test":  # BN--1
        model, param_names = setup_norm_test(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "norm_alpha":  # BN--0.1
        model, param_names = setup_alpha_norm(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "memo":
        model, param_names = setup_memo(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "tent":
        model, param_names = setup_tent(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "cotta":
        model, param_names = setup_cotta(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "lame":
        model, param_names = setup_lame(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "adacontrast":
        model, param_names = setup_adacontrast(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "eata":
        model, param_names = setup_eata(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "sar":
        model = setup_sar(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "rotta":
        model = setup_rotta(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == 'owttt':
        model, param_names = setup_owttt(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == 'sotta':
        model = setup_sotta(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == 'odin':
        model = setup_odin(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == 'stamp':
        model = setup_stamp(base_model, cfg)
    else:
        raise ValueError(f"Adaptation method '{cfg.MODEL.ADAPTATION}' is not supported!")
    return model
