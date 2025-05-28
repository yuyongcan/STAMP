import logging

logger = logging.getLogger(__name__)
import os

import numpy as np
from src.methods import setup_model
from src.utils.utils import get_accuracy, merge_cfg_from_args, get_args
from src.utils.conf import cfg, load_cfg_fom_args
from src.data.data import load_ood_dataset_test


def validation(cfg):
    model = setup_model(cfg)
    # get the test sequence containing the corruptions or domain names
    dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    if cfg.MODEL.CONTINUAL == 'Fully':
        dom_names_loop = [dom_names_all[0]]
    elif cfg.MODEL.CONTINUAL == 'Continual':
        dom_names_loop = dom_names_all
    # setup the severities for the gradual setting

    severities = [cfg.CORRUPTION.SEVERITY[0]]

    accs = []
    aucs = []
    h_scores = []

    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):
        if cfg.MODEL.CONTINUAL == 'Fully':
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
        elif cfg.MODEL.CONTINUAL == 'Continual':
            logger.warning("not resetting model")

        for severity in severities:
            testset, test_loader = load_ood_dataset_test(cfg.DATA_DIR, cfg.CORRUPTION.ID_DATASET,
                                                         cfg.CORRUPTION.OOD_DATASET, cfg.CORRUPTION.NUM_OOD_SAMPLES,
                                                         batch_size=cfg.TEST.BATCH_SIZE,
                                                         domain=domain_name, level=severity,
                                                         adaptation=cfg.MODEL.ADAPTATION,
                                                         workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                                         ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                                         num_aug=cfg.TEST.N_AUGMENTATIONS if cfg.MODEL.ADAPTATION != 'aug_robust' else cfg.AUG_ROBUST.NUM_AUG)
            for epoch in range(cfg.TEST.EPOCH):
                acc, auc = get_accuracy(
                    model, data_loader=test_loader, cfg=cfg)
            h_score = 2 * acc * auc / (acc + auc)
            accs.append(acc)
            aucs.append(auc)
            h_scores.append(h_score)
            logger.info(
                f"{cfg.CORRUPTION.ID_DATASET} with {cfg.CORRUPTION.OOD_DATASET} [#samples={len(testset)}][{domain_name}]"
                f":acc: {acc:.2%}, auc: {auc:.2%}, h-score: {h_score:.2%}")

        logger.info(f"mean acc: {np.mean(accs):.2%}, "
                    f"mean auc: {np.mean(aucs):.2%}, "
                    f"mean h-score: {np.mean(h_scores):.2%}")


if __name__ == "__main__":
    args = get_args()
    args.output_dir = args.output_dir if args.output_dir else 'validation_os'
    load_cfg_fom_args(args.cfg, args.output_dir)
    merge_cfg_from_args(cfg, args)
    cfg.CORRUPTION.SOURCE_DOMAIN = cfg.CORRUPTION.SOURCE_DOMAINS[0]
    logger.info(cfg)
    validation(cfg)
