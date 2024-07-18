import logging, os
import torch, random
import numpy as np

def get_logger(args):
    current_project = f"{args.project_dir}/{args.current_project_name}"

    if not os.path.exists(current_project):
        os.mkdir(current_project)

    log_name = os.path.join(current_project, "train.log")
    logging.basicConfig(level=logging.DEBUG,
                    filename=log_name,
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger()

def write_import_arsg_to_file(args, logger):
    logger.warning(args)
    logger.info(f"batch_size:{args.batch_size}")
    logger.info(f"train_data_nums:{args.train_data_nums}")
    logger.info(f"lr:{args.lr}")
    logger.info(f"num_epoch:{args.num_epoch}")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True