import argparse, time


def init_argument():
    # 项目地址
    #PROJECT_DATA_DIR = "/home/ycshi/dialogue-summary/T5-pegasus/data"
    TEST_RESULT_PATH = "/home/ycshi/dialogue-summary/T5-pegasus/cache/JD"
    current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(time.time()))

    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--train_data_path', default=f'')
    parser.add_argument('--valid_data_path', default=f'')
    parser.add_argument('--test_data_path', default=f'')
    parser.add_argument('--train_data_nums', default=100,type=int, help="训练数据多少")
    parser.add_argument('--shuffle',  action="store_true", help="data if use shuffle")
    parser.add_argument('--cache',  action="store_true", help="if use cache")
    parser.add_argument('--cache_path', default='')

    parser.add_argument('--pretrain_model', default='/home/ycshi/huggingface/imxlyt5-pegasus')
    parser.add_argument('--trained_model', default='1200_dialogs')
    parser.add_argument('--seed', default=42, type=int, help="seed")

    parser.add_argument('--num_epoch', default=15, type=int, help='number of epoch')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--infference_batch_size', default=8, help='batch size')
    parser.add_argument('--lr', default=5e-5, help='learning rate')
    parser.add_argument('--data_parallel', default=True,action="store_true", help="if use multy gpus")

    parser.add_argument('--encoder_max_len', default=512, help='max length of inputs')
    parser.add_argument('--decoder_max_len', default=128, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=128, help='max length of outputs')
    parser.add_argument('--result_path', default=f"{TEST_RESULT_PATH}")
    parser.add_argument('--result_file', default="predict_result.tsv")


    parser.add_argument('--project_dir', default="")
    parser.add_argument('--result_name', default="results_512_new_no_shuffle")
    parser.add_argument('--current_project_name', default=f"{current_time}", help='project name')

    args = parser.parse_args()
    return args