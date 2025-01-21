from utils.config_utils import load_yaml, build_record_folder, get_args
from run_HERBS import run_HERBS_train
from run_clf_model import run_clf_model_train
def main():
    """
    run main process

    Args:
        opt (_type_): option
    """

    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)

    model_type = args.model_type
    args.exp_name = args.exp_name + f'_{model_type}'
    if model_type == 'HERBS':
        run_HERBS_train(args)
    else:
        run_clf_model_train(args)

if __name__ == '__main__':
    main()