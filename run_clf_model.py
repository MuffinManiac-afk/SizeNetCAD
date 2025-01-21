import sys,os,time
import torch
import wandb,shutil
from utils.config_utils import build_record_folder
from models.classification_model import clf_model
from utils.truncate_dataset import truncate_new_dataset
from sklearn.model_selection import StratifiedKFold
from data.dataset import build_loader
from eval import get_length_dict
from utils.lr_schedule import get_lr
from data.dataset import get_class2num
from utils.combine_bg import combine_bg
os.environ['TORCH_HOME']=os.path.join('pretrained_model') #set pretrained model folder path

def set_environment(args, is_train_aug=False, fold_img_label=-1,img_size=224,batch_size=32):
    """
        create data loader, model
    """
    print("Setting Environment...")

    args.device = torch.device(args.which_gpu)

    ### = = = =  Dataset and Data Loader = = = =
    print("Building Dataloader....")

    train_loader, val_loader = build_loader(args, is_train_aug, fold_img_label,img_size,batch_size)

    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")

    if train_loader is not None:
        print("    Train Samples: {} (batch: {})".format(len(train_loader.dataset), len(train_loader)))
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")

    class2num = get_class2num(args.train_root)
    args.num_classes = len(class2num)
    ### = = = =  Model = = = =
    print("Building Model....")
    batch_size = 32
    learning_rate = 0.001
    model = clf_model(len(class2num), learning_rate, batch_size, args.model_type).to('cuda')
    return train_loader, val_loader, model
def train(args,num_epochs,model,train_data_loader,valid_data_loader):
    """
        training process
    """
    # init accuracy, loss
    min_loss = sys.float_info.max
    the_current_loss = 0.0
    best_acc = 0.0
    # init loss,accuracy
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        train_loss, train_acc = model.train_model(args,train_data_loader)
        val_loss, val_acc = model.run_validation(args,valid_data_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if args.use_wandb:
            msg = {}
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(model.optimizer)
            msg["train_acc/acc"] = train_acc
            msg["val_acc/acc"] = val_acc
            msg["train_loss/loss"] = train_loss
            msg["val_loss/loss"] = val_loss
            wandb.log(msg)

        print(
            "Epoch:{}/{}  Training Loss:{:.3f} || Training Acc {:.3f} % ||  valid Loss:{:.3f} ||  valid Acc {:.3f} %".format(
                epoch + 1,
                num_epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc))

        min_loss = min(the_current_loss, min_loss)
        if val_acc >= best_acc:
            best_acc = val_acc
            save_dict = {
                'model_name': args.model_type,
                'num_class': args.num_classes,
                'model_state_dict': model.state_dict(),
                "epoch": epoch,
                'img_size': args.data_size,
                'class2num': get_class2num(args.train_root),
                'real_length_dict': args.length_dict,
            }
            torch.save(save_dict, os.path.join(args.save_dir, "save_model", "best.pth"))

    del train_data_loader
    del valid_data_loader

def start_train(args,fold_img_label=None,idx_fold=''):
    """
        train model and save best model
    """
    batch_size = 32
    args.data_size = 224
    best_eval_name = "null"
    dataset_path = args.train_root
    is_train_aug = args.is_train_aug
    train_loader, val_loader, model = set_environment(args,is_train_aug,fold_img_label,args.data_size,batch_size)
    best_acc = 0.0
    try:
        args.length_dict= get_length_dict(os.path.join(args.train_root,'log.txt'))
    except:
        args.length_dict=None
    if args.use_wandb:
        wandb.init(entity=args.wandb_entity,
                   project=args.project_name,
                   name=args.exp_name,
                   config=args)
        wandb.run.summary["best_acc"] = best_acc
        wandb.run.summary["best_eval_name"] = best_eval_name
        wandb.run.summary["best_epoch"] = 0

    train(args,args.max_epochs,model,train_loader, val_loader)

    if args.use_wandb:
        wandb.finish()
def run_clf_model_train(args):
    """
        truncate train dataset,
        train combine_bg,
        cross validation,
        run model　train process
        """
    start_time = time.time()

    if args.which_gpu == 'cuda:0':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    elif args.which_gpu == 'cuda:1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if args.is_truncate_train_dataset:
        while args.train_root[-1] == '/': args.train_root = args.train_root[:-1]
        truncate_save_path = os.path.join(args.train_root.split(os.path.split(args.train_root)[-1])[0],
                                 'train_truncated')
        truncate_new_dataset(args.train_root, truncate_save_path, args.div_num)
        args.train_root = truncate_save_path
    if args.train_combine_bg:
        while args.train_root[-1] == '/': args.train_root = args.train_root[:-1]
        combine_bg_save_path = os.path.join(args.train_root.split(os.path.split(args.train_root)[-1])[0],
                                            'train_combine_bg')
        from data.dataset import get_train_image_list

        class2num = get_class2num(args.train_root)
        num_classes = len(class2num)
        print("[dataset] class number:", num_classes)
        original_img_path_list, original_img_classes_list = get_train_image_list(args.train_root, class2num)
        combine_bg(args,args.train_root, combine_bg_save_path, original_img_path_list)
        args.train_root = combine_bg_save_path
    if args.is_using_cross_validation:
        skf = StratifiedKFold(n_splits=args.cross_validation_folds, random_state=32, shuffle=True)
        print(f'total {args.cross_validation_folds} fold')
        from data.dataset import get_train_image_list
        class2num = get_class2num(args.train_root)
        num_classes = len(class2num)
        print("[dataset] class number:", num_classes)
        original_img_path_list, original_img_classes_list = get_train_image_list(args.train_root, class2num)
        print(f'total ori:{len(original_img_path_list)}')
        for i, (train_index, valid_index) in enumerate(skf.split(original_img_path_list, original_img_classes_list)):
            # if i<3: continue
            print(f"Fold {i}:")
            # print(f"  Train: index={train_index}")
            # print(f"  valid:  index={valid_index}")

            args.exp_name = args.exp_name.split('_fold_')[0] + '_fold_' + str(i)
            build_record_folder(args)
            start_train(args, (train_index, valid_index), str(i))
            end_time = time.time() - start_time
            print(f'train time:{end_time}')

    else:
        build_record_folder(args)
        start_train(args)
        end_time = time.time() - start_time
        print(f'train time:{end_time}')
        with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
            f.write(f'{args.exp_name} model \n')
            f.write(f'train time:{end_time}')
            f.write('\n')
    if args.is_truncate_train_dataset:
        print('remove truncate folder')
        shutil.rmtree(truncate_save_path, ignore_errors=True)
    if args.train_combine_bg:
        print('remove train_combine_bg folder')
        shutil.rmtree(combine_bg_save_path, ignore_errors=True)


if __name__ == '__main__':
    pass