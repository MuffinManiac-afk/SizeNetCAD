import torch,os
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import wandb
import warnings
import time

from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils.costom_logger import timeLogger
from utils.config_utils import build_record_folder
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from eval import evaluate, cal_train_metrics, suppression, cal_evalute_metrics
from data.dataset import get_class2num
from eval import get_length_dict
warnings.simplefilter("ignore")
os.environ['TORCH_HOME']=os.path.join('pretrained_model') #set pretrained model folder path
from sklearn.model_selection import StratifiedKFold
from utils.truncate_dataset import truncate_new_dataset
from utils.truncated_dataset_v3 import truncate_new_dataset_axis
from utils.combine_bg import combine_bg
def eval_freq_schedule(args, epoch: int):
    if epoch >= args.max_epochs * 0.95:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.9:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.8:
        args.eval_freq = 2

def set_environment(args, tlogger,is_train_aug=False,fold_img_label=-1):
    """
    create data loader, optimizer, schedule, scaler, amp_context, start_epoch
    """
    
    print("Setting Environment...")

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    ### = = = =  Dataset and Data Loader = = = =  
    tlogger.print("Building Dataloader....")
    
    train_loader, val_loader = build_loader(args,is_train_aug,fold_img_label)

    tmp_list = os.listdir(args.train_root)
    machine_type = tmp_list[0].split('-')[0]
    
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
    tlogger.print()
    class2num = get_class2num(args.train_root)
    args.num_classes = len(class2num)
    ### = = = =  Model = = = =
    tlogger.print("Building Model....")
    model = MODEL_GETTER[args.backbone_model_name](
        use_fpn = args.use_fpn,
        fpn_size = args.fpn_size,
        use_selection = args.use_selection,
        num_classes = args.num_classes,
        num_selects = args.num_selects,
        use_combiner = args.use_combiner,
    ) # about return_nodes, we use our default setting
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
        # model.load_state_dict(checkpoint['model'])
        # start_epoch = checkpoint['epoch']
        pretrained_dict = checkpoint['model_state_dict']
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                    k in model_dict and v.size() == model_dict[k].size()}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        start_epoch = 0
    else:
        start_epoch = 0

    model.to(args.device)
    tlogger.print()

    if train_loader is None:
        return train_loader, val_loader, model, None, None, None, None

    ### = = = =  Optimizer = = = =
    tlogger.print("Building Optimizer....")
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, nesterov=True, momentum=0.9, weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)

    # if args.pretrained is not None:
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    tlogger.print()

    schedule = cosine_decay(args, len(train_loader))

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    return train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch



def GCE_Loss(logits, target, k=8):

    batch_size, num_classes = logits.size()

    # 計算真實類別的logits
    logits_true = logits[range(batch_size), target]

    # 構建所有負類別的集合
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[range(batch_size), target] = False

    # 取出所有負類別的logits
    logits_neg = logits[mask].view(batch_size, num_classes - 1)

    # 找到前k高置信度的負類別的logits
    topk_logits_neg, _ = torch.topk(logits_neg, k, dim=1)

    # 計算GCE損失
    loss = -logits_true + torch.logsumexp(torch.cat([logits_true.unsqueeze(1), topk_logits_neg], dim=1), dim=1)
    
    return loss.mean()


def train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader):
    """
    training process
    """
    loss_function = F.cross_entropy
    # loss_function = GCE_Loss
    optimizer.zero_grad()
    total_batchs = len(train_loader)  # just for log
    show_progress = [x / 10 for x in range(11)]  # just for log
    progress_i = 0

    # temperature = 2 ** (epoch // 10 - 1)
    temperature = 0.5 ** (epoch // 10) * args.temperature
    # temperature = args.temperature

    n_left_batchs = len(train_loader) % args.update_freq

    for batch_id, (ids, datas, labels) in enumerate(train_loader):
        model.train()
        """ = = = = adjust learning rate = = = = """
        iterations = epoch * len(train_loader) + batch_id
        adjust_lr(iterations, optimizer, schedule)

        # temperature = (args.temperature - 1) * (get_lr(optimizer) / args.max_lr) + 1

        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        datas, labels = datas.to(args.device), labels.to(args.device)

        with amp_context():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'

            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            outs = model(datas)
            out = {}
            for k, v in outs.items():
                tmp = v.clone()
                out[k] = tmp
            loss = 0.
            for name in outs:

                if "FPN1_" in name:
                    if args.lambda_b0 != 0:
                        aux_name = name.replace("FPN1_", "")
                        gt_score_map = outs[aux_name].detach()
                        thres = torch.Tensor(model.selector.thresholds[aux_name])
                        gt_score_map = suppression(gt_score_map, thres, temperature)
                        logit = F.log_softmax(outs[name] / temperature, dim=-1)
                        loss_b0 = nn.KLDivLoss()(logit, gt_score_map)
                        loss += args.lambda_b0 * loss_b0
                    else:
                        loss_b0 = 0.0

                elif "select_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")
                    if args.lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        loss_s = loss_function(logit,labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += args.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")

                    if args.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, args.num_classes]) - 1
                        labels_0 = labels_0.to(args.device)
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += args.lambda_n * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not args.use_fpn:
                        raise ValueError("FPN not use here.")
                    if args.lambda_b != 0:
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = loss_function(outs[name].mean(1), labels)
                        loss += args.lambda_b * loss_b
                    else:
                        loss_b = 0.0

                elif "comb_outs" in name:
                    if not args.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if args.lambda_c != 0:
                        loss_c = loss_function(outs[name], labels)
                        loss += args.lambda_c * loss_c

                elif "ori_out" in name:
                    loss_ori = loss_function(outs[name], labels)
                    loss += loss_ori

            if batch_id < len(train_loader) - n_left_batchs:
                loss /= args.update_freq
            else:
                loss /= n_left_batchs

        """ = = = = calculate gradient = = = = """
        loss=loss.to(args.device)
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        if (batch_id + 1) % args.update_freq == 0 or (batch_id + 1) == len(train_loader):
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()  # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()

        """ log (MISC) """
        if args.use_wandb and ((batch_id + 1) % args.log_freq == 0):
            model.eval()
            msg = {}
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(optimizer)
            cal_train_metrics(args, msg, out, labels, batch_size, model.selector.thresholds, loss_function)
            wandb.log(msg)

        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(".." + str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1

def start_train(args, tlogger,fold_img_label=None,idx_fold=''):
    """
    train model and save best model
    """
    is_train_aug=args.is_train_aug
    train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch = set_environment(args, tlogger,is_train_aug,fold_img_label)
    args.model_type='HERBS'
    best_acc = 0.0
    best_eval_name = "null"
    try:
        length_dict= get_length_dict(os.path.join(args.train_root,'log.txt'))
    except:
        length_dict=None

    if args.use_wandb:
        wandb.init(entity=args.wandb_entity,
                   project=args.project_name,
                   name=args.exp_name,
                   config=args)
        wandb.run.summary["best_acc"] = best_acc
        wandb.run.summary["best_eval_name"] = best_eval_name
        wandb.run.summary["best_epoch"] = 0

    for epoch in range(start_epoch, args.max_epochs):

        """
        Train
        """
        if train_loader is not None:
            tlogger.print("Start Training {} Epoch".format(epoch+1))
            train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader)
            tlogger.print()
        else:
            from eval import eval_and_save
            eval_and_save(args, model, val_loader)
            break

        #eval_freq_schedule(args, epoch)

        if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
            """
            Evaluation
            """
            acc = -1
            if val_loader is not None:
                tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
                acc, eval_name, accs = evaluate(args, model, val_loader)
                tlogger.print("....BEST_ACC: {}% ({}%)".format(max(acc, best_acc), acc))
                tlogger.print()

            if args.use_wandb:
                loss_function = F.cross_entropy
                # loss_function = GCE_Loss
                msg = {}
                msg['info/epoch'] = epoch + 1
                msg['info/lr'] = get_lr(optimizer)
                cal_evalute_metrics(args, msg, val_loader, args.batch_size, model.selector.thresholds, model,loss_function)
                msg["val_acc/acc"] = acc
                wandb.log(msg)

            if acc >= best_acc:
                best_acc = acc
                best_eval_name = eval_name
                save_dict = {
                    'backbone_model_name': args.backbone_model_name,
                    'model_name': args.model_type,
                    'use_fpn': args.use_fpn,
                    'fpn_size': args.fpn_size,
                    'use_selection': args.use_selection,
                    'num_class': args.num_classes,
                    'num_selects': args.num_selects,
                    'use_combiner': args.use_combiner,
                    'model_state_dict': model.state_dict(),
                    "epoch": epoch,
                    'img_size': args.data_size,
                    'class2num': get_class2num(args.train_root),
                    'real_length_dict': length_dict,
                }
                torch.save(save_dict, os.path.join(args.save_dir, "save_model", f"{args.machine_type}-best.pth"))
            if args.use_wandb:
                wandb.run.summary["best_acc"] = best_acc
                wandb.run.summary["best_eval_name"] = best_eval_name
                wandb.run.summary["best_epoch"] = epoch + 1
    del train_loader
    del val_loader
    if args.use_wandb:
        wandb.finish()

def run_HERBS_train(HERBS_args):
    """
    truncate train dataset,
    train combine_bg,
    cross validation,
    run HERBS　train process
    """

    start_time = time.time()
    tlogger = timeLogger()
    tlogger.print("Reading Config...")
    tlogger.print()
    # set gpu
    if HERBS_args.which_gpu == 'cuda:0':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    elif HERBS_args.which_gpu == 'cuda:1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if HERBS_args.is_truncate_train_dataset:
        while HERBS_args.train_root[-1] == '/': HERBS_args.train_root = HERBS_args.train_root[:-1]
        save_path = os.path.join(HERBS_args.train_root.split(os.path.split(HERBS_args.train_root)[-1])[0],
                                 'train_truncated')
        truncate_new_dataset(HERBS_args.train_root, save_path, HERBS_args.div_num)
        # truncate_new_dataset_axis(HERBS_args.train_root, save_path, HERBS_args.div_num)
        HERBS_args.train_root = save_path

    if HERBS_args.is_truncate_train_dataset_partial:
        while HERBS_args.train_root[-1] == '/': HERBS_args.train_root = HERBS_args.train_root[:-1]
        save_path = os.path.join(HERBS_args.train_root.split(os.path.split(HERBS_args.train_root)[-1])[0],
                                 'train_truncated')
        truncate_new_dataset(HERBS_args.train_partial_root, save_path, HERBS_args.partial_div_num, True)
        # truncate_new_dataset_axis(HERBS_args.train_partial_root, save_path, HERBS_args.partial_div_num, True)
        HERBS_args.train_root = save_path

    if HERBS_args.train_combine_bg:
        while HERBS_args.train_root[-1] == '/': HERBS_args.train_root = HERBS_args.train_root[:-1]
        combine_bg_save_path = os.path.join(HERBS_args.train_root.split(os.path.split(HERBS_args.train_root)[-1])[0],
                                            'train_combine_bg')
        from data.dataset import get_train_image_list

        class2num = get_class2num(HERBS_args.train_root)
        num_classes = len(class2num)
        print("[dataset] class number:", num_classes)
        original_img_path_list, original_img_classes_list = get_train_image_list(HERBS_args.train_root, class2num)
        combine_bg(HERBS_args,HERBS_args.train_root, combine_bg_save_path, original_img_path_list)
        HERBS_args.train_root = combine_bg_save_path

    if HERBS_args.is_using_cross_validation:
        skf = StratifiedKFold(n_splits=HERBS_args.cross_validation_folds, random_state=32, shuffle=True)
        print(f'total {HERBS_args.cross_validation_folds} fold')
        from data.dataset import get_train_image_list
        class2num = get_class2num(HERBS_args.train_root)
        num_classes = len(class2num)
        print("[dataset] class number:", num_classes)
        original_img_path_list, original_img_classes_list = get_train_image_list(HERBS_args.train_root, class2num)
        print(f'total ori:{len(original_img_path_list)}')
        for i, (train_index, valid_index) in enumerate(skf.split(original_img_path_list, original_img_classes_list)):
            if i >=2:
                print(f"Fold {i}:")
                # print(f"  Train: index={train_index}")
                # print(f"  valid:  index={valid_index}")

                HERBS_args.exp_name = HERBS_args.exp_name.split('_fold_')[0] + '_fold_' + str(i)
                build_record_folder(HERBS_args)
                start_train(HERBS_args, tlogger, (train_index, valid_index), str(i))
                end_time = time.time() - start_time
                print(f'train time:{end_time}')
    else:
        build_record_folder(HERBS_args)
        start_train(HERBS_args, tlogger)
        end_time = time.time() - start_time
        print(f'train time:{end_time}')
        with open(os.path.join(HERBS_args.save_dir, 'log.txt'), 'a') as f:
            f.write(f'{HERBS_args.exp_name} model \n')
            f.write(f'train time:{end_time}')
            f.write('\n')
if __name__ == "__main__":
    pass