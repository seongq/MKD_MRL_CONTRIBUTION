import os
import itertools

from tqdm import tqdm
import wandb
import os, re, datetime
import json
import numpy as np, argparse, time, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss,  Model,  FocalLoss, Loss_Function
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
import torch.nn.functional as F
from tqdm import tqdm



# seed = 1475 # We use seed = 1475 on IEMOCAP and seed = 67137 on MELD
def seed_everything(args):
    if random.choice([True,False]):
        if args.Dataset=="IEMOCAP":
            seed = 1475
        elif args.Dataset=="MELD":
            seed = 67137
    else:
        seed = random.choice(range(1,1000000))
        
    args.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])



def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('/workspace/datasets/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset( train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader
    

def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda,optimizer=None, train=False, dataset='IEMOCAP',args=None):
    losses, preds, labels = [], [], []
    vids = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    it = tqdm(dataloader, total=len(dataloader), desc=f"Train {epoch+1}") if train else dataloader
    for data in it:
        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-2]] if cuda else data[:-2]
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
    
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        # print(label)
        if train:
            optimizer.zero_grad()
            logits, logits_uni_modal_student, logits_MKD_teacher, logits_MRL_feature, logits_modal, logits_uni_modal =  model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
            loss = 0

            loss_classification = loss_function(logits,label)
            loss += loss_classification
            if args.MKD:
                loss_MKD_teacher_classification = 0
                loss_MKD_student_classification = 0 
                loss_MKD_distillation = 0
                loss_MKD_lastlayer_classification = 0 
                loss_MKD_lastlayer_distillation = 0

                for key in logits_MKD_teacher.keys():
                    loss_MKD_teacher_classification += loss_function(logits_MKD_teacher[key], label)
                    
                for key in logits_uni_modal_student.keys():
                    loss_MKD_student_classification += loss_function(logits_uni_modal_student[key], label)
                    if args.MKD_last_layer:
                        loss_MKD_lastlayer_classification +=loss_function(logits_uni_modal[key], label)
                for key in logits_MKD_teacher.keys():

                    loss_MKD_distillation += F.kl_div(F.log_softmax(logits_uni_modal_student[key] , dim=1), F.softmax(logits_MKD_teacher[key].detach() , dim=1), reduction="batchmean")
                    if args.MKD_last_layer:
                        loss_MKD_lastlayer_distillation += F.kl_div(F.log_softmax(logits_uni_modal[key] , dim=1), F.softmax(logits_MKD_teacher[key].detach() , dim=1), reduction="batchmean")

                loss += loss_MKD_teacher_classification*args.MKD_teacher_classification_coeff
                loss += loss_MKD_student_classification*args.MKD_student_classification_coeff
                loss += loss_MKD_distillation*args.MKD_coeff
                loss += loss_MKD_lastlayer_classification*args.MKD_lastlayer_classification_coeff
                loss += loss_MKD_lastlayer_distillation*args.MKD_lastlayer_distillation_coeff
            
            if args.MRL:
                loss_MRL = 0 
                for key in logits_MRL_feature.keys():
                    loss_MRL += loss_function(logits_MRL_feature[key], label)

                loss += loss_MRL*args.MRL_coeff

            if args.calib:
                loss_calib_classification = 0
                loss_calib = 0
                for key in logits_modal.keys():
                    loss_calib_classification += loss_function(logits_modal[key], label)
                key_pairs = list(itertools.combinations(list(logits_modal.keys()),2))
                # print(key_pairs)

                for key in logits_modal.keys():
                    loss_calib += F.relu( F.softmax(logits_modal[key], dim=-1).max(dim=-1).values- F.softmax(logits,dim=-1).max(dim=-1).values)
                
                for key_a, key_b in key_pairs:
                    if (key_a in key_b) or (key_b in key_a):
                        if key_a in key_b:
                            more_modalities = key_b
                            less_modalities = key_a
                        elif key_b in key_a:
                            more_modalities = key_a
                            less_modalities = key_b
                        loss_calib += F.relu( F.softmax(logits_modal[less_modalities],dim=-1).max(dim=-1).values-F.softmax(logits_modal[more_modalities], dim=-1).max(dim=-1).values)
                loss_calib = loss_calib.mean()
                    
                loss+=loss_calib_classification*args.CALIB_classification_coeff
                loss+=loss_calib*args.CALIB_coeff
            loss.backward()
            optimizer.step()
            
        else:
            logits, _, _, _ ,_,_=  model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
            loss = loss_function(logits,label)
        
        preds.append(torch.argmax(logits, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan')

    vids += data[-1]
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids


if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument("--use_speaker_embedding", action="store_true", default=False)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00005, metavar='L2', help='L2 regularization weight')
    
    
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=300, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    
    parser.add_argument("--loss_cls", choices=("Focal", "NLL"))
    parser.add_argument('--stablizing', action="store_true")

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--testing', action='store_true', default=False, help='testing')
   
    parser.add_argument('--focal_prob', default='log_prob', choices=('prob', 'log_prob'), help='use probability or log_probability in focal loss')
    parser.add_argument("--MKD", action="store_true", default=False)
    parser.add_argument("--MRL", action="store_true", default=False)
    parser.add_argument("--MRL_efficient", action="store_true", default=False)
    parser.add_argument("--mrl_num_partition", type=int, default=3)
    parser.add_argument("--loss_type", required=True, choices=("Focal", "NLL"))
    parser.add_argument("--calib", action="store_true", default=False)
    
    parser.add_argument("--MKD_coeff",type=float, default=1)
    parser.add_argument("--MKD_teacher_classification_coeff",type=float, default=1)
    parser.add_argument("--MKD_student_classification_coeff", type=float,default=1)
    parser.add_argument("--MKD_last_layer", default=False, action="store_true")
    parser.add_argument("--MKD_lastlayer_classification_coeff", default= 1, type=float)
    parser.add_argument("--MKD_lastlayer_distillation_coeff", default=1, type=float)
    
    parser.add_argument("--MRL_coeff",type=float, default=1)
    parser.add_argument("--CALIB_coeff", default=1,type=float)
    parser.add_argument("--CALIB_classification_coeff", default=1,type=float)

    
    parser.add_argument("--using_MHA", default=False, action="store_true")
    parser.add_argument("--number_of_heads", default=2, type=int, choices=(1,2,4,8,16))
    
    parser.add_argument("--using_graph", default=False, action="store_true")
    parser.add_argument("--using_multimodal_graph", default=False, action="store_true")
    parser.add_argument("--num_K", default=4, type=int)
    parser.add_argument("--graph_hidden_dim", default=512, type=int)

    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(args)


    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        device = torch.device("cuda:0")
        print('Running on GPU')
    else:
        device = torch.device("cpu")
        print('Running on CPU')

  

    cuda       = args.cuda
    n_epochs   = args.epochs
   
    batch_size = args.batch_size
    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024 #feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']


    D_g = 1024 

    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes  = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1



    seed_everything(args)
    run_name = f"{timestamp}_{args.Dataset}"
    
    # wandb.init(
    #     project="testtesttest",   # ← 고정
    #     name=run_name,
    #     config=vars(args)
    # )

    
    
    wandb.init(
        project="CKPT_20250926_MKD_MRL_CALIB_MULTIMODAL_GRAPH",   # ← 고정
        name=run_name,
        config=vars(args)
    )
    run   = wandb.run
    wb_id = run.id if run is not None else "noWB"
    wb_nm = (run.name or wb_id) if run is not None else "noWB"
    wb_ver = wandb.__version__
    model = Model(
        MKD = args.MKD,
        use_speaker_embedding=args.use_speaker_embedding,
                
                n_speakers=n_speakers,
                n_classes=n_classes,
                dropout=args.dropout,
                no_cuda=args.no_cuda,
                D_m_a = D_audio,
                D_m_v = D_visual,
                D_m_l = D_text,
                hidden_dim = D_g,
                dataset=args.Dataset,
                MRL = args.MRL,
                MRL_efficient = args.MRL_efficient,
                mrl_num_partition = args.mrl_num_partition,
                calib = args.calib,
                MKD_last_layer = args.MKD_last_layer,
                using_MHA = args.using_MHA,
                number_of_heads = args.number_of_heads,
                using_graph= args.using_graph,
                using_multimodal_graph = args.using_multimodal_graph,
                num_K = args.num_K,
                graph_hidden_dim = args.graph_hidden_dim,
                args = args)


    if cuda:
        model.to(device)

    loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])

    loss_function = Loss_Function(args.class_weight, loss_weights.to(device) if cuda else loss_weights, args.loss_type, args.Dataset, args.focal_prob)

        

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    lr = args.lr
    
    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=4)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=4)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    best_acc = None
    all_fscore, all_acc, all_loss = [], [], []

    import os, re, datetime

 
    
    setting_dir = str(timestamp) + "_"+ args.Dataset 

    save_root = os.path.join("CKPT_20250925_MKD_MRL_CALIB", setting_dir)              # ← setting 요약이 폴더명에 들어감
    os.makedirs(save_root, exist_ok=True)
    
    args_with_wb = {**vars(args), "wandb_id": wb_id, "wandb_name": wb_nm, "wandb_version": wb_ver}
    with open(os.path.join(save_root, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args_with_wb, f, indent=2, ensure_ascii=False, sort_keys=True)

    # metrics.jsonl 준비 (에폭별 기록) ← NEW
    metrics_path = os.path.join(save_root, "metrics.jsonl")
    open(metrics_path, "w", encoding="utf-8").close()  # 비우고 시작
    
    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, train_fscore, _ = train_or_eval_graph_model(model, loss_function, train_loader, e, cuda,  \
                                                                                 optimizer, True, dataset=args.Dataset,args=args)
        valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_graph_model(model, loss_function, valid_loader, e, cuda,  \
                                                                              dataset=args.Dataset,args=args)
        test_loss, test_acc, test_label, test_pred, test_fscore, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda,  \
                                                                                               dataset=args.Dataset,args=args)
        all_fscore.append(test_fscore)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred = test_loss, test_label, test_pred

        if best_fscore is None or test_fscore > best_fscore:
            best_fscore = test_fscore
            best_acc = test_acc
            best_label, best_pred = test_label, test_pred

            # 1) 기존 체크포인트(.pt) 싹 지우기
            for fname in os.listdir(save_root):
                if fname.endswith(".pt"):
                    fpath = os.path.join(save_root, fname)
                    try:
                        os.remove(fpath)
                    except OSError as e:
                        print(f"[WARN] remove fail: {fpath} ({e})")

            # 2) 새 모델 저장
            ckpt_name = f"{args.Dataset}_bestF1-{best_fscore:.2f}_bestAcc-{best_acc:.2f}.pt"
            ckpt_path = os.path.join(save_root, ckpt_name)

            torch.save({
                "epoch": e + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_f1": best_fscore,
                "best_acc": best_acc,
                "test_f1": test_fscore,
                "test_acc": test_acc,
                "args": vars(args),
            }, ckpt_path)

                        # ==== Save BEST-only artifacts: confusion matrix + per-class ACC/F1 ====
            labels_list = list(range(n_classes))

            cm = confusion_matrix(test_label, test_pred, labels=labels_list)
            support = cm.sum(axis=1)
            correct = np.diag(cm)
            per_class_acc = np.divide(
                correct, support,
                out=np.zeros_like(correct, dtype=float),
                where=support != 0
            )
            per_class_f1 = f1_score(
                test_label, test_pred,
                labels=labels_list, average=None
            )

            # (1) 로컬 저장: JSON + NPY + CSV
            best_per_class_path = os.path.join(save_root, "best_per_class_metrics.json")
            with open(best_per_class_path, "w", encoding="utf-8") as f:
                json.dump({
                    "epoch": int(e+1),
                    "best_f1_overall": float(best_fscore),
                    "best_acc_overall": float(best_acc),
                    "per_class": {
                        str(i): {
                            "support": int(support[i]),
                            "acc": float(per_class_acc[i]),
                            "f1": float(per_class_f1[i]),
                        }
                        for i in labels_list
                    }
                }, f, ensure_ascii=False, indent=2)

            # 혼동행렬: npy/csv로도 저장
            np.save(os.path.join(save_root, "best_confusion_matrix.npy"), cm)
            np.savetxt(os.path.join(save_root, "best_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

            # (2) W&B 로깅: best/* 네임스페이스로 올리기
            wandb.log({
                "best/epoch": int(e+1),
                "best/test_f1": float(best_fscore),
                "best/test_acc": float(best_acc),
                **{f"best/per_class_f1/class_{i}": float(per_class_f1[i]) for i in labels_list},
                **{f"best/per_class_acc/class_{i}": float(per_class_acc[i]) for i in labels_list},
                # 혼동행렬도 best로 기록
                "best/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=test_label,
                    preds=test_pred,
                    class_names=[str(i) for i in range(n_classes)]
                ),
            }, step=e+1)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
    
        print ('----------best F-Score:', max(all_fscore))
        if e % 10 == 0:
            print('Best performance so far..')
            print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
                # ===== Per-class metrics (ACC / F1 / support) =====
        labels_list = list(range(n_classes))

        cm = confusion_matrix(test_label, test_pred, labels=labels_list)
        support = cm.sum(axis=1)                  # 클래스별 샘플 수
        correct = np.diag(cm)                     # 클래스별 정분류 수
        per_class_acc = np.divide(
            correct, support,
            out=np.zeros_like(correct, dtype=float),
            where=support != 0
        )
        per_class_f1 = f1_score(
            test_label, test_pred,
            labels=labels_list, average=None
        )

        # (A) W&B 테이블 스냅샷
        pc_table = wandb.Table(columns=["epoch", "class", "support", "acc", "f1"])
        for i in labels_list:
            pc_table.add_data(int(e+1), int(i), int(support[i]),
                              float(per_class_acc[i]), float(per_class_f1[i]))
        wandb.log({"per_class/metrics_table": pc_table}, step=e+1)

        # (B) W&B 스칼라(라인차트)
        scalar_log = {}
        for i in labels_list:
            scalar_log[f"per_class/acc/class_{i}"] = float(per_class_acc[i])
            scalar_log[f"per_class/f1/class_{i}"]  = float(per_class_f1[i])
        wandb.log(scalar_log, step=e+1)

        # (C) 로컬 JSONL 저장
        per_class_path = os.path.join(save_root, "per_class_metrics.jsonl")
        with open(per_class_path, "a", encoding="utf-8") as f:
            record = {
                "epoch": int(e+1),
                "per_class": {
                    str(i): {
                        "support": int(support[i]),
                        "acc": float(per_class_acc[i]),
                        "f1": float(per_class_f1[i]),
                    }
                    for i in labels_list
                }
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

        wandb.log({
            'epoch': e+1,
            'train/loss': train_loss, 'train/acc': train_acc, 'train/f1': train_fscore,
            'test/loss':  test_loss,  'test/acc':  test_acc,  'test/f1':  test_fscore,
            'best/test_f1': best_fscore if best_fscore is not None else test_fscore,
            'best/test_acc': best_acc if best_acc is not None else test_acc,
            'time_sec': round(time.time()-start_time, 2),
        }, step=e+1)
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_label,
                preds=test_pred,
                class_names=[str(i) for i in range(n_classes)]
            )
        }, step=e+1)
        with open(metrics_path, "a", encoding="utf-8") as f:
            json.dump({"epoch": int(e+1), "test_acc": float(test_acc), "test_f1": float(test_fscore)}, f, ensure_ascii=False)
            f.write("\n")

    if not args.testing:
        print('Test performance..')
        print ('F-Score:', max(all_fscore))

        print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
        print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))


wandb.finish()
