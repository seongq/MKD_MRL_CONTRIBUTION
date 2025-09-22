import os
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
from model import MaskedNLLLoss,  Model,  FocalLoss
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
        if train:
            optimizer.zero_grad()
        
        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-2]] if cuda else data[:-2]
        

        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
    
        logits,_ = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        if args.loss_cls == "Focal":
            loss_classification = loss_function(logits, label)
        elif args.loss_cls == "NLL":
            loss_classification = loss_function(F.log_softmax(logits, 1), label)
        loss =  loss_classification
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if args.MKD and train:
            logits_MKD = model.MKD_teacher_forward([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
            loss = 0
            if args.loss_cls == "Focal":
                loss += loss_function(logits_MKD['a'], label)
                loss += loss_function(logits_MKD['v'], label)
                loss += loss_function(logits_MKD['l'], label)
            elif args.loss_cls == "NLL":
                loss += loss_function(F.log_softmax(logits_MKD['a'], 1), label)
                loss += loss_function(F.log_softmax(logits_MKD['v'], 1), label)
                loss += loss_function(F.log_softmax(logits_MKD['l'], 1), label)
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                logits_MKD = model.MKD_teacher_forward([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
            logits, logits_uni_modal = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
            loss = 0
            if args.loss_cls == "Focal":
                loss += loss_function(logits_uni_modal['a'], label)
                loss += loss_function(logits_uni_modal['v'], label)
                loss += loss_function(logits_uni_modal['l'], label)
                loss += loss_function(logits, label)
            elif args.loss_cls == "NLL":
                loss += loss_function(F.log_softmax(logits_uni_modal['a'], 1), label)
                loss += loss_function(F.log_softmax(logits_uni_modal['v'], 1), label)
                loss += loss_function(F.log_softmax(logits_uni_modal['l'], 1), label)
                loss += loss_function(F.log_softmax(logits, 1), label)
                
            loss += F.kl_div(F.log_softmax(logits_uni_modal['a'],1), F.softmax(logits_MKD['a'],1), reduction='batchmean')
            loss += F.kl_div(F.log_softmax(logits_uni_modal['v'],1), F.softmax(logits_MKD['v'],1), reduction='batchmean')
            loss += F.kl_div(F.log_softmax(logits_uni_modal['l'],1), F.softmax(logits_MKD['l'],1), reduction='batchmean')
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
        
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
    wandb.init(
        project="MGLRA_END_ETRIGAJA_202509221540",   # ← 고정
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
                args = args)


    if cuda:
        model.to(device)

    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])

    if args.Dataset == 'MELD':
        if args.loss_cls == "Focal":
            loss_function = FocalLoss(focal_prob=args.focal_prob,args=args)
        elif args.loss_cls == "NLL":
            loss_function = nn.NLLLoss()
    elif args.Dataset == "IEMOCAP":
        if args.loss_cls == "Focal":
            loss_function  = FocalLoss(focal_prob=args.focal_prob,args=args)
        elif args.loss_cls == "NLL":
            if args.class_weight:
                loss_function  = nn.NLLLoss(loss_weights.to(device) if cuda else loss_weights)
            else:
                loss_function  = nn.NLLLoss()
        

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

    save_root = os.path.join("CKPT", setting_dir)              # ← setting 요약이 폴더명에 들어감
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
