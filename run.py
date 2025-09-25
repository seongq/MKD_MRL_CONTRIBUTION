# runner.py
import os
import random
import subprocess
import datetime
import time

EPOCHS       = [50,70,90,80,110,60,100,150,200,300,500,1000]
BATCH_SIZES  = [1,2,4,8,16,32,64]
LRS          = [1e-5, 1e-4, 1e-3]
DROPOUTS     = [0.1,0.2,0.3,0.5]
DATASETS     = ["MELD"]
LOSSES       = ["Focal", "NLL"]
FOCAL_PROB   = ["prob", "log_prob"]
SLEEP_SECS   = 10
NUM_K_CHOICES = [1,2,3,4,5,6,7,8,9,10]
i = 0
while True:
    try:
        i += 1
        ts         = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ds         = random.choice(DATASETS)
        bs         = random.choice(BATCH_SIZES)
        lr         = random.choice(LRS)
        dropout    = random.choice(DROPOUTS)
        loss_type  = random.choice(LOSSES)
        epoch      = random.choice(EPOCHS)
        focal_prob = random.choice(FOCAL_PROB)

        cmd = [
            "python", "train.py",
            "--Dataset", ds,
            "--batch-size", str(bs),
            "--lr", str(lr),
            "--dropout", str(dropout),
            "--epochs", str(epoch),
            "--focal_prob", focal_prob,
            "--loss_type", loss_type,   # 필수 인자
        ]

        # 스위치 옵션
        if random.choice([True, False]): cmd.append("--class-weight")
        if random.choice([True, False]): cmd.append("--stablizing")
        if random.choice([True, False]): cmd.append("--use_speaker_embedding")

        # MKD 옵션 (켜지면 관련 계수도 랜덤)
        if random.choice([True, False]):
            cmd.append("--MKD")
            if random.choice([True,False]):
                cmd.append("--MKD_last_layer")
         
        if random.choice([True, False]):
            cmd.append("--using_MHA")
            cmd += ["--number_of_heads", str(random.choice([1,2,4,8,16]))]
        # MRL 옵션 (켜지면 efficient/partition/계수 랜덤)
        if random.choice([True, False]):
            cmd.append("--MRL")
            if random.choice([True, False]):
                cmd.append("--MRL_efficient")
            cmd += ["--mrl_num_partition", str(random.randint(1,11))]

        # CALIB 옵션 (켜지면 계수 랜덤)
        if random.choice([True, False]):
            cmd.append("--calib")

        if random.choice([True, False]):
            cmd.append("--using_graph")
        
            cmd += ["--num_K", str(random.choice(NUM_K_CHOICES))]
            # hidden_dim(=1024)과 맞추려면 2*graph_hidden_dim == hidden_dim => 512 고정
            
        
        # GPU 선택
        env = os.environ.copy()
        gpu_choice = random.choice(["0", "1"])
        env["CUDA_VISIBLE_DEVICES"] = gpu_choice  # train.py 내부에선 cuda:0 사용 가정

        print(f"[{i:05d}][{ts}] GPU={gpu_choice} | run: {' '.join(cmd)}", flush=True)
        ret = subprocess.run(cmd, check=False, env=env).returncode
        if ret != 0:
            print(f"[{i:05d}] exit code {ret} (continue)", flush=True)

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
        break
    except Exception as e:
        print(f"[{i:05d}] error: {e} (continue)", flush=True)

    time.sleep(SLEEP_SECS)



# # runner.py
# import os
# import random
# import subprocess
# import datetime
# import time
# EPOCHS=[50,70,90,80,110,60, 100,150,200,300,500,1000]
# BATCH_SIZES = [1,2,4,8,16,32,64]
# LRS = [0.0001,0.001,0.00001]
# DROPOUT = [0.1,0.2,0.3]
# DATASETS = ["IEMOCAP","MELD"]
# LOSS = ["Focal", "NLL"]


# FOCAL_PROB = ['prob', 'log_prob']
# SLEEP_SECS = 10

# i = 0

# while True:
#     try:
#         i += 1
#         bs = random.choice(BATCH_SIZES)
#         lr = random.choice(LRS)
#         ds = random.choice(DATASETS)
#         dropout = random.choice(DROPOUT)
#         loss = random.choice(LOSS)
#         ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         epoch = random.choice(EPOCHS)
#         focal_prob = random.choice(FOCAL_PROB)
#         cmd = [
#             "python", "train.py",
#             "--Dataset", ds,
#             "--batch-size", str(bs),
#             "--lr", str(lr),

#             "--loss_cls", loss,
#             "--dropout", str(dropout),

           
#             "--epochs", str(epoch),
#             "--focal_prob", focal_prob,
#         ]

#         # 여기서 GPU 0 또는 1을 랜덤 선택
#         if random.choice([True, False]):
#             cmd.append("--class-weight")
#         if random.choice([True,False]):
#             cmd.append("--stablizing")
#         if random.choice([True,False]):
#             cmd.append("--use_speaker_embedding")
#         if random.choice([True,False]):
#             cmd.append("--MKD")

#         gpu_choice = random.choice(["0", "1"])
#         env = os.environ.copy()
#         env["CUDA_VISIBLE_DEVICES"] = gpu_choice  # train.py 내부에선 항상 cuda:0으로 보임

#         print(f"[{i:05d}][{ts}] GPU={gpu_choice} | run: {' '.join(cmd)}", flush=True)

#         ret = subprocess.run(cmd, check=False, env=env).returncode
#         if ret != 0:
#             print(f"[{i:05d}] exit code {ret} (continue)", flush=True)

#     except KeyboardInterrupt:
#         print("\nStopped by user (Ctrl+C).")
#         break
#     except Exception as e:
#         print(f"[{i:05d}] error: {e} (continue)", flush=True)

#     time.sleep(SLEEP_SECS)
