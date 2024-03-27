import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 시드 고정.
random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

model_name = "swsl_resnext50_32x4d"

# 실험 결과 대부분 30 epoch 전에 수렴하기 때문에 최대 epoch 30으로 설정.
epoch_size = 30
batch_size = 48

# 너무 낮으면 학습이 안되고, 너무 높으면 최적의 값으로 수렴하지 못함.
# 실험 결과 최적의 learning rate가 1e-4에서 public 리더보드에서 가장 좋은 성능을 보여
# 해당 learning rate 사용.
learning_rate = 1e-4
early_stop = 5
k_fold_num = 5

import timm
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import f1_score


def train(data_loader):
    model = timm.create_model(model_name, pretrained=True, num_classes=7).to(device=device)

    # 클래스 별로 이미지 수가 다르기 때문에 imbalance 문제를 완화하기 위해 가장 많은 클래스 이미지 수 / 각 클래스 이미지 수로 나눈 값을 가중치로 사용.
    class_num = [329, 205, 235, 134, 151, 245, 399]
    class_weight = torch.tensor(np.max(class_num) / class_num).to(device=device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    # pretrained 모델을 fine-tuning 하므로 feature map을 추출하는 레이어는 learing rate 0.1 비율만 적용.
    # 일반화 성능을 조금 더 높이기 위해 Adam 대신 AdamW 사용.
    # 참고자료 : https://hiddenbeginner.github.io/deeplearning/paperreview/2019/12/29/paper_review_AdamW.html
    feature_extractor = [m for n, m in model.named_parameters() if "fc" not in n]
    classifier = [p for p in model.fc.parameters()]
    params = [
        {"params": feature_extractor, "lr": learning_rate * 0.5},
        {"params": classifier, "lr": learning_rate}
    ]
    optimizer = AdamW(params, lr=learning_rate)

    # ConsineAnnealing Scheduler 적용.
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    result = {
        "train_loss": [],
        "valid_loss": [],
        "valid_acc": [],
        "valid_f1": [],
    }

    train_loader = data_loader["train_loader"]
    valid_loader = data_loader["valid_loader"]

    best_model_state = None
    best_f1 = 0
    early_stop_count = 0

    for epoch_idx in range(1, epoch_size + 1):
        model.train()

        iter_train_loss = []
        iter_valid_loss = []
        iter_valid_acc = []
        iter_valid_f1 = []

        for iter_idx, (train_imgs, train_labels) in enumerate(train_loader, 1):
            train_imgs, train_labels = train_imgs.to(device=device, dtype=torch.float), train_labels.to(device)

            optimizer.zero_grad()

            train_pred = model(train_imgs)
            train_loss = criterion(train_pred, train_labels)
            train_loss.backward()

            optimizer.step()
            iter_train_loss.append(train_loss.cpu().item())

            print(
                f"[Epoch {epoch_idx}/{epoch_size}] model training iteration {iter_idx}/{len(train_loader)}     ",
                end="\r",
            )

        with torch.no_grad():
            for iter_idx, (valid_imgs, valid_labels) in enumerate(valid_loader, 1):
                model.eval()

                valid_imgs, valid_labels = valid_imgs.to(device=device, dtype=torch.float), valid_labels.to(device)

                valid_pred = model(valid_imgs)
                valid_loss = criterion(valid_pred, valid_labels)

                iter_valid_loss.append(valid_loss.cpu().item())

                valid_pred_c = valid_pred.argmax(dim=-1)
                iter_valid_acc.extend((valid_pred_c == valid_labels).cpu().tolist())

                iter_f1_score = f1_score(y_true=valid_labels.cpu().numpy(), y_pred=valid_pred_c.cpu().numpy(),
                                         average="macro")
                iter_valid_f1.append(iter_f1_score)

                print(
                    f"[Epoch {epoch_idx}/{epoch_size}] model validation iteration {iter_idx}/{len(valid_loader)}     ",
                    end="\r"
                )

        epoch_train_loss = np.mean(iter_train_loss)
        epoch_valid_loss = np.mean(iter_valid_loss)
        epoch_valid_acc = np.mean(iter_valid_acc) * 100
        epoch_valid_f1_score = np.mean(iter_valid_f1)

        result["train_loss"].append(epoch_train_loss)
        result["valid_loss"].append(epoch_valid_loss)
        result["valid_acc"].append(epoch_valid_acc)
        result["valid_f1"].append(epoch_valid_f1_score)

        scheduler.step()

        print(
            f"[Epoch {epoch_idx}/{epoch_size}] "
            f"train loss : {epoch_train_loss:.4f} | "
            f"valid loss : {epoch_valid_loss:.4f} | valid acc : {epoch_valid_acc:.2f}% | valid f1 score : {epoch_valid_f1_score:.4f}"
        )

        if epoch_valid_f1_score > best_f1:
            best_f1 = epoch_valid_f1_score
            best_model_state = model.state_dict()
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count == early_stop:
            print("early stoped." + " " * 30)
            break

    return result, best_model_state

