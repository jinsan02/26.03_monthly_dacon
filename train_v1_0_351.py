import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm.auto import tqdm

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 10,           # 학습 횟수를 3 -> 10으로 늘림 (증강 적용 시 더 많이 학습해야 함)
    'LEARNING_RATE': 1e-4,  # 학습률을 조금 낮춰서 안정적으로 학습
    'BATCH_SIZE': 32,
    'SEED': 42
}

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG['SEED'])

# GPU 사용 가능 여부 확인 (WSL2 환경이면 cuda 잡힐 것임)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. 데이터 로드
# ==========================================
train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/dev.csv')
test_df = pd.read_csv('./data/sample_submission.csv')

print(f"학습 데이터(Train): {len(train_df)}개 - 고정 환경")
print(f"검증 데이터(Dev): {len(val_df)}개 - 무작위 환경 (중요!)")
print(f"평가 데이터(Test): {len(test_df)}개 - 제출용")

# ==========================================
# 3. 전처리 및 데이터 증강 (핵심 수정 부분)
# ==========================================
# Train: 고정된 환경의 이미지를 비틀고 색을 바꿔서 '무작위 환경'처럼 보이게 만듦
train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(p=0.5),       # 50% 확률로 좌우 반전
    transforms.RandomRotation(15),                # 각도 약간 비틀기
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 밝기와 대비 변화 (조명 변화 시뮬레이션)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test/Val: 실제 평가는 원본 이미지 그대로 해야 함
test_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. 데이터셋 클래스
# ==========================================
class MultiViewDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.label_map = {'stable': 0, 'unstable': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx]['id'])
        folder_path = os.path.join(self.root_dir, sample_id)
        
        views = []
        # front, top 두 시점의 이미지를 가져옴
        for name in ["front", "top"]:
            img_path = os.path.join(folder_path, f"{name}.png")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            views.append(image)
            
        if self.is_test:
            return views
        
        label = self.label_map[self.df.iloc[idx]['label']]
        return views, label

# 데이터로더 생성
train_dataset = MultiViewDataset(train_df, './data/train', train_transform, is_test=False)
val_dataset = MultiViewDataset(val_df, './data/dev', test_transform, is_test=False)
test_dataset = MultiViewDataset(test_df, './data/test', test_transform, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# ==========================================
# 5. 모델 정의
# ==========================================
class MultiViewResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MultiViewResNet, self).__init__()
        # 사전 학습된 ResNet18 사용
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 두 이미지의 특징(512+512)을 합쳐서 판단
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # 과적합 방지 위해 드롭아웃 상향
            nn.Linear(256, num_classes)
        )

    def forward(self, views):
        f1 = self.feature_extractor(views[0]).view(views[0].size(0), -1)
        f2 = self.feature_extractor(views[1]).view(views[1].size(0), -1)
        combined = torch.cat((f1, f2), dim=1)
        return self.classifier(combined)

# ==========================================
# 6. 학습 및 검증 함수
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for views, labels in tqdm(loader, desc="Training"):
        views = [v.to(device) for v in views]
        labels = labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(views).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for views, labels in tqdm(loader, desc="Validation"):
            views = [v.to(device) for v in views]
            labels = labels.to(device).float()
            
            outputs = model(views).view(-1)
            probs = torch.sigmoid(outputs)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # LogLoss 계산 (대회 평가 기준)
    eps = 1e-15
    p = np.clip(all_probs, eps, 1 - eps)
    logloss_score = -np.mean(all_labels * np.log(p) + (1 - all_labels) * np.log(1 - p))
    
    # 정확도(Accuracy) 계산
    preds = (all_probs > 0.5).astype(int)
    acc_score = np.mean(preds == all_labels)
    
    return logloss_score, acc_score

# ==========================================
# 7. 메인 실행 루프 (Best Model 저장 추가)
# ==========================================
if __name__ == '__main__':
    # 모델 저장할 폴더 만들기
    if not os.path.exists('./models'):
        os.makedirs('./models')

    model = MultiViewResNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])

    best_score = float('inf') # LogLoss는 낮을수록 좋음
    best_acc = 0

    print("\n[Start Training]")
    for epoch in range(1, CFG['EPOCHS'] + 1):
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_logloss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{CFG['EPOCHS']}]")
        print(f"  - Train Loss: {avg_train_loss:.4f}")
        print(f"  - Val LogLoss: {val_logloss:.4f} (낮을수록 좋음)")
        print(f"  - Val Accuracy: {val_acc:.2%} (높을수록 좋음)")
        
        # 모델 저장 로직: LogLoss가 더 낮아지면 저장 (가장 똑똑한 모델 기억하기)
        if val_logloss < best_score:
            best_score = val_logloss
            best_acc = val_acc
            torch.save(model.state_dict(), './models/best_model.pth')
            print(f"  >>> Best Model Saved! (Score: {best_score:.4f})")

    # --- 추론 (Inference) ---
    print("\n[Start Inference]")
    
    # 학습이 끝난 후, 가장 성능이 좋았던 모델을 다시 불러옴
    model.load_state_dict(torch.load('./models/best_model.pth'))
    model.eval()
    
    all_probs = []
    with torch.no_grad():
        for views in tqdm(test_loader, desc="Inference"):
            views = [v.to(device) for v in views]
            outputs = model(views).view(-1)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())

    # 결과 저장
    all_probs = np.array(all_probs)
    submission = pd.DataFrame({
        'id': test_df['id'],
        'unstable_prob': all_probs,      # 불안정 확률
        'stable_prob': 1.0 - all_probs   # 안정 확률
    })

    submission.to_csv('submission.csv', encoding='UTF-8-sig', index=False)
    print(f"\n최종 제출 파일 저장 완료: submission.csv")
    print(f"Dev 셋 기준 최고 정확도: {best_acc:.2%}")