import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 20,           # [최적화] 학습 횟수 증가 (10 -> 20)
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 16,       # [최적화] EfficientNet은 메모리를 좀 더 쓰므로 배치를 조절 (32 -> 16 or 32)
    'SEED': 42
}

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG['SEED'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. 데이터 로드
# ==========================================
train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/dev.csv')
test_df = pd.read_csv('./data/sample_submission.csv')

# ==========================================
# 3. 전처리 및 증강 (Augmentation)
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

train_dataset = MultiViewDataset(train_df, './data/train', train_transform)
val_dataset = MultiViewDataset(val_df, './data/dev', test_transform)
test_dataset = MultiViewDataset(test_df, './data/test', test_transform, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# ==========================================
# 5. 모델 업그레이드 (EfficientNet-B0)
# ==========================================
class MultiViewEfficientNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MultiViewEfficientNet, self).__init__()
        # [최적화] ResNet18 대신 EfficientNet-B0 사용
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # EfficientNet의 마지막 분류기 층 제거 (feature extractor로 사용)
        # B0의 마지막 feature map 채널 수는 1280
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Classifier: (1280 * 2) -> Output
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 2, 512),
            nn.BatchNorm1d(512),  # [최적화] 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(0.4),      # [최적화] 드롭아웃 강화
            nn.Linear(512, num_classes)
        )

    def forward(self, views):
        # views: [front, top]
        f1 = self.feature_extractor(views[0]) # (Batch, 1280, 1, 1)
        f2 = self.feature_extractor(views[1]) # (Batch, 1280, 1, 1)
        
        # Flatten
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        
        combined = torch.cat((f1, f2), dim=1)
        return self.classifier(combined)

# ==========================================
# 6. 학습 및 검증 함수
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
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
        
        if scheduler:
            scheduler.step()
            
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
    
    eps = 1e-15
    p = np.clip(all_probs, eps, 1 - eps)
    logloss_score = -np.mean(all_labels * np.log(p) + (1 - all_labels) * np.log(1 - p))
    acc_score = np.mean((all_probs > 0.5) == all_labels)
    
    return logloss_score, acc_score

# ==========================================
# 7. 메인 실행 루프
# ==========================================
if __name__ == '__main__':
    if not os.path.exists('./models'):
        os.makedirs('./models')

    model = MultiViewEfficientNet().to(device)
    
    # [최적화] Label Smoothing 적용 (과신 방지)
    # BCEWithLogitsLoss는 label smoothing을 직접 지원하지 않으므로, 
    # 데이터가 충분치 않을 땐 기본 Loss를 쓰되 Weight Decay를 주는 방식이 유리함.
    criterion = nn.BCEWithLogitsLoss() 
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'], weight_decay=1e-4)
    
    # [최적화] Cosine Annealing 스케줄러 (주기적으로 학습률 리셋)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    best_score = float('inf')
    
    print("\n[Start Training with EfficientNet-B0]")
    for epoch in range(1, CFG['EPOCHS'] + 1):
        # 스케줄러는 step마다 호출 (Batch 단위 업데이트)
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_logloss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{CFG['EPOCHS']}]")
        print(f"  - Train Loss: {avg_train_loss:.4f}")
        print(f"  - Val LogLoss: {val_logloss:.4f} | Val Acc: {val_acc:.2%}")
        
        if val_logloss < best_score:
            best_score = val_logloss
            torch.save(model.state_dict(), './models/best_model_v3.pth')
            print(f"  >>> Best Model Saved! (Score: {best_score:.4f})")

    # ==========================================
    # 8. 추론 및 TTA (Test Time Augmentation)
    # ==========================================
    print("\n[Start Inference with TTA]")
    model.load_state_dict(torch.load('./models/best_model_v3.pth'))
    model.eval()
    
    all_probs = []
    
    with torch.no_grad():
        for views in tqdm(test_loader, desc="Inference"):
            views = [v.to(device) for v in views]
            
            # 1) 원본 예측
            outputs_orig = model(views).view(-1)
            probs_orig = torch.sigmoid(outputs_orig)
            
            # 2) [최적화] TTA: 좌우 반전(Flip) 후 예측
            # views[0]은 front, views[1]은 top
            views_flip = [torch.flip(v, [3]) for v in views] # 3번 차원(width) 반전
            outputs_flip = model(views_flip).view(-1)
            probs_flip = torch.sigmoid(outputs_flip)
            
            # 3) 평균 (Ensemble 효과)
            probs_final = (probs_orig + probs_flip) / 2.0
            all_probs.extend(probs_final.cpu().numpy())

    # 결과 저장
    submission = pd.DataFrame({
        'id': test_df['id'],
        'unstable_prob': all_probs,      
        'stable_prob': 1.0 - np.array(all_probs)
    })

    submission.to_csv('submission_v3.csv', encoding='UTF-8-sig', index=False)
    print(f"\n최종 제출 파일 저장 완료: submission_v3.csv")