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
    'EPOCHS': 3,
    'LEARNING_RATE': 1e-3,
    'BATCH_SIZE': 32,
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
# 2. 데이터 로드 및 전처리 정의
# ==========================================
# 데이터 경로 설정
train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/dev.csv')
test_df = pd.read_csv('./data/sample_submission.csv')

print(f"학습 데이터 개수: {len(train_df)}")
print(f"검증 데이터 개수: {len(val_df)}")

# 전처리 (베이스라인과 동일 - 증강 없음)
train_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. 데이터셋 클래스 정의
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

# 데이터로더 생성
train_dataset = MultiViewDataset(train_df, './data/train', train_transform, is_test=False)
val_dataset = MultiViewDataset(val_df, './data/dev', test_transform, is_test=False)
test_dataset = MultiViewDataset(test_df, './data/test', test_transform, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)

# ==========================================
# 4. 모델 정의 (MultiViewResNet)
# ==========================================
class MultiViewResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MultiViewResNet, self).__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, views):
        f1 = self.feature_extractor(views[0]).view(views[0].size(0), -1)
        f2 = self.feature_extractor(views[1]).view(views[1].size(0), -1)
        combined = torch.cat((f1, f2), dim=1)
        return self.classifier(combined)

# ==========================================
# 5. 학습 및 검증 함수 (클래스 밖으로 이동)
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
    
    all_probs = np.array(all_probs, dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.float64)
    
    eps = 1e-15
    p = np.clip(all_probs, eps, 1 - eps)
    logloss_score = -np.mean(all_labels * np.log(p) + (1 - all_labels) * np.log(1 - p))
    acc_score = np.mean((all_probs > 0.5) == all_labels)
    
    return logloss_score, acc_score

# ==========================================
# 6. 메인 실행 루프
# ==========================================
if __name__ == '__main__':
    model = MultiViewResNet().to(device)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])

    # --- Training Loop ---
    print("\n[Start Training]")
    for epoch in range(1, CFG['EPOCHS'] + 1):
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_logloss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch}]")
        print(f"  - Train Loss: {avg_train_loss:.4f}")
        print(f"  - Val Log-Loss: {val_logloss:.6f} | Val Acc: {val_acc:.4f}")

    # --- Inference ---
    print("\n[Start Inference]")
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
        'unstable_prob': all_probs,
        'stable_prob': 1.0 - all_probs
    })

    submission.to_csv('submission.csv', encoding='UTF-8-sig', index=False)
    print("submission.csv 저장 완료.")