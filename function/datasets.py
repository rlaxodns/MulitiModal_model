from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
import joblib
import torchvision.transforms as T
from torchvision.transforms import ToPILImage

class MultiModalDataset_PCA(Dataset):
    def __init__(self, video_root, keypoint_root, label_map,
                 num_frames=32, image_size=(224,224),
                 pca_dir=".", pca_fmt="keypoint_{action}pca100.joblib"):
        """
        pca_dir: PCA 파일들이 있는 디렉토리
        pca_fmt: 클래스 이름을 {action} 으로 포맷팅한 PCA 파일명 템플릿
        """
        self.samples = []  # (video_path, keypoint_path, label, action)
        self.label_map = label_map
        self.num_frames = num_frames
        self.transform = T.Compose([
            ToPILImage(),
            T.Resize(image_size),
            T.ToTensor()
        ])

        # 클래스별 PCA 모델 캐시
        self.pca_cache = {}
        self.pca_dir = pca_dir
        self.pca_fmt = pca_fmt

        # 샘플 목록 구성
        for action, label in label_map.items():
            video_folder = os.path.join(video_root, action)
            keypoint_folder = os.path.join(keypoint_root, action)
            if not os.path.isdir(video_folder) or not os.path.isdir(keypoint_folder):
                continue
            for fname in os.listdir(video_folder):
                if fname.endswith('.mp4'):
                    base = os.path.splitext(fname)[0]
                    vp = os.path.join(video_folder, fname)
                    kp = os.path.join(keypoint_folder, f'raw_{base}.csv')
                    if os.path.exists(kp):
                        self.samples.append((vp, kp, label, action))
        print(f"로드된 샘플 수: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, keypoint_path, label, action = self.samples[idx]
        video_tensor = self.load_video_tensor(video_path)  # (3, T, H, W)

        # 1) 균등 샘플링
        kp_full = np.loadtxt(keypoint_path, delimiter=',', dtype=np.float32)  
        idxs = np.linspace(0, kp_full.shape[0] - 1, num=self.num_frames).astype(int)
        kp_sampled = kp_full[idxs, :]  # (num_frames, 225)

        # 2) 클래스별 PCA 로드(한 번만)
        if action not in self.pca_cache:
            pca_file = os.path.join(self.pca_dir, self.pca_fmt.format(action=action))
            if not os.path.exists(pca_file):
                raise FileNotFoundError(f"PCA 파일 없음: {pca_file}")
            self.pca_cache[action] = joblib.load(pca_file)

        pca = self.pca_cache[action]
        kp_reduced = pca.transform(kp_sampled)  # (num_frames, n_components)

        # 3) Tensor 변환
        keypoint_tensor = torch.from_numpy(kp_reduced).float()  # (num_frames, n_components)

        return video_tensor, keypoint_tensor, torch.tensor(label, dtype=torch.long)

    def load_video_tensor(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = torch.linspace(0, total - 1, steps=self.num_frames).long().tolist()

        frames = []
        for i in range(total):
            ret, frame = cap.read()
            if not ret: break
            if i in idxs:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(frame))

        cap.release()

        # 부족한 프레임은 0 패딩
        while len(frames) < self.num_frames:
            frames.append(torch.zeros_like(frames[0]))

        return torch.stack(frames[:self.num_frames], dim=1)  # (3, T, H, W)

from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
import pandas as pd
import random
from pathlib import Path

import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np
import torchvision.transforms as T


class PrecomputedVideoKPDataset(Dataset):
    def __init__(self, metadata_csv: str, label_map: dict): # label_map 인자 추가
        self.meta = pd.read_csv(metadata_csv, encoding="utf-8-sig")
        required = ["file_path", "keypoint_path", "action"]
        for c in required:
            if c not in self.meta.columns:
                raise KeyError(f"missing column: {c}")

        self.samples = []
        for _, r in self.meta.iterrows():
            v, k, action_str = str(r["file_path"]), str(r["keypoint_path"]), r["action"]
            # action_str이 label_map에 있고, 파일 경로가 유효한 경우에만 추가
            if action_str in label_map and Path(v).exists() and Path(k).exists():
                # int() 대신 label_map을 사용하여 정수 인덱스로 변환
                label_idx = label_map[action_str]
                self.samples.append((v, k, label_idx))

        print(f"[INFO] dataset samples: {len(self.samples)}")
        self.to_tensor = T.ToTensor()

    def __len__(self): return len(self.samples)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vpath, kpath, label = self.samples[idx]
        
        # .mp4 파일을 읽고 프레임으로 변환하는 로직으로 변경
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {vpath}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.to_tensor(frame))
        
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames could be read from video: {vpath}")
        
        # 여기서 프레임 수를 NUM_FRAMES에 맞춥니다.
        current_num_frames = len(frames)
        target_num_frames = 32 # NUM_FRAMES 값을 직접 사용하거나, 클래스 변수로 정의

        if current_num_frames < target_num_frames:
            # 프레임 수가 부족하면 마지막 프레임을 복사하여 패딩
            padding_frames = [frames[-1]] * (target_num_frames - current_num_frames)
            frames.extend(padding_frames)
        elif current_num_frames > target_num_frames:
            # 프레임 수가 너무 많으면 처음부터 균등하게 샘플링
            indices = np.linspace(0, current_num_frames - 1, target_num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        video = torch.stack(frames, dim=1)

        # 키포인트 파일은 .npy 형식이므로 np.load() 사용
        # allow_pickle=True는 여전히 필요할 수 있습니다.
        kp_array = np.load(kpath, allow_pickle=True).astype(np.float32)
        
        # 키포인트 배열도 프레임 수에 맞춥니다.
        kp_current_num = kp_array.shape[0]
        if kp_current_num < target_num_frames:
            padding_kp = np.tile(kp_array[-1:], (target_num_frames - kp_current_num, 1))
            kp_array = np.vstack([kp_array, padding_kp])
        elif kp_current_num > target_num_frames:
            indices = np.linspace(0, kp_current_num - 1, target_num_frames, dtype=int)
            kp_array = kp_array[indices]

        keypoint = torch.from_numpy(kp_array)
        label_t = torch.tensor(label, dtype=torch.long)
        
        return video, keypoint, label_t

class MultiModalDataset(Dataset):
    def __init__(self,
                 metadata_csv: str,
                 label_map: dict | None = None,
                 num_frames: int = 32,
                 image_size: tuple[int, int] = (224, 224),
                 seed: int = 0):
        super().__init__()
        self.metadata = pd.read_csv(metadata_csv)

        # 컬럼명 점검
        required_cols = ['file_path', 'keypoint_path', 'action']
        for col in required_cols:
            if col not in self.metadata.columns:
                raise KeyError(f"[ERROR] metadata_csv에 '{col}' 컬럼이 없습니다. "
                               f"현재 컬럼: {list(self.metadata.columns)}")

        # 라벨맵 생성
        if label_map is None:
            uniq_actions = sorted(self.metadata['action'].dropna().unique().tolist())
            self.label_map = {a: i for i, a in enumerate(uniq_actions)}
            print(f"[INFO] label_map 자동 생성: {self.label_map}")
        else:
            self.label_map = label_map

        self.samples = []
        for _, row in self.metadata.iterrows():
            vp, kp, act = row['file_path'], row['keypoint_path'], row['action']
            if pd.isna(vp) or pd.isna(kp) or act not in self.label_map:
                continue
            if os.path.exists(vp) and os.path.exists(kp):
                self.samples.append((vp, kp, self.label_map[act]))

        # 샘플 0개면 원인 진단
        if len(self.samples) == 0:
            print("\n[WARN] 샘플 수가 0입니다. 원인 진단을 시작합니다...\n")
            # 1) 경로 존재 여부
            vp_exists = self.metadata['file_path'].apply(os.path.exists)
            kp_exists = self.metadata['keypoint_path'].apply(os.path.exists)
            print(f"[CHECK] 영상 경로 존재 개수: {vp_exists.sum()} / {len(vp_exists)}")
            print(f"[CHECK] 키포인트 경로 존재 개수: {kp_exists.sum()} / {len(kp_exists)}")
            if vp_exists.sum() == 0:
                print("[CAUSE] 모든 영상 경로가 존재하지 않습니다. 경로 설정을 확인하세요.")
            if kp_exists.sum() == 0:
                print("[CAUSE] 모든 키포인트 경로가 존재하지 않습니다. 경로 설정을 확인하세요.")

            # 2) 라벨 매핑 여부
            unique_actions = set(self.metadata['action'].dropna())
            missing_labels = [a for a in unique_actions if a not in self.label_map]
            if missing_labels:
                print(f"[CAUSE] label_map에 없는 action이 있습니다: {missing_labels}")

            # 3) NaN 데이터 여부
            nan_rows = self.metadata[self.metadata[['file_path', 'keypoint_path', 'action']].isna().any(axis=1)]
            if not nan_rows.empty:
                print(f"[CAUSE] NaN 데이터가 있는 행 {len(nan_rows)}개 발견")

        # 샘플 셔플
        random.seed(seed)
        random.shuffle(self.samples)
        print(f"[INFO] 로드된 샘플 수: {len(self.samples)}")

        self.num_frames = num_frames
        self.transforms = T.Compose([
            ToPILImage(),
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, keypoint_path, label = self.samples[idx]
        video_tensor = self._load_video_tensor(video_path)
        keypoint_tensor = self._load_keypoint_tensor(keypoint_path)
        return video_tensor, keypoint_tensor, torch.tensor(label, dtype=torch.long)

    def _load_keypoint_tensor(self, kp_path):
        try:
            # np.loadtxt 대신 pandas의 read_csv를 사용
            # header=None으로 헤더가 없음을 명시
            # encoding='utf-8-sig'로 BOM 처리
            # dtype=np.float32로 데이터 타입 지정
            kp_full = pd.read_csv(kp_path, header=None, encoding='utf-8-sig', dtype=np.float32).values
            
            # 여기서 샘플링 로직을 추가
            idxs = np.linspace(0, kp_full.shape[0] - 1, num=self.num_frames).astype(int)
            kp_sampled = kp_full[idxs, :]
            
            return torch.from_numpy(kp_sampled).float()
        except Exception as e:
            print(f"[ERROR] 파일 로드 중 오류 발생: {kp_path}")
            raise e

    def _load_video_tensor(self, vid_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(vid_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, total - 1, num=self.num_frames, dtype=int)
        frames = []
        for frame_idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transforms(frame))
        cap.release()

        if not frames:
            raise RuntimeError(f"영상 로드 실패: {vid_path}")

        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        return torch.stack(frames, dim=1)  # (3, T, H, W)