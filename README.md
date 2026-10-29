# Griptape Local Video Nodes

MiniMax-H3 (FL2VA / Ref2VA / LoRA), Wan2.1-Fun-Control, AnimateDiff를 Griptape Nodes에서 로컬 GPU로 실행하기 위한 비디오 노드 패키지입니다.

---

## 📌 포함된 노드

| 노드 | 설명 |
|------|------|
| **Local MiniMax Video Generation** | MiniMax-H3 33B 옴니모달 비디오 생성 (T2V, I2V, R2V, LoRA 융합, 32kHz 네이티브 오디오 동시 생성) |
| **Local Wan Video Generation** | Wan2.1 Fun-Control 기반 비디오 생성 (카메라 컨트롤 지원) |
| **Local AnimateDiff Generation** | AnimateDiff 모션 모듈 기반 비디오 생성 |

---

## 🏗️ 아키텍처

```text
Griptape Desktop
└── Video Generation Node (HTTP 비동기 통신)
        ↓
    Video Gen Server (FastAPI, port 8090)
        ├── MiniMax-H3 Engine (FL2VA / Ref2VA / In-place LoRA Fusion)
        ├── Qwen3-VL 7B Multimodal Encoder
        ├── Wan2.1 Engine (Fun-Control)
        └── Native Stereo Audio Synthesis (32kHz)
```

Griptape Desktop 본체와 비디오 생성 서버가 별도 프로세스 및 독립 가상환경(`.venv`)으로 분리되어 있어, PyTorch 버전 충돌이나 메모리 간섭 없이 안전하게 동작합니다.

---

## ⚙️ 요구사항

- **OS**: Windows 10/11
- **GPU**: NVIDIA RTX 3000 / 4000 / 5000 시리즈 (12GB ~ 24GB VRAM 권장)
- **RAM**: 최소 32GB (64GB 권장)
- **Griptape Nodes Desktop**: 설치되어 있어야 함
- **Python**: 3.12 (uv 또는 venv)

---

## 🚀 설치 및 사용법

### 1. 레포지토리 클론
```bash
git clone https://github.com/qvrex00-stack/griptape-local-video-nodes.git
cd griptape-local-video-nodes
```

### 2. Griptape 라이브러리에 노드 배포
```powershell
python install.py --griptape-dir "C:\Foundry\Griptape"
```

### 3. 비디오 서버 실행
노드 실행 시 자동으로 서버가 백그라운드에서 기동되거나, 수동으로 실행할 수 있습니다:
```bat
cd video_gen_server
start_server.bat
```
