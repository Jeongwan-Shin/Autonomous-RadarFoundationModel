# 독립 평가 번들 — 2026-08-08

다른 서버로 옮겨서 **테스트만** 하기 위한 폴더입니다. 원본 아카이브도 3.6 GB
파케이도 필요 없습니다. 아이템마다 모델이 실제로 받는 것이 안에 들어 있습니다.

## 먼저 받을 것 두 개

코드는 이 저장소에 있고, **큰 파일 두 개는 드라이브에서** 받습니다.

**https://drive.google.com/drive/folders/1zKflCPWc3kIL8W9qjhr5npqkNQenO_Bw**

| 파일 | 크기 | 무엇 |
|---|---|---|
| `model_8b_v4_step8100.tar.gz` | 14 GB | 학습된 체크포인트 (step 8,100, SFT 완료본) |
| `data_test_200.tar.gz` | 약 1.2 GB | 테스트 아이템 29종 × 200건 |

둘 다 이 폴더(`20260808_test/`) 안에서 풉니다.

```bash
cd 20260808_test
tar xzf model_8b_v4_step8100.tar.gz   # → vlm_8B_v4_step8100_20260809/
tar xzf data_test_200.tar.gz          # → data/
```

원본 Qwen3-VL-8B(17 GB)는 **받을 필요가 없습니다.** 가중치는 체크포인트 것을 쓰고
토크나이저와 프로세서 설정만 쓰는데, 그 파일들(11 MB)은 `base/` 에 저장소와 함께
들어 있습니다.

## 구성

```
20260808_test/
  run_eval.py          평가 실행 ← 이것만 돌리면 됩니다
  verify_bundle.py     모델을 올리기 전에 번들이 온전한지 확인
  export_items.py      번들을 만든 스크립트 — 원본 서버 전용, 옮긴 뒤에는 안 씁니다
  requirements.txt
  base/                토크나이저·프로세서 설정 (가중치 없음, 11 MB)
  ravl/                필요한 모듈 사본 (저장소를 참조하지 않습니다)
    connector.py  radar_encoder.py  task_scorers.py
    number_tokens.py  number_vocab.txt
  data/                ← 드라이브에서 받아 풉니다
    manifest.json      태스크별 건수, 레이더 채널 이름
    <task>.jsonl       아이템 목록 — 프롬프트 전문과 정답
    <task>/000..199/
      frames/f00.jpg…  그 아이템에 실제로 들어가는 프레임 (384x216)
      radar.npz        레이더 반사점 (패딩 제거)
```

## 실행

```bash
pip install -r requirements.txt        # torch 는 requirements.txt 주석 참고
python verify_bundle.py                # 모델 없이 초 단위로 끝납니다
python run_eval.py                     # 인자 없이 그대로
```

GPU 한 장, bf16 으로 약 20 GiB 를 씁니다.

일부만 빠르게 보려면:

```bash
python run_eval.py --tasks det_objects_azdeg,qa --items 20
```

## 나오는 것

| | |
|---|---|
| `results/scores.json` | 태스크별 점수 |
| `results/generations.jsonl` | 생성 전량 — 태스크, 클립, 생성, 정답 |

**점수만 보지 마세요.** 커버리지 0.00 이나 F1 0.00 은 모델이 못 푼 것이 아니라
채점기가 답을 못 읽은 경우가 있었습니다. `generations.jsonl` 을 열어 실제로 무엇을
내놓았는지 확인해야 그 둘이 구별됩니다.

## 주의할 점

**어휘가 154,165 입니다.** 레이더 자리표시자 2 개와 숫자 토큰 2,503 개가 원본
151,669 에 더해진 값입니다. `ravl/number_vocab.txt` 는 학습에 쓰인 것과 **글자
하나까지 같아야** 합니다 — 순서가 바뀌면 151,672 이후의 모든 id 가 밀립니다.
`run_eval.py` 가 시작할 때 모델과 토크나이저의 어휘 크기를 비교해 다르면 멈춥니다.

**레이더 주입 방식이 학습과 같아야 합니다.** `inputs_embeds` 로 넘기면 Qwen3-VL 의
비디오 스캐터가 꺼집니다. 그래서 입력 임베딩 모듈에 forward hook 을 걸어 자리표시자
행만 덮어씁니다. `run_eval.py` 의 `RadarInjector` 가 학습 코드와 같은 구현입니다.

**생성 길이 상한이 태스크마다 다릅니다.** CoT 는 근거를 먼저 쓰므로 평문의
+640 토큰을 받습니다. 이름이 표에 없는 태스크는 짧은 기본값 48 을 받아 답이
잘리는데, 그 실패는 "못 푸는 모델" 과 똑같이 보입니다.

**디스크는 32 GB 가 필요합니다** — 받은 tar 15 GB 와 푼 체크포인트 17 GB. 체크포인트를
풀고 나면 `model_8b_v4_step8100.tar.gz` 를 지워 14 GB 를 회수할 수 있습니다.

## 이 체크포인트의 점수

테스트 200 건씩, `data/` 와 같은 분할로 잰 값입니다.

| 태스크 | 지표 | 값 |
|---|---|---|
| `track_step_azdeg` | F1 | 0.698 |
| `track_step_bbox` | F1 | 0.679 |
| `det_objects_azdeg` | F1 | 0.276 |
| `det_objects_3dbbox` | F1 · size MAE · yaw MAE | 0.267 · 0.297 m · 18.8° |
| `motion_seg_bbox` | F1 | 0.258 |
| `motion_seg_azdeg` | F1 | 0.248 |
| `plan_ego_xy` | 변위 MAE | 0.914 m |
| `plan_ego_control` | 속도 MAE | 0.571 m/s |
| `agent_traj_azdeg` | 거리 MAE · 커버리지 | 1.882 m · 0.965 |
| `agent_traj_bbox` | IoU · 커버리지 | 0.365 · 0.967 |
| `qa` / `qa_cot` | 정답률 | 0.500 / 0.565 |

낮은 F1 을 곧바로 능력 부족으로 읽지 마세요. 이 표를 만드는 동안 채점기 결함을
두 개 고쳤고, 둘 다 모델이 아니라 채점기가 답을 못 읽던 것이었습니다 —
`agent_traj_bbox` 는 커버리지 0.00 으로 나오다가 형식을 읽게 하니 0.967 이
됐습니다.
