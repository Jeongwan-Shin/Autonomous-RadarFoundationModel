# 독립 평가 번들 — 2026-08-08

다른 서버로 옮겨서 **테스트만** 하기 위한 폴더입니다. 원본 아카이브도 3.7 GB
파케이도 필요 없습니다. 아이템마다 모델이 실제로 받는 것이 안에 들어 있습니다.

## 배치가 클립 중심입니다

전에는 태스크마다 200건씩 뽑았습니다. 그래서 태스크끼리 클립이 거의 겹치지
않았고 — `det_objects` 의 198개 클립과 `plan_ego` 의 199개 클립 중 공통은
**1개** 였습니다 — 같은 장면을 놓고 태스크를 비교할 수가 없었습니다. 아이템은
자기가 쓰는 프레임만 들고 있어서(`det_objects` 1장, `track_step` 5장) 클립을
통째로 볼 수도 없었습니다.

지금은 **클립 10개**를 먼저 고르고, 그 클립이 만들어 내는 태스크를 전부 그
옆에 모읍니다. 고른 열 개는 모두 **31종 태스크를 빠짐없이** 냅니다.

```
data/
  manifest.json           클립 수, 아이템 수, 태스크별 건수, 레이더 채널 이름
  clips.jsonl             클립 한 줄씩 — id, 프레임 수, 아이템 수, 태스크 목록
  clips/<clip_id>/
    frames/f00.jpg … f19.jpg   20 프레임 (1 Hz), 클립당 한 벌
    radar/<key>.npz            레이더 창. (프레임, 표본 속도) 가 같으면 공유
                               f03_instant / f05_hz4 / clip 같은 이름
    ego.npy                    ego 상태 원본 (speed, accel, yaw-rate)
    ego.txt                    프롬프트가 인용하는 그대로의 문자열
    tasks.jsonl                이 클립이 내놓는 모든 아이템
  by_task/<task>.jsonl    같은 아이템을 태스크로 색인한 것 — run_eval 이 읽습니다
```

아이템은 프레임을 **번호로** 가리킵니다(`"frames": [1, 2, 3, 4, 5]`). 같은
프레임이 스무 번 복사되지 않으므로 번들이 1.4 GB 에서 **78 MB** 로 줄었습니다.

한 클립 안에서 태스크가 무엇을 묻는지 이렇게 보입니다:

```
det_objects_azdeg  (프레임 [3], 레이더 f03_instant)
  Q: List every road user in the forward sector with its class, range and azimuth.
  A: automobile 4 m az +48 deg; automobile 4 m az -50 deg; …
agent_traj_xy      (프레임 [2, 3], 레이더 f03_hz10)
  Q: Track #7 is a automobile at (+27.2, -2.6) m. Where will it be over the next 3 seconds?
  A: +1s (+18.5, +3.2); +2s (+7.6, +7.2); +3s leaves the forward sector
track_step_azdeg   (프레임 [1, 2, 3, 4, 5], 레이더 f05_hz4)
  Q: Continue tracking: list the road users at this instant, reusing the id …
  A: #35 automobile 4 m az +48 deg; #85 automobile 4 m az -51 deg; …
qa                 (프레임 [0 … 19], 레이더 clip)
```

**점수를 말하기에는 작은 표본입니다.** 클립 10개 · 아이템 3,410건이라, 여기서
나오는 값은 지표가 아니라 참고입니다. 채점용으로 쓰려면 클립 수를 올려
다시 만드세요:

```bash
python export_items.py --clips 200        # 원본 서버에서만 돌아갑니다
```

## 먼저 받을 것

코드는 이 저장소에 있고, **모델은 드라이브에서** 받습니다.

**https://drive.google.com/drive/folders/1zKflCPWc3kIL8W9qjhr5npqkNQenO_Bw**

| 파일 | 크기 | md5 | 무엇 |
|---|---|---|---|
| `model_8b_v9_6ch_step2200.tar.gz` | 14.0 GB | `268d6f23…` | 체크포인트 (step 2,200) |
| `data_10clips.tar.gz` | 49.7 MB | `4aa3a118…` | 클립 10개 · 아이템 3,410건 |

**둘은 짝입니다. 섞어 쓰면 안 됩니다.** 예전에 올라가 있던
`model_8b_v4_step8100.tar.gz` 는 레이더를 8 채널로 읽고, 지금 `data/` 는 6
채널로 나갑니다 — 형상이 맞지 않아 죽거나, 더 나쁘게는 채널이 어긋난 채로
돌아갑니다. 그 체크포인트는 프레임 결함(아래) 이전 것이기도 합니다.

`data/` 는 78 MB 로 줄었지만(처음에는 1.4 GB) 저장소에는 넣지 않았습니다 —
깃 히스토리에 들어가면 되돌리기 어려워서입니다. 원본 서버에서는 직접 만들 수도
있습니다:

```bash
python export_items.py --clips 10
```

원본 Qwen3-VL-8B(17 GB)도 **받을 필요가 없습니다.** 가중치는 체크포인트 것을
쓰고 토크나이저와 프로세서 설정만 쓰는데, 그 파일들(11 MB)은 `base/` 에
들어 있습니다.

## 실행

```bash
pip install -r requirements.txt        # torch 는 requirements.txt 주석 참고
tar xzf data_10clips.tar.gz                # → data/
tar xzf model_8b_v9_6ch_step2200.tar.gz    # → vlm_8B_v9_6ch_step2200/
python verify_bundle.py                    # 모델 없이 초 단위로 끝납니다
python run_eval.py --checkpoint ./vlm_8B_v9_6ch_step2200
```

GPU 한 장, bf16 으로 약 20 GiB 를 씁니다. 일부만 빠르게 보려면:

```bash
python run_eval.py --tasks det_objects_azdeg,agent_traj_xy --items 20
```

## 나오는 것

| | |
|---|---|
| `results/scores.json` | 태스크별 점수 |
| `results/generations.jsonl` | 생성 전량 — 태스크, 클립, 생성, 정답 |

**점수만 보지 마세요.** 커버리지 0.00 이나 F1 0.00 은 모델이 못 푼 것이 아니라
채점기가 답을 못 읽은 경우가 있었습니다. `generations.jsonl` 을 열어 실제로
무엇을 내놓았는지 확인해야 그 둘이 구별됩니다.

## 주의할 점

**어휘가 154,165 입니다.** 레이더 자리표시자 2 개와 숫자 토큰 2,503 개가 원본
151,669 에 더해진 값입니다. `ravl/number_vocab.txt` 는 학습에 쓰인 것과 **글자
하나까지 같아야** 합니다 — 순서가 바뀌면 151,672 이후의 모든 id 가 밀립니다.
`run_eval.py` 가 시작할 때 모델과 토크나이저의 어휘 크기를 비교해 다르면
멈춥니다.

**`ravl/` 사본은 손으로 복사하지 마세요.** `export_items.py` 가 번들을 만들
때마다 저장소에서 다시 복사합니다. 손으로 두었을 때 실제로 낡았습니다 —
`agent_traj_xy` 가 생긴 뒤 번들의 채점기는 그 태스크를 몰라 일반 텍스트로
떨어뜨렸고, 그 실패는 점수 0.00 과 구별되지 않습니다.

**레이더 주입 방식이 학습과 같아야 합니다.** `inputs_embeds` 로 넘기면
Qwen3-VL 의 비디오 스캐터가 꺼집니다. 그래서 입력 임베딩 모듈에 forward hook 을
걸어 자리표시자 행만 덮어씁니다. `run_eval.py` 의 `RadarInjector` 가 학습
코드와 같은 구현입니다.

**생성 길이 상한이 태스크마다 다릅니다.** CoT 는 근거를 먼저 쓰므로 평문의
+640 토큰을 받습니다. 이름이 표에 없는 태스크는 짧은 기본값 48 을 받아 답이
잘리는데, 그 실패는 "못 푸는 모델" 과 똑같이 보입니다.

## 태스크 31종

| 태스크 | 건수 | | 태스크 | 건수 |
|---|---|---|---|---|
| `det_objects_azdeg` | 60 | | `plan_ego_xy` | 80 |
| `det_objects_3dbbox` | 60 | | `plan_ego_control` | 80 |
| `track_step_azdeg` | 150 | | `agent_traj_xy` | 69 |
| `track_step_bbox` | 146 | | `agent_traj_azdeg` | 69 |
| `motion_seg_azdeg` | 88 | | `agent_traj_bbox` | 68 |
| `motion_seg_bbox` | 88 | | `qa` | 139 |
| `desc_radar` | 208 | | `radar_probe` | 208 |
| `desc_objects` | 208 | | `radar_transfer` | 208 |
| `desc_complementarity` | 202 | | `desc_ego_maneuver` | 208 |
| `desc_clip_summary` | 10 | | | |

여기에 `_cot` 짝이 각각 붙습니다. `agent_traj_xy` 는 궤적을 미터로 내는
형식이고 — 질문 시점의 ego 프레임에서 (x, y), `plan_ego_xy` 와 같은 프레임 —
`azdeg` 와 `bbox` 는 각 horizon 시점 기준이라 서로 좌표계가 다릅니다.
지시문이 각자 자기 프레임을 밝힙니다.
