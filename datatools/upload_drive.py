#!/usr/bin/env python3
"""큰 파일을 구글 드라이브 폴더에 올린다. 이어올리기와 무결성 확인까지.

14 GB 를 한 번에 밀어 넣으면 중간에 끊겼을 때 처음부터 다시 해야 한다. 재개
가능 업로드(resumable)로 조각을 나눠 보내고, 끝나면 드라이브가 계산한 md5 를
로컬 값과 대조한다. 대조하지 않으면 "올라간 것 같다" 까지만 알 수 있다.

인증은 서비스 계정 JSON 하나로 한다. OAuth 흐름은 브라우저가 필요하고, 그
결과로 나오는 토큰을 어딘가에 붙여넣어야 하는데 -- 서비스 계정 키는 서버에
직접 두면 되므로 비밀이 대화나 로그를 지나가지 않는다.

    # 1) GCP 에서 서비스 계정을 만들고 JSON 키를 받는다
    # 2) 그 키를 서버에 직접 옮긴다 (scp)
    # 3) 드라이브 폴더를 그 계정 이메일과 "편집자" 로 공유한다
    #    (키 파일의 client_email 값)
    export GOOGLE_APPLICATION_CREDENTIALS=/path/key.json
    python -m datatools.upload_drive --folder <폴더ID> dist/model_8b_v4_step8100.tar.gz

서비스 계정은 자기 저장 용량이 없다. 공유 드라이브가 아닌 개인 폴더에 올리면
할당량 오류가 나므로, 그때는 `--impersonate you@example.com` 으로 위임하거나
공유 드라이브를 쓴다.
"""

import argparse
import hashlib
import os
import sys

SCOPES = ["https://www.googleapis.com/auth/drive"]
CHUNK = 64 * 1024 * 1024      # 64 MiB. 작으면 왕복이 늘고 크면 재시도 비용이 커진다


def md5(path, block=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def service(impersonate=None):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key or not os.path.exists(key):
        raise SystemExit(
            "GOOGLE_APPLICATION_CREDENTIALS 가 서비스 계정 JSON 을 가리켜야 "
            "합니다. 키 파일은 서버에 직접 두세요 -- 내용을 붙여넣지 마시고.")
    creds = service_account.Credentials.from_service_account_file(key,
                                                                  scopes=SCOPES)
    if impersonate:
        creds = creds.with_subject(impersonate)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def existing(api, folder, name):
    """같은 이름의 파일이 이미 있으면 그 id -- 새 사본을 쌓지 않고 덮어쓴다."""
    q = (f"'{folder}' in parents and name = '{name}' and trashed = false")
    got = api.files().list(q=q, fields="files(id,name,md5Checksum,size)",
                           supportsAllDrives=True,
                           includeItemsFromAllDrives=True).execute()
    return (got.get("files") or [None])[0]


def upload(api, folder, path, replace=True):
    from googleapiclient.http import MediaFileUpload
    name = os.path.basename(path)
    size = os.path.getsize(path)
    local = md5(path)
    print(f"{name}  {size/1e9:.2f} GB  md5 {local}", flush=True)

    prior = existing(api, folder, name)
    if prior and prior.get("md5Checksum") == local:
        print("  이미 같은 내용이 올라가 있습니다 -- 건너뜁니다", flush=True)
        return prior["id"]

    media = MediaFileUpload(path, chunksize=CHUNK, resumable=True)
    if prior and replace:
        req = api.files().update(fileId=prior["id"], media_body=media,
                                 fields="id,md5Checksum,size",
                                 supportsAllDrives=True)
        print(f"  같은 이름이 있어 덮어씁니다 ({prior['id']})", flush=True)
    else:
        req = api.files().create(body={"name": name, "parents": [folder]},
                                 media_body=media,
                                 fields="id,md5Checksum,size",
                                 supportsAllDrives=True)
    done, last = None, -1
    while done is None:
        status, done = req.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            if pct >= last + 5:
                last = pct
                print(f"  {pct:>3}%", flush=True)
    remote = done.get("md5Checksum")
    if remote != local:
        raise SystemExit(f"!! md5 불일치: 로컬 {local} vs 드라이브 {remote}")
    print(f"  완료, md5 일치 -- https://drive.google.com/file/d/{done['id']}",
          flush=True)
    return done["id"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folder", required=True, help="드라이브 폴더 ID")
    ap.add_argument("--impersonate", default=None,
                    help="개인 폴더에 올릴 때 위임할 사용자 이메일")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)

    api = service(args.impersonate)
    for p in args.paths:
        if not os.path.exists(p):
            print(f"!! 없음: {p}", flush=True)
            continue
        upload(api, args.folder, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
