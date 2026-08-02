#!/bin/bash
set -uo pipefail

###############################################################################
# nuScenes Dataset Downloader
# - Downloads all files via wget, extracts each .tgz, removes the .tgz
# - Safe to interrupt and re-run: partial downloads resume, completed files
#   are skipped via marker files.
#
# Usage:
#   bash download_nuscenes.sh
###############################################################################

BASE_DIR="/NHNHOME/workspace/dataset/Auto_data/nuScenes"
TRAINVAL_DIR="${BASE_DIR}/trainval"
TEST_DIR="${BASE_DIR}/test"
STAGE_DIR="${BASE_DIR}/.staging"    # in-flight .tgz downloads live here
MARK_DIR="${BASE_DIR}/.completed"   # one marker file per fully extracted tgz

MAX_TRIES=3                         # download attempts per file

declare -A URLS=(
    ["v1.0-trainval_meta.tgz"]="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz"
    ["v1.0-trainval01_blobs.tgz"]="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz"
    ["v1.0-trainval02_blobs.tgz"]="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz"
    ["v1.0-trainval03_blobs.tgz"]="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_blobs.tgz"
    ["v1.0-trainval04_blobs.tgz"]="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz"
    ["v1.0-trainval05_blobs.tgz"]="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz"
    ["v1.0-trainval06_blobs.tgz"]="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_blobs.tgz"
    ["v1.0-trainval07_blobs.tgz"]="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz"
    ["v1.0-trainval08_blobs.tgz"]="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz"
    ["v1.0-trainval09_blobs.tgz"]="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz"
    ["v1.0-trainval10_blobs.tgz"]="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz"
    ["v1.0-test_meta.tgz"]="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz"
    ["v1.0-test_blobs.tgz"]="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz"
)

# Ordered file list (associative arrays don't preserve order).
# Meta first so the directory structure is usable early.
FILES=(
    "v1.0-trainval_meta.tgz"
    "v1.0-test_meta.tgz"
    "v1.0-trainval01_blobs.tgz"
    "v1.0-trainval02_blobs.tgz"
    "v1.0-trainval03_blobs.tgz"
    "v1.0-trainval04_blobs.tgz"
    "v1.0-trainval05_blobs.tgz"
    "v1.0-trainval06_blobs.tgz"
    "v1.0-trainval07_blobs.tgz"
    "v1.0-trainval08_blobs.tgz"
    "v1.0-trainval09_blobs.tgz"
    "v1.0-trainval10_blobs.tgz"
    "v1.0-test_blobs.tgz"
)

# Map each file to its target directory
declare -A TARGET_DIR=(
    ["v1.0-trainval_meta.tgz"]="$TRAINVAL_DIR"
    ["v1.0-trainval01_blobs.tgz"]="$TRAINVAL_DIR"
    ["v1.0-trainval02_blobs.tgz"]="$TRAINVAL_DIR"
    ["v1.0-trainval03_blobs.tgz"]="$TRAINVAL_DIR"
    ["v1.0-trainval04_blobs.tgz"]="$TRAINVAL_DIR"
    ["v1.0-trainval05_blobs.tgz"]="$TRAINVAL_DIR"
    ["v1.0-trainval06_blobs.tgz"]="$TRAINVAL_DIR"
    ["v1.0-trainval07_blobs.tgz"]="$TRAINVAL_DIR"
    ["v1.0-trainval08_blobs.tgz"]="$TRAINVAL_DIR"
    ["v1.0-trainval09_blobs.tgz"]="$TRAINVAL_DIR"
    ["v1.0-trainval10_blobs.tgz"]="$TRAINVAL_DIR"
    ["v1.0-test_meta.tgz"]="$TEST_DIR"
    ["v1.0-test_blobs.tgz"]="$TEST_DIR"
)

mkdir -p "$TRAINVAL_DIR" "$TEST_DIR" "$STAGE_DIR" "$MARK_DIR" || {
    echo "!! Cannot create $BASE_DIR. Check the path and permissions."
    exit 1
}

FAIL_LOG="${BASE_DIR}/failed_downloads.log"

log_fail() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$FAIL_LOG"
}

# Remote size in bytes, or empty if the server didn't answer
remote_size() {
    curl -sIL --max-time 60 "$1" \
        | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}'
}

local_size() {
    [ -f "$1" ] && stat -c %s "$1" || echo 0
}

human() {
    awk -v b="$1" 'BEGIN { printf "%.2f GB", b/1073741824 }'
}

TOTAL=${#FILES[@]}
CURRENT=0
FAIL_COUNT=0
DONE_COUNT=0

echo "================================================================"
echo "nuScenes download"
echo "  target : $BASE_DIR"
echo "  files  : $TOTAL"
echo "  free   : $(df -h "$BASE_DIR" | tail -1 | awk '{print $4}')"
echo "  started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

for FILE in "${FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    DEST="${TARGET_DIR[$FILE]}"
    URL="${URLS[$FILE]}"
    PART="${STAGE_DIR}/${FILE}"
    MARK="${MARK_DIR}/${FILE}.done"

    echo ""
    echo "================================================================"
    echo "[$CURRENT/$TOTAL] $FILE   ($(date '+%H:%M:%S'))"
    echo "================================================================"

    # Already extracted on a previous run?
    if [ -f "$MARK" ]; then
        echo "  -> Already extracted (marker present). Skipping."
        DONE_COUNT=$((DONE_COUNT + 1))
        continue
    fi

    EXPECTED=$(remote_size "$URL")
    if [ -z "$EXPECTED" ]; then
        echo "  !! Could not read Content-Length from server. Skipping."
        log_fail "NO CONTENT-LENGTH: $FILE | URL: $URL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    echo "  -> Expected size: $(human "$EXPECTED")"

    # Download with resume, verifying the final size each attempt.
    OK=0
    for TRY in $(seq 1 "$MAX_TRIES"); do
        HAVE=$(local_size "$PART")
        if [ "$HAVE" -eq "$EXPECTED" ]; then
            echo "  -> Already fully downloaded."
            OK=1
            break
        fi
        if [ "$HAVE" -gt 0 ]; then
            echo "  -> Resuming from $(human "$HAVE") (attempt $TRY/$MAX_TRIES)"
        else
            echo "  -> Downloading (attempt $TRY/$MAX_TRIES)"
        fi

        wget --continue --progress=dot:giga \
             --tries=5 --timeout=60 --waitretry=10 \
             -O "$PART" "$URL"

        HAVE=$(local_size "$PART")
        if [ "$HAVE" -eq "$EXPECTED" ]; then
            OK=1
            break
        fi

        echo "  !! Size mismatch: got $(human "$HAVE"), expected $(human "$EXPECTED")"
        # A resumed download that lands on the wrong size means the partial file
        # is unusable — start clean on the next attempt.
        if [ "$TRY" -lt "$MAX_TRIES" ]; then
            echo "  -> Discarding partial file and retrying from scratch."
            rm -f "$PART"
        fi
    done

    if [ "$OK" -ne 1 ]; then
        echo "  !! Download failed after $MAX_TRIES attempts. Skipping."
        log_fail "DOWNLOAD FAILED: $FILE | URL: $URL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    echo "  -> Extracting to $DEST ..."
    if ! tar -xzf "$PART" -C "$DEST"; then
        echo "  !! Extraction failed. Keeping .tgz in $STAGE_DIR for inspection."
        log_fail "EXTRACTION FAILED: $FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    touch "$MARK"
    rm -f "$PART"
    DONE_COUNT=$((DONE_COUNT + 1))
    echo "  -> Done. ($DONE_COUNT/$TOTAL complete)"
done

echo ""
echo "================================================================"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "Completed with $FAIL_COUNT failure(s). Re-run this script to retry them."
    echo "Log: $FAIL_LOG"
    echo "----------------------------------------------------------------"
    cat "$FAIL_LOG"
else
    echo "All $DONE_COUNT/$TOTAL files complete. No failures."
fi
echo "Data saved to:"
echo "  trainval -> $TRAINVAL_DIR"
echo "  test     -> $TEST_DIR"
echo "================================================================"
du -sh "$TRAINVAL_DIR" "$TEST_DIR" 2>/dev/null
