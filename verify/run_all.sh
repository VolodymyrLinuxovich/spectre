#!/bin/sh
# Build the verifier and check every case listed in a case directory's index.txt.
# Usage: verify/run_all.sh [case_dir]   (default: verify/cases)
set -u
here="$(cd "$(dirname "$0")" && pwd)"
dir="${1:-$here/cases}"
make -s -C "$here" verify || exit 2

if [ ! -f "$dir/index.txt" ]; then
    echo "no $dir/index.txt; run verify/export.py first" >&2
    exit 2
fi

total=0
failed=0
while read -r case_file; do
    [ -z "$case_file" ] && continue
    total=$((total + 1))
    "$here/verify" "$dir/$case_file" || failed=$((failed + 1))
done < "$dir/index.txt"

echo "$((total - failed)) of $total verdicts reproduced"
[ "$failed" -eq 0 ]
