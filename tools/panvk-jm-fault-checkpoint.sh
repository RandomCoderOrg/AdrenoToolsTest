#!/usr/bin/env bash
set -euo pipefail

PKG=com.example.adrenotoolstest2
COMPONENT="$PKG/.MainActivity"
ADB_BIN="${ADB:-adb}"
RUNS="${RUNS:-1}"
ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
APK="$ROOT/app/build/outputs/apk/debug/app-debug.apk"
DRIVER="/storage/emulated/0/Android/data/$PKG/files/libvulkan_panfrost.so"
OUT="${EVIDENCE_DIR:-/tmp/panvk-jm-fault-$(date +%Y%m%d-%H%M%S)}"
MODES=(submit-eio ppoll-eio ppoll-timeout read-eio done-failure)

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf '%s\n' 'Usage: [ADB=adb] [RUNS=1] [EVIDENCE_DIR=/tmp/...] panvk-jm-fault-checkpoint.sh'
    exit 0
fi
[[ $# -eq 0 && "$RUNS" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid arguments or RUNS\n' >&2; exit 2; }

umask 077
mkdir -p "$OUT"
cp "$0" "$OUT/harness.sh"
LOGCAT="$OUT/logcat.txt"
SUMMARY="$OUT/summary.txt"
SUP="$OUT/supervisor.log"
RESULT=INCONCLUSIVE
REASON='unexpected script exit'
SERIAL=
BOOT0=
KPRCS=
LOGCAT_PID=
PID_VALUE=
START_TICKS=
COMPLETED=0

stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
note() { printf '%s PanvkJmHarness %s\n' "$(stamp)" "$*" | tee -a "$SUP"; }
adb() { "$ADB_BIN" -s "$SERIAL" "$@"; }
online() { [[ "$(adb get-state 2>/dev/null || true)" == device ]]; }
die_i() { RESULT=INCONCLUSIVE; REASON="$*"; note "INCONCLUSIVE $*"; exit 2; }
die_f() { RESULT=FAIL; REASON="$*"; note "FAIL $*"; exit 1; }
need_online() { online || die_i 'adb device disconnected'; }
need_logcat() {
    if [[ -z "$LOGCAT_PID" ]] || ! kill -0 "$LOGCAT_PID" 2>/dev/null; then
        die_i 'continuous logcat capture died'
    fi
}

sha_host() {
    if command -v shasum >/dev/null; then shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null; then sha256sum "$1" | awk '{print $1}'
    else return 1
    fi
}

finish() {
    local rc=$? boot1=UNAVAILABLE files
    trap - EXIT INT TERM
    set +e
    [[ -z "$LOGCAT_PID" ]] || { kill "$LOGCAT_PID" 2>/dev/null; wait "$LOGCAT_PID" 2>/dev/null; }
    [[ -z "$SERIAL" ]] || boot1="$(adb shell cat /proc/sys/kernel/random/boot_id 2>/dev/null | tr -d '\r')"
    [[ -n "$boot1" ]] || boot1=UNAVAILABLE
    printf 'result=%s\nreason=%s\nexit_code=%s\ncompleted_cases=%s\nboot_id_start=%s\nboot_id_end=%s\n' \
        "$RESULT" "$REASON" "$rc" "$COMPLETED" "${BOOT0:-UNAVAILABLE}" "$boot1" >"$SUMMARY"
    files="$(mktemp /tmp/panvk-jm-files.XXXXXX)"
    find "$OUT" -type f ! -name SHA256SUMS | LC_ALL=C sort >"$files"
    (cd "$OUT" || exit; while IFS= read -r file; do
        file=".${file#"$OUT"}"
        if command -v shasum >/dev/null; then shasum -a 256 "$file"; else sha256sum "$file"; fi
    done <"$files" >SHA256SUMS)
    rm -f "$files"
    exit "$rc"
}
trap finish EXIT
trap 'die_i "interrupted by signal"' INT TERM

read_pid() {
    local text rc words
    set +e
    text="$(adb shell pidof "$PKG" 2>/dev/null)"
    rc=$?
    set -e
    text="$(printf '%s' "$text" | tr -d '\r' | awk '{$1=$1;print}')"
    if [[ $rc -ne 0 ]]; then need_online; PID_VALUE=; return; fi
    [[ -n "$text" ]] || { PID_VALUE=; return; }
    read -r -a words <<<"$text"
    [[ ${#words[@]} -eq 1 && "${words[0]}" =~ ^[0-9]+$ ]] || die_i "ambiguous pidof output: $text"
    PID_VALUE="${words[0]}"
}

read_ticks() {
    local text rc
    set +e
    text="$(adb shell cat "/proc/$1/stat" 2>/dev/null)"
    rc=$?
    set -e
    [[ $rc -eq 0 ]] || { need_online; die_i "cannot read /proc/$1/stat"; }
    START_TICKS="$(printf '%s\n' "$text" | tr -d '\r' | awk '{print $22}')"
    [[ "$START_TICKS" =~ ^[0-9]+$ ]] || die_i "malformed /proc/$1/stat"
}

wait_new_pid() {
    local deadline=$(( $(date +%s) + 20 ))
    PID_VALUE=
    while [[ $(date +%s) -le $deadline ]]; do
        need_online
        need_logcat
        read_pid
        [[ -z "$PID_VALUE" ]] || return
        sleep 0.1
    done
    die_i 'launch produced no PID within 20s'
}

pid_log() { need_logcat; awk -v pid="$1" '$2 == pid' "$LOGCAT" >"$2"; }

force_stop_generation() {
    local pid="$1" ticks="$2" prefix="$3" expected_kprcs="$4" deadline state proc_state kprcs_state stat rc now exit0
    read_pid
    [[ "$PID_VALUE" == "$pid" ]] || die_i "PID $pid is not active before force-stop"
    read_ticks "$pid"
    [[ "$START_TICKS" == "$ticks" ]] || die_i "PID $pid generation changed before force-stop"
    kprcs_state="$(adb shell "if [ -e '$KPRCS/$pid' ]; then echo PRESENT; else echo ABSENT; fi" 2>/dev/null | tr -d '\r')"
    [[ "$kprcs_state" == "$expected_kprcs" ]] || die_i "kprcs/$pid was $kprcs_state before force-stop; expected $expected_kprcs"
    printf 'timestamp=%s phase=before-force-stop package=PRESENT proc=PRESENT_SAME_GENERATION kprcs=%s expected_kprcs=%s\n' \
        "$(stamp)" "$kprcs_state" "$expected_kprcs" >"$prefix-cleanup.txt"
    adb shell am force-stop "$PKG" >"$prefix-force-stop.txt" 2>&1 || { need_online; die_f 'force-stop failed'; }
    deadline=$(( $(date +%s) + 10 ))
    while [[ $(date +%s) -le $deadline ]]; do
        need_online
        need_logcat
        read_pid
        [[ -z "$PID_VALUE" ]] && state=ABSENT || state="PRESENT:$PID_VALUE"
        set +e
        stat="$(adb shell "if [ -r /proc/$pid/stat ]; then cat /proc/$pid/stat; else echo ABSENT; fi" 2>/dev/null | tr -d '\r')"
        rc=$?
        set -e
        [[ $rc -eq 0 ]] || { need_online; die_i 'cannot inspect old /proc generation'; }
        if [[ "$stat" == ABSENT ]]; then
            proc_state=ABSENT
        else
            now="$(printf '%s\n' "$stat" | awk '{print $22}')"
            [[ "$now" =~ ^[0-9]+$ ]] || die_i 'malformed old /proc evidence'
            if [[ "$now" == "$ticks" ]]; then
                proc_state=PRESENT_SAME_GENERATION
            else
                die_i "PID $pid was reused"
            fi
        fi
        kprcs_state="$(adb shell "if [ -e '$KPRCS/$pid' ]; then echo PRESENT; else echo ABSENT; fi" 2>/dev/null | tr -d '\r')"
        [[ "$kprcs_state" == PRESENT || "$kprcs_state" == ABSENT ]] || die_i 'cannot inspect kprcs cleanup'
        printf 'timestamp=%s package=%s proc=%s kprcs=%s\n' "$(stamp)" "$state" "$proc_state" "$kprcs_state" >>"$prefix-cleanup.txt"
        if [[ "$state" == ABSENT && "$proc_state" == ABSENT && "$kprcs_state" == ABSENT ]]; then
            deadline=$(( $(date +%s) + 5 ))
            while [[ $(date +%s) -le $deadline ]]; do
                adb shell dumpsys activity exit-info "$PKG" >"$prefix-exit-info.txt" 2>&1 || { need_online; die_i 'missing exit-info'; }
                exit0="$(awk '/ApplicationExitInfo #0:/{take=1; next} /ApplicationExitInfo #1:/{exit} take{print}' "$prefix-exit-info.txt")"
                if grep -Eq "[[:space:]]pid=$pid([[:space:]]|$)" <<<"$exit0" &&
                   grep -Fq 'reason=10 (USER REQUESTED) subreason=21 (FORCE STOP) status=0' <<<"$exit0"; then
                    return
                fi
                sleep 0.1
            done
            die_i "PID $pid newest exit-info is not a successful force-stop"
        fi
        sleep 0.1
    done
    die_f "PID $pid, /proc, or kprcs survived force-stop"
}

verify_fault_run() {
    local mode="$1" run="$2" prefix="$3" pid="$4" ticks="$5" deadline log armed consumed records
    log="$prefix-pid-log.txt"
    deadline=$(( $(date +%s) + 20 ))
    while [[ $(date +%s) -le $deadline ]]; do
        need_online
        need_logcat
        read_pid
        [[ "$PID_VALUE" == "$pid" ]] || die_i "$run PID vanished or changed before evidence completed"
        read_ticks "$pid"
        [[ "$START_TICKS" == "$ticks" ]] || die_i "$run PID generation changed"
        pid_log "$pid" "$log"
        records="$(grep -Fc 'jm-fault-app: result=' "$log" || true)"
        armed="$(grep -Fc "panvk: jm-fault run=$run mode=$mode state=ARMED " "$log" || true)"
        consumed="$(grep -Fc "panvk: jm-fault run=$run mode=$mode state=CONSUMED " "$log" || true)"
        [[ "$records" == 1 && "$armed" == 1 && "$consumed" == 1 ]] && break
        sleep 0.1
    done
    records="$(grep -Fc 'jm-fault-app: result=' "$log" || true)"
    [[ "$records" == 1 ]] || die_i "$run has $records app result records"
    armed="$(grep -Fc "panvk: jm-fault run=$run mode=$mode state=ARMED " "$log" || true)"
    consumed="$(grep -Fc "panvk: jm-fault run=$run mode=$mode state=CONSUMED " "$log" || true)"
    [[ "$armed" == 1 ]] || die_i "$run has $armed matching ARMED records"
    [[ "$consumed" == 1 ]] || die_i "$run has $consumed matching CONSUMED records"
    ! grep -Eq 'panvk: jm-fault .* state=(UNCONSUMED|INJECTION_ERROR) ' "$log" || die_i "$run emitted UNCONSUMED or INJECTION_ERROR"
    grep -Fq "jm-fault-app: result=PASS run=$run mode=$mode " "$log" || die_f "$run app contract did not pass"
    grep -F "jm-fault-app: result=" "$log" >"$prefix-app-result.txt"
    grep -F "panvk: jm-fault run=$run mode=$mode state=" "$log" >"$prefix-driver-events.txt"
}

verify_recovery() {
    local prefix="$1" pid="$2" ticks="$3" deadline log missing mark
    local marks=(
        'buffer-copy[stock]: PASS' 'buffer-copy[panvk]: PASS' 'compute[panvk]: PASS'
        'lifecycle-churn[panvk]: SUMMARY result=PASS_STABLE' 'jm-async[panvk]: PASS mode=waves'
        'jm-async[panvk]: PASS mode=single-submit-chain' 'jm-async[panvk]: PASS mode=binary-semaphore'
        'image-transfer[panvk]: PASS' 'sample[panvk]: PASS' 'indexed[panvk]: PASS'
        'blend[panvk]: PASS' 'depth[panvk]: PASS' 'ready extent=' 'first frame presented'
    )
    log="$prefix-pid-log.txt"
    deadline=$(( $(date +%s) + 35 ))
    while [[ $(date +%s) -le $deadline ]]; do
        need_online
        need_logcat
        read_pid
        [[ "$PID_VALUE" == "$pid" ]] || die_i 'recovery PID vanished or changed'
        read_ticks "$pid"
        [[ "$START_TICKS" == "$ticks" ]] || die_i 'recovery PID generation changed'
        pid_log "$pid" "$log"
        missing=
        for mark in "${marks[@]}"; do
            grep -Fq -- "$mark" "$log" || { missing="$mark"; break; }
        done
        [[ -z "$missing" ]] && break
        sleep 0.2
    done
    [[ -z "$missing" ]] || die_i "recovery missing gate: $missing"
    ! grep -Fq 'jm-fault-app:' "$log" || die_f 'no-fault recovery entered fault branch'
    ! grep -Fq 'panvk: jm-fault run=' "$log" || die_f 'no-fault recovery armed a selector'
    ! grep -Eq '(buffer-copy|compute|lifecycle-churn|jm-async|image-transfer|sample|indexed|blend|depth)\[panvk\]: FAIL|VK_ERROR_DEVICE_LOST' "$log" || die_f 'no-fault recovery gate failed'
}

note "suite phase=preconditions runs=$RUNS evidence=$OUT"
set +e
DEVICES="$($ADB_BIN devices 2>&1)"
rc=$?
set -e
[[ $rc -eq 0 ]] || die_i 'adb devices failed'
ROWS="$(printf '%s\n' "$DEVICES" | awk 'NR>1 && NF>=2 {print $1 " " $2}')"
[[ "$(printf '%s\n' "$ROWS" | awk 'NF{n++} END{print n+0}')" == 1 && "$(printf '%s\n' "$ROWS" | awk '$2=="device"{n++} END{print n+0}')" == 1 ]] || die_i 'need exactly one authorized adb device'
SERIAL="$(printf '%s\n' "$ROWS" | awk '$2=="device"{print $1}')"
need_online
adb shell pm path "$PKG" >"$OUT/package-path.txt" 2>&1 || die_i "$PKG is not installed"
PACKAGE_PATH="$(sed -n 's/^package://p' "$OUT/package-path.txt" | tr -d '\r')"
[[ -n "$PACKAGE_PATH" && "$(printf '%s\n' "$PACKAGE_PATH" | wc -l | awk '{print $1}')" == 1 ]] || die_i "$PKG has no unique base APK"
[[ -f "$APK" ]] || die_i "local APK missing: $APK"
APK_SHA="$(sha_host "$APK")" || die_i 'no SHA-256 utility'
INSTALLED_APK_SHA="$(adb shell "toybox sha256sum '$PACKAGE_PATH'" 2>/dev/null | tr -d '\r' | awk '{print $1}')"
[[ "$INSTALLED_APK_SHA" =~ ^[0-9a-fA-F]{64}$ ]] || die_i 'cannot hash installed APK'
[[ "$INSTALLED_APK_SHA" == "$APK_SHA" ]] || die_i 'installed APK does not match local checkpoint build'
DRIVER_SHA="$(adb shell "toybox sha256sum '$DRIVER' 2>/dev/null || sha256sum '$DRIVER' 2>/dev/null" | tr -d '\r' | awk '{print $1}')"
[[ "$DRIVER_SHA" =~ ^[0-9a-fA-F]{64}$ ]] || die_i "driver missing or unreadable: $DRIVER"
KLIST="$(adb shell "for d in /sys/devices/platform/*.mali/kprcs; do [ -d \"\$d\" ] && [ -r \"\$d\" ] && echo \"\$d\"; done" 2>/dev/null | tr -d '\r')"
[[ "$(printf '%s\n' "$KLIST" | awk 'NF{n++} END{print n+0}')" == 1 ]] || die_i 'expected one readable Mali kprcs directory'
KPRCS="$(printf '%s\n' "$KLIST" | awk 'NF{print;exit}')"
BOOT0="$(adb shell cat /proc/sys/kernel/random/boot_id | tr -d '\r')"
[[ -n "$BOOT0" ]] || die_i 'missing boot ID'
printf 'serial=%s\nboot_id=%s\nruns=%s\nkprcs=%s\napk=%s\napk_sha256=%s\ninstalled_apk=%s\ninstalled_apk_sha256=%s\ndriver=%s\ndriver_sha256=%s\n' \
    "$SERIAL" "$BOOT0" "$RUNS" "$KPRCS" "$APK" "$APK_SHA" "$PACKAGE_PATH" "$INSTALLED_APK_SHA" "$DRIVER" "$DRIVER_SHA" >"$OUT/metadata.txt"

adb shell am force-stop "$PKG" >/dev/null 2>&1 || die_i 'initial force-stop failed'
adb logcat -b all -c || die_i 'cannot clear logcat'
: >"$LOGCAT"
"$ADB_BIN" -s "$SERIAL" logcat -b all -v epoch >"$LOGCAT" 2>&1 &
LOGCAT_PID=$!
sleep 0.3
need_logcat

for ((rep=1; rep<=RUNS; rep++)); do
    for mode in "${MODES[@]}"; do
        run="m1_${rep}_${mode//-/_}_$(date +%s)"
        prefix="$OUT/$run"
        read_pid
        [[ -z "$PID_VALUE" ]] || die_f "$run began with package PID $PID_VALUE"
        note "run=$run mode=$mode phase=fault-launch"
        adb shell am start -W -n "$COMPONENT" --es panvk_jm_fault "$run:$mode" >"$prefix-am-start.txt" 2>&1 || { need_online; die_f "$run am start failed"; }
        wait_new_pid
        fault_pid="$PID_VALUE"
        read_ticks "$fault_pid"
        fault_ticks="$START_TICKS"
        printf 'pid=%s\nstart_ticks=%s\n' "$fault_pid" "$fault_ticks" >"$prefix-process.txt"
        verify_fault_run "$mode" "$run" "$prefix" "$fault_pid" "$fault_ticks"
        fault_kprcs=PRESENT
        if [[ "$mode" == ppoll-timeout ]]; then
            fault_kprcs=ABSENT
        fi
        force_stop_generation "$fault_pid" "$fault_ticks" "$prefix" "$fault_kprcs"

        recovery="$prefix-recovery"
        note "run=$run phase=cold-recovery"
        adb shell am start -W -n "$COMPONENT" >"$recovery-am-start.txt" 2>&1 || { need_online; die_f "$run recovery start failed"; }
        wait_new_pid
        recovery_pid="$PID_VALUE"
        read_ticks "$recovery_pid"
        recovery_ticks="$START_TICKS"
        printf 'pid=%s\nstart_ticks=%s\n' "$recovery_pid" "$recovery_ticks" >"$recovery-process.txt"
        verify_recovery "$recovery" "$recovery_pid" "$recovery_ticks"
        force_stop_generation "$recovery_pid" "$recovery_ticks" "$recovery" PRESENT
        boot_now="$(adb shell cat /proc/sys/kernel/random/boot_id | tr -d '\r')"
        [[ "$boot_now" == "$BOOT0" ]] || die_i 'device rebooted during suite'
        COMPLETED=$((COMPLETED + 1))
        note "run=$run mode=$mode phase=passed"
    done
done

FAULTS="$OUT/fault-scan.txt"
grep -Ei "Fatal signal|ANR in $PKG|>>> $PKG <<<|Mali[^[:cntrl:]]*(MMU fault|GPU reset|job fault|hang)|kbase[^[:cntrl:]]*(MMU fault|GPU reset|job fault|hang)" "$LOGCAT" >"$FAULTS" || true
[[ ! -s "$FAULTS" ]] || die_f 'fault scan matched a GPU fault, fatal signal, tombstone, or ANR'
printf 'clean: no configured fault patterns matched the captured interval\n' >"$FAULTS"
RESULT=PASS
REASON="$COMPLETED fault cases and cold recoveries passed"
note "PASS cases=$COMPLETED"
