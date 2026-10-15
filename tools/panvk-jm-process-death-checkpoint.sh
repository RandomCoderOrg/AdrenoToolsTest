#!/usr/bin/env bash
set -euo pipefail

PKG=com.example.adrenotoolstest2
COMPONENT="$PKG/.MainActivity"
ADB_BIN="${ADB:-adb}"
RUNS="${RUNS:-3}"
EXPECTED_SERIAL="${EXPECTED_SERIAL:-}"
DRIVER_IMAGE="${DRIVER_IMAGE:-}"
ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
APK="$ROOT/app/build/outputs/apk/debug/app-debug.apk"
DRIVER="/storage/emulated/0/Android/data/$PKG/files/libvulkan_panfrost.so"
OUT="${EVIDENCE_DIR:-/tmp/panvk-jm-process-death-$(date +%Y%m%d-%H%M%S)}"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf '%s\n' 'Usage: DRIVER_IMAGE=/path/libvulkan_panfrost.so [ADB=adb] [EXPECTED_SERIAL=...] [RUNS=3] [EVIDENCE_DIR=/tmp/...] panvk-jm-process-death-checkpoint.sh'
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
TRACKED_PID=
TRACKED_TICKS=
LAUNCH_PENDING=0
CLEANUP_CLASS=OK
CLEANUP_REASON=
COMPLETED=0

stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
note() { printf '%s PanvkJmDeathHarness %s\n' "$(stamp)" "$*" | tee -a "$SUP"; }
adb() { "$ADB_BIN" -s "$SERIAL" "$@"; }
online() { [[ -n "$SERIAL" && "$(adb get-state 2>/dev/null || true)" == device ]]; }
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

read_pid_raw() {
    local text rc had_errexit=0
    [[ $- == *e* ]] && had_errexit=1
    set +e
    text="$(adb shell pidof "$PKG" 2>/dev/null)"
    rc=$?
    [[ $had_errexit -eq 0 ]] || set -e
    text="$(printf '%s' "$text" | tr -d '\r' | awk '{$1=$1;print}')"
    if [[ $rc -ne 0 || -z "$text" ]]; then
        PID_VALUE=
        return 0
    fi
    [[ "$text" =~ ^[0-9]+$ ]] || return 1
    PID_VALUE="$text"
}

read_ticks_raw() {
    local text
    text="$(adb shell cat "/proc/$1/stat" 2>/dev/null | tr -d '\r')" || return 1
    START_TICKS="$(printf '%s\n' "$text" | awk '{print $22}')"
    [[ "$START_TICKS" =~ ^[0-9]+$ ]]
}

best_effort_cleanup() {
    local pid rc deadline stat now
    CLEANUP_CLASS=OK
    CLEANUP_REASON=
    [[ -n "$SERIAL" ]] || return 0

    if [[ "$LAUNCH_PENDING" == 1 && -z "$TRACKED_PID" ]]; then
        deadline=$(( $(date +%s) + 5 ))
        PID_VALUE=
        while [[ $(date +%s) -le $deadline && -z "$PID_VALUE" ]]; do
            if ! online; then
                CLEANUP_CLASS=INCONCLUSIVE
                CLEANUP_REASON='adb unavailable while resolving a pending launch'
                return 0
            fi
            if ! read_pid_raw; then
                CLEANUP_CLASS=INCONCLUSIVE
                CLEANUP_REASON='ambiguous PID while resolving a pending launch'
                return 0
            fi
            [[ -n "$PID_VALUE" ]] || sleep 0.1
        done
        if [[ -z "$PID_VALUE" ]]; then
            printf 'source=launch-pending-fallback generation=ABSENT\n' >>"$OUT/trap-cleanup.txt"
            LAUNCH_PENDING=0
            return 0
        fi
        pid="$PID_VALUE"
        if ! read_ticks_raw "$pid"; then
            CLEANUP_CLASS=INCONCLUSIVE
            CLEANUP_REASON='cannot establish starttime for a pending launch'
            return 0
        fi
        TRACKED_PID="$pid"
        TRACKED_TICKS="$START_TICKS"
        LAUNCH_PENDING=0
        printf 'source=launch-pending-fallback pid=%s start_ticks=%s\n' \
            "$TRACKED_PID" "$TRACKED_TICKS" >>"$OUT/trap-cleanup.txt"
    fi

    [[ -n "$TRACKED_PID" && -n "$TRACKED_TICKS" ]] || return 0
    if ! online; then
        CLEANUP_CLASS=INCONCLUSIVE
        CLEANUP_REASON='adb unavailable before cleanup generation proof'
        return 0
    fi
    if ! read_pid_raw; then
        CLEANUP_CLASS=INCONCLUSIVE
        CLEANUP_REASON='ambiguous PID before cleanup generation proof'
        return 0
    fi
    pid="$PID_VALUE"
    if [[ -n "$pid" && "$pid" != "$TRACKED_PID" ]]; then
        CLEANUP_CLASS=INCONCLUSIVE
        CLEANUP_REASON='current package PID differs from tracked generation'
        return 0
    fi
    stat="$(adb shell "if [ -r /proc/$TRACKED_PID/stat ]; then cat /proc/$TRACKED_PID/stat; else echo ABSENT; fi" 2>/dev/null | tr -d '\r')"
    if [[ "$stat" == ABSENT ]]; then
        [[ -z "$pid" ]] || {
            CLEANUP_CLASS=INCONCLUSIVE
            CLEANUP_REASON='tracked PID is active but its starttime is unavailable'
            return 0
        }
        TRACKED_PID=
        TRACKED_TICKS=
        return 0
    fi
    now="$(printf '%s\n' "$stat" | awk '{print $22}')"
    if [[ ! "$now" =~ ^[0-9]+$ || "$now" != "$TRACKED_TICKS" ]]; then
        CLEANUP_CLASS=INCONCLUSIVE
        CLEANUP_REASON='tracked PID generation cannot be reproved before cleanup'
        return 0
    fi

    adb shell am force-stop "$PKG" >>"$OUT/trap-cleanup.txt" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
        if online; then
            CLEANUP_CLASS=FAIL
            CLEANUP_REASON='force-stop failed for the reverified tracked generation'
        else
            CLEANUP_CLASS=INCONCLUSIVE
            CLEANUP_REASON='adb disconnected during tracked-generation force-stop'
        fi
        return 0
    fi

    deadline=$(( $(date +%s) + 5 ))
    while [[ $(date +%s) -le $deadline ]]; do
        if ! online || ! read_pid_raw; then
            CLEANUP_CLASS=INCONCLUSIVE
            CLEANUP_REASON='cleanup result cannot be observed'
            return 0
        fi
        pid="$PID_VALUE"
        if [[ -n "$pid" && "$pid" != "$TRACKED_PID" ]]; then
            CLEANUP_CLASS=INCONCLUSIVE
            CLEANUP_REASON='a different package generation appeared during cleanup'
            return 0
        fi
        stat="$(adb shell "if [ -r /proc/$TRACKED_PID/stat ]; then cat /proc/$TRACKED_PID/stat; else echo ABSENT; fi" 2>/dev/null | tr -d '\r')"
        if [[ "$stat" == ABSENT && -z "$pid" ]]; then
            TRACKED_PID=
            TRACKED_TICKS=
            return 0
        fi
        if [[ "$stat" != ABSENT ]]; then
            now="$(printf '%s\n' "$stat" | awk '{print $22}')"
            if [[ ! "$now" =~ ^[0-9]+$ || "$now" != "$TRACKED_TICKS" ]]; then
                CLEANUP_CLASS=INCONCLUSIVE
                CLEANUP_REASON='tracked PID was reused during cleanup observation'
                return 0
            fi
        fi
        sleep 0.1
    done
    CLEANUP_CLASS=FAIL
    CLEANUP_REASON='reverified tracked generation survived force-stop'
}

finish() {
    local rc=$? boot1=UNAVAILABLE files
    trap - EXIT INT TERM
    set +e
    best_effort_cleanup
    [[ -z "$LOGCAT_PID" ]] || { kill "$LOGCAT_PID" 2>/dev/null; wait "$LOGCAT_PID" 2>/dev/null; }
    [[ -z "$SERIAL" ]] || boot1="$(adb shell cat /proc/sys/kernel/random/boot_id 2>/dev/null | tr -d '\r')"
    [[ -n "$boot1" ]] || boot1=UNAVAILABLE
    if [[ "$CLEANUP_CLASS" == FAIL ]]; then
        RESULT=FAIL
        rc=1
    elif [[ "$CLEANUP_CLASS" == INCONCLUSIVE && "$RESULT" != FAIL ]]; then
        RESULT=INCONCLUSIVE
        rc=2
    fi
    [[ "$CLEANUP_CLASS" == OK ]] || REASON="$CLEANUP_REASON"
    if [[ -n "$BOOT0" && "$boot1" == UNAVAILABLE && "$RESULT" == PASS ]]; then
        RESULT=INCONCLUSIVE
        REASON='ending boot ID evidence is unavailable'
        rc=2
    elif [[ -n "$BOOT0" && "$boot1" != UNAVAILABLE && "$boot1" != "$BOOT0" ]]; then
        RESULT=FAIL
        REASON='device boot ID changed'
        rc=1
    fi
    printf 'result=%s\nreason=%s\nexit_code=%s\ncompleted_runs=%s\nboot_id_start=%s\nboot_id_end=%s\n' \
        "$RESULT" "$REASON" "$rc" "$COMPLETED" "${BOOT0:-UNAVAILABLE}" "$boot1" >"$SUMMARY"
    files="$(mktemp /tmp/panvk-jm-death-files.XXXXXX)"
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

track_new_generation() {
    local deadline=$(( $(date +%s) + 20 ))
    PID_VALUE=
    while [[ $(date +%s) -le $deadline ]]; do
        need_online
        read_pid_raw || die_i 'ambiguous pidof output'
        if [[ -n "$PID_VALUE" ]]; then
            read_ticks_raw "$PID_VALUE" || die_i 'cannot read launched process starttime'
            TRACKED_PID="$PID_VALUE"
            TRACKED_TICKS="$START_TICKS"
            LAUNCH_PENDING=0
            return 0
        fi
        sleep 0.1
    done
    die_i 'launch produced no PID within 20s'
}

pid_log() { need_logcat; awk -v pid="$1" '$2 == pid' "$LOGCAT" >"$2"; }

parse_lifetime_create() {
    local line="$1" payload token key value seen=' ' count=0
    local -a tokens=()
    L_ERROR=''
    L_EVENT='' L_ROLE='' L_PID='' L_INSTANCE='' L_PHYSICAL='' L_LOGICAL=''
    L_KMOD='' L_FD='' L_PATH='' L_COUNT=0
    [[ "$line" == *'panvk: jm-lifetime '* ]] || { L_ERROR='missing marker'; return 1; }
    payload="${line#*panvk: jm-lifetime }"
    [[ -n "$payload" && "$payload" != *'panvk: jm-lifetime '* ]] || { L_ERROR='duplicate or empty marker'; return 1; }
    read -r -a tokens <<<"$payload"
    for token in "${tokens[@]}"; do
        [[ "$token" == *=* ]] || { L_ERROR="token lacks equals: $token"; return 1; }
        key="${token%%=*}"
        value="${token#*=}"
        [[ -n "$key" && -n "$value" && "$value" != *=* ]] || { L_ERROR="malformed token: $token"; return 1; }
        [[ "$seen" != *" $key "* ]] || { L_ERROR="duplicate key: $key"; return 1; }
        seen+="$key "
        count=$((count + 1))
        case "$key" in
            event) L_EVENT="$value" ;;
            role) L_ROLE="$value" ;;
            pid) L_PID="$value" ;;
            instance) L_INSTANCE="$value" ;;
            physical) L_PHYSICAL="$value" ;;
            logical) L_LOGICAL="$value" ;;
            kmod) L_KMOD="$value" ;;
            fd) L_FD="$value" ;;
            path) L_PATH="$value" ;;
            *) L_ERROR="unknown key: $key"; return 1 ;;
        esac
    done
    L_COUNT="$count"
    [[ "$L_PID" =~ ^[0-9]+$ && "$L_INSTANCE" =~ ^0x[0-9a-fA-F]+$ &&
       "$L_PHYSICAL" =~ ^0x[0-9a-fA-F]+$ && "$L_KMOD" =~ ^0x[0-9a-fA-F]+$ &&
       "$L_FD" =~ ^[0-9]+$ ]] || { L_ERROR='invalid typed field'; return 1; }
}

parse_fault_record() {
    local line="$1" payload token key value seen=' ' count=0
    local -a tokens=()
    F_ERROR=''
    F_RUN='' F_MODE='' F_STATE='' F_TARGET='' F_REMAINING='' F_LAST=''
    F_COMPLETED='' F_HEAD='' F_COUNT='' F_FREE='' F_LOST='' F_FAILED='' F_ERRNO=''
    [[ "$line" == *'panvk: jm-fault '* ]] || { F_ERROR='missing marker'; return 1; }
    payload="${line#*panvk: jm-fault }"
    [[ -n "$payload" && "$payload" != *'panvk: jm-fault '* ]] || { F_ERROR='duplicate or empty marker'; return 1; }
    read -r -a tokens <<<"$payload"
    for token in "${tokens[@]}"; do
        [[ "$token" == *=* ]] || { F_ERROR="token lacks equals: $token"; return 1; }
        key="${token%%=*}"
        value="${token#*=}"
        [[ -n "$key" && -n "$value" && "$value" != *=* ]] || { F_ERROR="malformed token: $token"; return 1; }
        [[ "$seen" != *" $key "* ]] || { F_ERROR="duplicate key: $key"; return 1; }
        seen+="$key "
        count=$((count + 1))
        case "$key" in
            run) F_RUN="$value" ;;
            mode) F_MODE="$value" ;;
            state) F_STATE="$value" ;;
            target) F_TARGET="$value" ;;
            remaining) F_REMAINING="$value" ;;
            last_seqno) F_LAST="$value" ;;
            completed_seqno) F_COMPLETED="$value" ;;
            ring_head) F_HEAD="$value" ;;
            ring_count) F_COUNT="$value" ;;
            free_atoms) F_FREE="$value" ;;
            lost) F_LOST="$value" ;;
            failed_events_observed_injected) F_FAILED="$value" ;;
            error) F_ERRNO="$value" ;;
            *) F_ERROR="unknown key: $key"; return 1 ;;
        esac
    done
    [[ $count -eq 13 ]] || { F_ERROR="field count $count"; return 1; }
    [[ "$F_RUN" =~ ^[A-Za-z0-9._-]+$ && "$F_MODE" == hold-after-submit &&
       "$F_STATE" =~ ^[A-Z_]+$ ]] || { F_ERROR='invalid identity field'; return 1; }
    for value in "$F_TARGET" "$F_REMAINING" "$F_LAST" "$F_COMPLETED" "$F_HEAD" \
                 "$F_COUNT" "$F_FREE" "$F_LOST" "$F_FAILED"; do
        [[ "$value" =~ ^[0-9]+$ ]] || { F_ERROR='invalid unsigned field'; return 1; }
    done
    [[ "$F_ERRNO" =~ ^-?[0-9]+$ ]] || { F_ERROR='invalid error field'; return 1; }
}

parser_self_test() {
    local create='panvk: jm-lifetime event=CREATE role=physical pid=7 instance=0x1 physical=0x2 kmod=0x3 fd=9 path=/dev/mali0'
    local ready='panvk: jm-fault run=self mode=hold-after-submit state=READY_FOR_KILL target=1 remaining=0 last_seqno=1 completed_seqno=0 ring_head=0 ring_count=1 free_atoms=254 lost=0 failed_events_observed_injected=0 error=0'
    parse_lifetime_create "$create" && [[ "$L_COUNT" == 8 && "$L_FD" == 9 ]] &&
        parse_fault_record "$ready" && [[ "$F_STATE" == READY_FOR_KILL && "$F_COUNT" == 1 ]] &&
        ! parse_lifetime_create "$create fd=10" &&
        ! parse_fault_record "${ready/ remaining=0/}"
}

capture_diagnostics() {
    local pid="$1" prefix="$2" rc
    set +e
    adb shell run-as "$PKG" ls -l "/proc/$pid/fd" >"$prefix-fds.txt" 2>"$prefix-fds.stderr"; rc=$?
    printf 'rc=%s\n' "$rc" >"$prefix-fds.status"
    adb shell run-as "$PKG" cat "/proc/$pid/maps" >"$prefix-maps.txt" 2>"$prefix-maps.stderr"; rc=$?
    printf 'rc=%s\n' "$rc" >"$prefix-maps.status"
    adb shell run-as "$PKG" cat "/proc/$pid/status" >"$prefix-status.txt" 2>"$prefix-status.stderr"; rc=$?
    printf 'rc=%s\n' "$rc" >"$prefix-status.status"
    adb shell "run-as $PKG sh -c 'for f in /proc/$pid/task/*/comm; do printf \"%s \" \"\${f%/comm}\"; cat \"\$f\" || echo ENOENT; done'" \
        >"$prefix-tasks.txt" 2>"$prefix-tasks.stderr"; rc=$?
    printf 'rc=%s\n' "$rc" >"$prefix-tasks.status"
    adb shell "find '$KPRCS/$pid' -maxdepth 2 -type f -print -exec cat '{}' ';'" \
        >"$prefix-gpu-accounting.txt" 2>"$prefix-gpu-accounting.stderr"; rc=$?
    printf 'rc=%s\n' "$rc" >"$prefix-gpu-accounting.status"
    set -e
}

maps_has_exact_path() {
    awk -v path="$1" '$NF == path { found = 1 } END { exit !found }' "$2"
}

maps_has_deleted_path() {
    awk -v path="$1" '$NF == "(deleted)" && $(NF - 1) == path { found = 1 } END { exit !found }' "$2"
}

verify_private_driver() {
    local prefix="$1" hash_output hash_rc private_hash private_path path_rc maps_rc
    local user_path="/data/user/0/$PKG/files/libvulkan_panfrost.so"
    local data_path="/data/data/$PKG/files/libvulkan_panfrost.so"
    local maps_path alias_used=0
    set +e
    hash_output="$(adb shell run-as "$PKG" toybox sha256sum files/libvulkan_panfrost.so \
        2>"$prefix-private-driver.stderr" | tr -d '\r')"
    hash_rc=$?
    private_path="$(adb shell "run-as $PKG sh -c 'cd files && printf \"%s/libvulkan_panfrost.so\\n\" \"\$PWD\"'" \
        2>"$prefix-private-path.stderr" | tr -d '\r')"
    path_rc=$?
    set -e
    private_hash="$(printf '%s\n' "$hash_output" | awk 'NF == 2 {print $1}')"
    printf 'hash_rc=%s\nsha256=%s\nexpected_sha256=%s\npath_rc=%s\npath=%s\n' \
        "$hash_rc" "${private_hash:-UNAVAILABLE}" "$DRIVER_IMAGE_SHA" "$path_rc" \
        "${private_path:-UNAVAILABLE}" >"$prefix-private-driver.txt"
    [[ $hash_rc -eq 0 && "$private_hash" =~ ^[0-9a-f]{64}$ ]] ||
        die_i "$prefix cannot hash the app-private PanVK driver"
    [[ "$private_hash" == "$DRIVER_IMAGE_SHA" ]] ||
        die_f "$prefix app-private PanVK driver hash differs from DRIVER_IMAGE"
    [[ $path_rc -eq 0 && "$private_path" == /* && "$private_path" != *$'\n'* ]] ||
        die_i "$prefix cannot establish the app-private PanVK driver path"
    maps_rc="$(sed -n 's/^rc=//p' "$prefix-maps.status")"
    [[ "$maps_rc" == 0 ]] || die_i "$prefix maps evidence is unavailable"
    if maps_has_deleted_path "$user_path" "$prefix-maps.txt" ||
       maps_has_deleted_path "$data_path" "$prefix-maps.txt"; then
        die_i "$prefix app-private PanVK mapping is deleted; loaded-inode attestation is unavailable"
    fi
    if [[ "$private_path" == "$user_path" ]]; then
        maps_path="$user_path"
        if ! maps_has_exact_path "$maps_path" "$prefix-maps.txt"; then
            maps_path="$data_path"
            alias_used=1
        fi
    elif [[ "$private_path" == "$data_path" ]]; then
        maps_path="$data_path"
        if ! maps_has_exact_path "$maps_path" "$prefix-maps.txt"; then
            maps_path="$user_path"
            alias_used=1
        fi
    else
        die_f "$prefix app-private PanVK driver resolved outside the standard Android app-data aliases"
    fi
    printf 'maps_path=%s\nalias_used=%s\n' "$maps_path" "$alias_used" >>"$prefix-private-driver.txt"
    maps_has_exact_path "$maps_path" "$prefix-maps.txt" ||
        die_f "$prefix maps do not contain the app-private PanVK driver path"
}

verify_fd_target() {
    local pid="$1" fd="$2" label="$3" prefix="$4" target rc
    set +e
    target="$(adb shell run-as "$PKG" readlink "/proc/$pid/fd/$fd" 2>"$prefix-$label-fd.stderr" | tr -d '\r')"
    rc=$?
    set -e
    printf 'fd=%s rc=%s target=%s\n' "$fd" "$rc" "${target:-UNAVAILABLE}" >"$prefix-$label-fd.txt"
    [[ $rc -eq 0 && -n "$target" ]] || die_i "$prefix cannot resolve $label fd $fd"
    [[ "$target" == /dev/mali0 ]] || die_f "$prefix $label fd $fd resolves to $target"
}

verify_hold_evidence() {
    local run="$1" prefix="$2" pid="$3" ticks="$4" deadline log lifetime_count fault_count app_count
    local line i=0 physical_line logical_line armed_line ready_line
    local physical_instance='' physical_object='' physical_kmod='' physical_fd=''
    local logical_kmod='' logical_fd=''
    log="$prefix-pid-log.txt"
    deadline=$(( $(date +%s) + 20 ))
    while [[ $(date +%s) -le $deadline ]]; do
        need_online
        need_logcat
        read_pid_raw || die_i "$run has ambiguous pidof output"
        [[ "$PID_VALUE" == "$pid" ]] || die_i "$run PID vanished or changed before READY evidence"
        read_ticks_raw "$pid" || die_i "$run cannot read starttime"
        [[ "$START_TICKS" == "$ticks" ]] || die_i "$run PID generation changed"
        pid_log "$pid" "$log"
        lifetime_count="$(grep -Fc 'panvk: jm-lifetime ' "$log" || true)"
        fault_count="$(grep -Fc 'panvk: jm-fault ' "$log" || true)"
        app_count="$(grep -Fc 'jm-fault-app: result=' "$log" || true)"
        [[ "$app_count" == 0 ]] || die_f "$run unexpectedly returned from the accepted-but-unretired submit"
        ! grep -Fq 'panvk: jm-lifetime event=DESTROY_' "$log" || die_f "$run destroyed a PanVK owner before process death"
        ! grep -Eq 'panvk: jm-fault .* state=(UNCONSUMED|INJECTION_ERROR) ' "$log" ||
            die_f "$run emitted UNCONSUMED or INJECTION_ERROR"
        [[ "$lifetime_count" -le 2 && "$fault_count" -le 2 ]] ||
            die_i "$run emitted duplicate or extra log evidence"
        [[ "$lifetime_count" == 2 && "$fault_count" == 2 ]] && break
        sleep 0.1
    done
    [[ "$lifetime_count" == 2 && "$fault_count" == 2 ]] || die_i "$run missing CREATE, ARMED, or READY evidence"
    ! grep -Eq 'panvk: jm-fault .* state=(UNCONSUMED|INJECTION_ERROR) ' "$log" || die_f "$run emitted UNCONSUMED or INJECTION_ERROR"
    ! grep -Fq 'jm-fault-app: result=' "$log" || die_f "$run unexpectedly returned from the accepted-but-unretired submit"
    ! grep -Fq 'panvk: jm-lifetime event=DESTROY_' "$log" || die_f "$run destroyed a PanVK owner before process death"

    awk 'index($0, "panvk: jm-lifetime ") { print }' "$log" >"$prefix-lifetime.txt"
    if sed 's/^.*panvk: jm-lifetime //' "$prefix-lifetime.txt" | LC_ALL=C sort | uniq -d | grep -q .; then
        die_i "$run contains duplicate lifetime payloads"
    fi
    while IFS= read -r line; do
        i=$((i + 1))
        parse_lifetime_create "$line" || die_i "$run lifetime record $i malformed: $L_ERROR"
        [[ "$L_PID" == "$pid" ]] || die_f "$run lifetime record belongs to PID $L_PID"
        if [[ $i -eq 1 ]]; then
            [[ "$L_EVENT" == CREATE && "$L_ROLE" == physical && "$L_COUNT" == 8 ]] ||
                die_f "$run first lifetime record is not physical CREATE"
            [[ -z "$L_LOGICAL" && "$L_PATH" == /dev/mali0 ]] || die_f "$run physical CREATE contract mismatch"
            physical_instance="$L_INSTANCE"
            physical_object="$L_PHYSICAL"
            physical_kmod="$L_KMOD"
            physical_fd="$L_FD"
        else
            [[ "$L_EVENT" == CREATE && "$L_ROLE" == logical && "$L_COUNT" == 9 ]] ||
                die_f "$run second lifetime record is not logical CREATE"
            [[ "$L_LOGICAL" =~ ^0x[0-9a-fA-F]+$ && "$L_PATH" == /dev/mali0 ]] ||
                die_f "$run logical CREATE contract mismatch"
            [[ "$L_INSTANCE" == "$physical_instance" && "$L_PHYSICAL" == "$physical_object" ]] ||
                die_f "$run physical/logical CREATE identities do not pair"
            logical_kmod="$L_KMOD"
            logical_fd="$L_FD"
            [[ "$logical_kmod" != "$physical_kmod" && "$logical_fd" != "$physical_fd" ]] ||
                die_f "$run physical/logical owners are not distinct"
        fi
    done <"$prefix-lifetime.txt"

    grep -F 'panvk: jm-fault ' "$log" >"$prefix-driver-events.txt"
    if sed 's/^.*panvk: jm-fault //' "$prefix-driver-events.txt" | LC_ALL=C sort | uniq -d | grep -q .; then
        die_i "$run contains duplicate fault payloads"
    fi
    while IFS= read -r line; do
        parse_fault_record "$line" || die_i "$run malformed fault record: $F_ERROR"
        [[ "$F_RUN" == "$run" && "$F_MODE" == hold-after-submit ]] || die_f "$run fault record identity mismatch"
        [[ "$F_STATE" == ARMED || "$F_STATE" == READY_FOR_KILL ]] || die_f "$run unexpected fault state $F_STATE"
    done <"$prefix-driver-events.txt"
    line="$(grep -F "panvk: jm-fault run=$run mode=hold-after-submit state=ARMED " "$log" || true)"
    [[ -n "$line" && "$(printf '%s\n' "$line" | wc -l | awk '{print $1}')" == 1 ]] || die_i "$run missing unique ARMED record"
    parse_fault_record "$line" || die_i "$run malformed ARMED record: $F_ERROR"
    [[ "$F_RUN" == "$run" && "$F_STATE" == ARMED ]] || die_f "$run ARMED identity mismatch"
    line="$(grep -F "panvk: jm-fault run=$run mode=hold-after-submit state=READY_FOR_KILL " "$log" || true)"
    [[ -n "$line" && "$(printf '%s\n' "$line" | wc -l | awk '{print $1}')" == 1 ]] || die_i "$run missing unique READY_FOR_KILL record"
    parse_fault_record "$line" || die_i "$run malformed READY_FOR_KILL record: $F_ERROR"
    [[ "$F_RUN" == "$run" && "$F_STATE" == READY_FOR_KILL && "$F_TARGET" == 1 &&
       "$F_REMAINING" == 0 && "$F_LAST" -gt "$F_COMPLETED" && "$F_COUNT" -gt 0 &&
       "$F_FREE" -lt 255 && "$F_LOST" == 0 && "$F_ERRNO" == 0 ]] ||
        die_f "$run READY_FOR_KILL state contradicts an accepted-but-unretired submit"

    physical_line="$(awk 'index($0, "panvk: jm-lifetime event=CREATE role=physical ") { print NR }' "$log")"
    logical_line="$(awk 'index($0, "panvk: jm-lifetime event=CREATE role=logical ") { print NR }' "$log")"
    armed_line="$(awk -v p="panvk: jm-fault run=$run mode=hold-after-submit state=ARMED " 'index($0, p) { print NR }' "$log")"
    ready_line="$(awk -v p="panvk: jm-fault run=$run mode=hold-after-submit state=READY_FOR_KILL " 'index($0, p) { print NR }' "$log")"
    [[ "$physical_line" =~ ^[0-9]+$ && "$logical_line" =~ ^[0-9]+$ && "$armed_line" =~ ^[0-9]+$ &&
       "$ready_line" =~ ^[0-9]+$ ]] || die_i "$run cannot establish event order"
    [[ "$physical_line" -lt "$logical_line" && "$physical_line" -lt "$ready_line" &&
       "$logical_line" -lt "$ready_line" && "$armed_line" -lt "$ready_line" ]] ||
        die_f "$run CREATE and ARMED records did not precede READY_FOR_KILL"

    kprcs_state="$(adb shell "if [ -e '$KPRCS/$pid' ]; then echo PRESENT; else echo ABSENT; fi" 2>/dev/null | tr -d '\r')"
    [[ "$kprcs_state" == PRESENT ]] || die_f "$run kprcs/$pid is not present at READY_FOR_KILL"
    printf 'kprcs=%s\n' "$kprcs_state" >"$prefix-kprcs-before-kill.txt"
    verify_fd_target "$pid" "$physical_fd" physical "$prefix"
    verify_fd_target "$pid" "$logical_fd" logical "$prefix"
    capture_diagnostics "$pid" "$prefix"
    verify_private_driver "$prefix"
}

force_stop_generation() {
    local pid="$1" ticks="$2" prefix="$3" deadline state proc_state kprcs_state stat now exit0
    need_online
    read_pid_raw || die_i "$prefix has ambiguous pidof output before force-stop"
    [[ "$PID_VALUE" == "$pid" ]] || die_i "$prefix tracked PID is not current before force-stop"
    read_ticks_raw "$pid" || die_i "$prefix cannot read tracked starttime before force-stop"
    [[ "$START_TICKS" == "$ticks" ]] || die_i "$prefix tracked generation changed before force-stop"
    adb shell am force-stop "$PKG" >"$prefix-force-stop.txt" 2>&1 || { need_online; die_f "$prefix force-stop failed"; }
    deadline=$(( $(date +%s) + 10 ))
    while [[ $(date +%s) -le $deadline ]]; do
        need_online
        need_logcat
        read_pid_raw || die_i "$prefix has ambiguous pidof output during cleanup"
        [[ -z "$PID_VALUE" ]] && state=ABSENT || state="PRESENT:$PID_VALUE"
        [[ -z "$PID_VALUE" || "$PID_VALUE" == "$pid" ]] ||
            die_i "$prefix a different package generation appeared during cleanup"
        stat="$(adb shell "if [ -r /proc/$pid/stat ]; then cat /proc/$pid/stat; else echo ABSENT; fi" 2>/dev/null | tr -d '\r')" ||
            die_i "$prefix cannot inspect old /proc generation"
        if [[ "$stat" == ABSENT ]]; then
            proc_state=ABSENT
        else
            now="$(printf '%s\n' "$stat" | awk '{print $22}')"
            [[ "$now" =~ ^[0-9]+$ ]] || die_i "$prefix malformed old /proc evidence"
            [[ "$now" == "$ticks" ]] || die_i "$prefix PID $pid was reused"
            proc_state=PRESENT_SAME_GENERATION
        fi
        kprcs_state="$(adb shell "if [ -e '$KPRCS/$pid' ]; then echo PRESENT; else echo ABSENT; fi" 2>/dev/null | tr -d '\r')"
        [[ "$kprcs_state" == PRESENT || "$kprcs_state" == ABSENT ]] || die_i "$prefix cannot inspect kprcs cleanup"
        printf 'timestamp=%s package=%s proc=%s kprcs=%s\n' "$(stamp)" "$state" "$proc_state" "$kprcs_state" >>"$prefix-cleanup.txt"
        if [[ "$state" == ABSENT && "$proc_state" == ABSENT && "$kprcs_state" == ABSENT ]]; then
            deadline=$(( $(date +%s) + 5 ))
            while [[ $(date +%s) -le $deadline ]]; do
                adb shell dumpsys activity exit-info "$PKG" >"$prefix-exit-info.txt" 2>&1 || { need_online; die_i "$prefix missing exit-info"; }
                exit0="$(awk '/ApplicationExitInfo #0:/{take=1; next} /ApplicationExitInfo #1:/{exit} take{print}' "$prefix-exit-info.txt")"
                if grep -Eq "[[:space:]]pid=$pid([[:space:]]|$)" <<<"$exit0" &&
                   grep -Fq 'reason=10 (USER REQUESTED) subreason=21 (FORCE STOP) status=0' <<<"$exit0"; then
                    TRACKED_PID=
                    TRACKED_TICKS=
                    return 0
                fi
                sleep 0.1
            done
            die_i "$prefix newest exit-info does not prove a successful force-stop of PID $pid"
        fi
        sleep 0.1
    done
    die_f "$prefix package, old /proc generation, or kprcs survived force-stop"
}

verify_no_post_kill_records() {
    local run="$1" prefix="$2" pid="$3" log
    log="$prefix-pid-log-after-kill.txt"
    sleep 0.2
    pid_log "$pid" "$log"
    ! grep -Fq 'jm-fault-app: result=' "$log" || die_f "$run emitted an app result"
    ! grep -Fq 'panvk: jm-lifetime event=DESTROY_' "$log" || die_f "$run emitted destructor records during process death"
    ! grep -Eq 'panvk: jm-fault .* state=(UNCONSUMED|INJECTION_ERROR) ' "$log" ||
        die_f "$run emitted UNCONSUMED or INJECTION_ERROR"
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
    missing=waiting
    while [[ $(date +%s) -le $deadline ]]; do
        need_online
        need_logcat
        read_pid_raw || die_i "$prefix has ambiguous recovery pidof output"
        [[ "$PID_VALUE" == "$pid" ]] || die_f "$prefix recovery PID vanished or changed"
        read_ticks_raw "$pid" || die_i "$prefix cannot read recovery starttime"
        [[ "$START_TICKS" == "$ticks" ]] || die_f "$prefix recovery PID generation changed"
        pid_log "$pid" "$log"
        missing=
        for mark in "${marks[@]}"; do
            grep -Fq -- "$mark" "$log" || { missing="$mark"; break; }
        done
        [[ -z "$missing" ]] && break
        sleep 0.2
    done
    [[ -z "$missing" ]] || die_f "$prefix recovery missing gate: $missing"
    ! grep -Fq 'jm-fault-app:' "$log" || die_f "$prefix no-fault recovery entered fault branch"
    ! grep -Fq 'panvk: jm-fault run=' "$log" || die_f "$prefix no-fault recovery armed a selector"
    ! grep -Eq '(buffer-copy|compute|lifecycle-churn|jm-async|image-transfer|sample|indexed|blend|depth)\[panvk\]: FAIL|VK_ERROR_DEVICE_LOST' "$log" ||
        die_f "$prefix normal recovery gate failed"
}

parser_self_test || die_i 'parser self-test failed'
printf 'PASS\n' >"$OUT/parser-self-test.txt"
[[ -n "$DRIVER_IMAGE" && -f "$DRIVER_IMAGE" ]] || die_i 'DRIVER_IMAGE must name the exact test driver'

note "suite phase=preconditions runs=$RUNS evidence=$OUT"
DEVICES="$($ADB_BIN devices 2>&1)" || die_i 'adb devices failed'
ROWS="$(printf '%s\n' "$DEVICES" | awk 'NR>1 && NF>=2 {print $1 " " $2}')"
[[ "$(printf '%s\n' "$ROWS" | awk 'NF{n++} END{print n+0}')" == 1 &&
   "$(printf '%s\n' "$ROWS" | awk '$2=="device"{n++} END{print n+0}')" == 1 ]] ||
    die_i 'need exactly one authorized adb device'
SERIAL="$(printf '%s\n' "$ROWS" | awk '$2=="device"{print $1}')"
[[ -z "$EXPECTED_SERIAL" || "$SERIAL" == "$EXPECTED_SERIAL" ]] || die_i "connected serial $SERIAL does not match EXPECTED_SERIAL"
need_online
read_pid_raw || die_i 'ambiguous pidof output at precondition'
[[ -z "$PID_VALUE" ]] || die_i "package already running as PID $PID_VALUE"

adb shell pm path "$PKG" >"$OUT/package-path.txt" 2>&1 || die_i "$PKG is not installed"
PACKAGE_PATH="$(sed -n 's/^package://p' "$OUT/package-path.txt" | tr -d '\r')"
[[ -n "$PACKAGE_PATH" && "$(printf '%s\n' "$PACKAGE_PATH" | wc -l | awk '{print $1}')" == 1 ]] || die_i "$PKG has no unique base APK"
[[ -f "$APK" ]] || die_i "local APK missing: $APK"
APK_SHA="$(sha_host "$APK")" || die_i 'no SHA-256 utility'
INSTALLED_APK_SHA="$(adb shell "toybox sha256sum '$PACKAGE_PATH'" 2>/dev/null | tr -d '\r' | awk '{print $1}')"
[[ "$INSTALLED_APK_SHA" == "$APK_SHA" ]] || die_i 'installed APK does not match local checkpoint build'
DRIVER_IMAGE_SHA="$(sha_host "$DRIVER_IMAGE")" || die_i 'cannot hash DRIVER_IMAGE'
DRIVER_SHA="$(adb shell "toybox sha256sum '$DRIVER' 2>/dev/null || sha256sum '$DRIVER' 2>/dev/null" | tr -d '\r' | awk '{print $1}')"
[[ "$DRIVER_SHA" == "$DRIVER_IMAGE_SHA" ]] || die_i 'deployed driver does not match DRIVER_IMAGE'
PRIVATE_DRIVER_SHA="$(adb shell run-as "$PKG" toybox sha256sum files/libvulkan_panfrost.so 2>/dev/null | tr -d '\r' | awk '{print $1}' || true)"
KLIST="$(adb shell "for d in /sys/devices/platform/*.mali/kprcs; do [ -d \"\$d\" ] && [ -r \"\$d\" ] && echo \"\$d\"; done" 2>/dev/null | tr -d '\r')"
[[ "$(printf '%s\n' "$KLIST" | awk 'NF{n++} END{print n+0}')" == 1 ]] || die_i 'expected one readable Mali kprcs directory'
KPRCS="$(printf '%s\n' "$KLIST" | awk 'NF{print;exit}')"
BOOT0="$(adb shell cat /proc/sys/kernel/random/boot_id | tr -d '\r')"
[[ -n "$BOOT0" ]] || die_i 'missing boot ID'
PRODUCT="$(adb shell getprop ro.product.device | tr -d '\r')"
FINGERPRINT="$(adb shell getprop ro.build.fingerprint | tr -d '\r')"
FINGERPRINT_SHA="$(printf '%s' "$FINGERPRINT" | { if command -v shasum >/dev/null; then shasum -a 256; else sha256sum; fi; } | awk '{print $1}')"
printf 'serial=%s\nproduct=%s\nbuild_fingerprint_sha256=%s\nboot_id=%s\nruns=%s\nkprcs=%s\napk=%s\napk_sha256=%s\ninstalled_apk=%s\ninstalled_apk_sha256=%s\ndriver_image=%s\ndriver_image_sha256=%s\ndeployed_driver=%s\ndeployed_driver_sha256=%s\nprivate_driver_sha256=%s\n' \
    "$SERIAL" "$PRODUCT" "$FINGERPRINT_SHA" "$BOOT0" "$RUNS" "$KPRCS" "$APK" "$APK_SHA" \
    "$PACKAGE_PATH" "$INSTALLED_APK_SHA" "$DRIVER_IMAGE" "$DRIVER_IMAGE_SHA" "$DRIVER" "$DRIVER_SHA" \
    "${PRIVATE_DRIVER_SHA:-UNAVAILABLE}" >"$OUT/metadata.txt"

adb logcat -b all -c || die_i 'cannot clear logcat'
: >"$LOGCAT"
"$ADB_BIN" -s "$SERIAL" logcat -b all -v epoch >"$LOGCAT" 2>&1 &
LOGCAT_PID=$!
sleep 0.3
need_logcat

for ((rep=1; rep<=RUNS; rep++)); do
    run="m1_death_${rep}_$(date +%s)"
    prefix="$OUT/$run"
    read_pid_raw || die_i "$run has ambiguous pidof output before launch"
    [[ -z "$PID_VALUE" ]] || die_f "$run began with package PID $PID_VALUE"
    note "run=$run phase=fault-launch"
    LAUNCH_PENDING=1
    adb shell am start -n "$COMPONENT" --es panvk_jm_fault "$run:hold-after-submit" >"$prefix-am-start.txt" 2>&1 ||
        { need_online; die_i "$run am start result is unavailable"; }
    track_new_generation
    fault_pid="$TRACKED_PID"
    fault_ticks="$TRACKED_TICKS"
    printf 'pid=%s\nstart_ticks=%s\n' "$fault_pid" "$fault_ticks" >"$prefix-process.txt"
    verify_hold_evidence "$run" "$prefix" "$fault_pid" "$fault_ticks"
    force_stop_generation "$fault_pid" "$fault_ticks" "$prefix"
    verify_no_post_kill_records "$run" "$prefix" "$fault_pid"

    recovery="$prefix-recovery"
    note "run=$run phase=cold-recovery"
    LAUNCH_PENDING=1
    adb shell am start -n "$COMPONENT" >"$recovery-am-start.txt" 2>&1 ||
        { need_online; die_i "$run recovery start result is unavailable"; }
    track_new_generation
    recovery_pid="$TRACKED_PID"
    recovery_ticks="$TRACKED_TICKS"
    [[ "$recovery_pid" != "$fault_pid" || "$recovery_ticks" != "$fault_ticks" ]] || die_f "$run recovery reused the killed generation"
    printf 'pid=%s\nstart_ticks=%s\n' "$recovery_pid" "$recovery_ticks" >"$recovery-process.txt"
    verify_recovery "$recovery" "$recovery_pid" "$recovery_ticks"
    force_stop_generation "$recovery_pid" "$recovery_ticks" "$recovery"
    boot_now="$(adb shell cat /proc/sys/kernel/random/boot_id 2>/dev/null | tr -d '\r')" ||
        { need_online; die_i "$run cannot read ending boot ID"; }
    [[ -n "$boot_now" ]] || die_i "$run ending boot ID is unavailable"
    [[ "$boot_now" == "$BOOT0" ]] || die_f "$run device rebooted"
    COMPLETED=$((COMPLETED + 1))
    note "run=$run phase=passed"
done

FAULTS="$OUT/fault-scan.txt"
need_logcat
grep -Ei "Fatal signal|ANR in $PKG|>>> $PKG <<<|Mali[^[:cntrl:]]*(MMU fault|GPU reset|job fault|hang)|kbase[^[:cntrl:]]*(MMU fault|GPU reset|job fault|hang)" "$LOGCAT" >"$FAULTS" || true
[[ ! -s "$FAULTS" ]] || die_f 'fault scan matched a GPU fault, fatal signal, tombstone, or ANR'
printf 'clean: no configured fault patterns matched the captured interval\n' >"$FAULTS"
RESULT=PASS
REASON="$COMPLETED process-death runs and cold recoveries passed"
note "PASS runs=$COMPLETED"
