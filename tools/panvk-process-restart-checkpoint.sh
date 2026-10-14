#!/usr/bin/env bash
set -euo pipefail

PKG=com.example.adrenotoolstest2
COMPONENT="$PKG/.MainActivity"
ADB_BIN="${ADB:-adb}"
RUNS="${RUNS:-10}"
ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
APK="$ROOT/app/build/outputs/apk/debug/app-debug.apk"
DRIVER="/storage/emulated/0/Android/data/$PKG/files/libvulkan_panfrost.so"
OUT="${EVIDENCE_DIR:-/tmp/tensor-g1-process-teardown-$(date +%Y%m%d-%H%M%S)}"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf '%s\n' 'Usage: [ADB=adb] [RUNS=10] [EVIDENCE_DIR=/tmp/...] panvk-process-restart-checkpoint.sh'
    exit 0
fi
[[ $# -eq 0 && "$RUNS" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid arguments or RUNS\n' >&2; exit 2; }

umask 077
mkdir -p "$OUT"
cp "$0" "$OUT/harness.sh"
SUP="$OUT/supervisor.log"; LOGCAT="$OUT/logcat.txt"; SUMMARY="$OUT/summary.txt"
: >"$SUP"; : >"$OUT/seen-pids.txt"
RESULT=INCONCLUSIVE; REASON='unexpected script exit'; SERIAL=; BOOT0=; KPRCS=
LOGCAT_PID=; COMPLETED=0; FINAL_PID=; PID_VALUE=; START_TICKS=; METRIC=

stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
note() { printf '%s PanvkHarness %s\n' "$(stamp)" "$*" | tee -a "$SUP"; }
adb() { "$ADB_BIN" -s "$SERIAL" "$@"; }
online() { [[ "$(adb get-state 2>/dev/null || true)" == device ]]; }
die_i() { RESULT=INCONCLUSIVE; REASON="$*"; note "INCONCLUSIVE $*"; exit 2; }
die_f() { RESULT=FAIL; REASON="$*"; note "FAIL $*"; exit 1; }
need_online() { online || die_i 'adb device disconnected'; }
need_logcat() {
    if [[ -z "$LOGCAT_PID" ]] || ! kill -0 "$LOGCAT_PID" 2>/dev/null; then die_i 'continuous logcat capture died'; fi
}

sha_host() {
    if command -v shasum >/dev/null; then shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null; then sha256sum "$1" | awk '{print $1}'
    else return 1; fi
}

finish() {
    local rc=$? boot1=UNAVAILABLE list
    trap - EXIT INT TERM; set +e
    list="$(mktemp /tmp/panvk-sha-files.XXXXXX)"
    [[ -z "$LOGCAT_PID" ]] || { kill "$LOGCAT_PID" 2>/dev/null; wait "$LOGCAT_PID" 2>/dev/null; }
    [[ -z "$SERIAL" ]] || boot1="$(adb shell cat /proc/sys/kernel/random/boot_id 2>/dev/null | tr -d '\r')"
    [[ -n "$boot1" ]] || boot1=UNAVAILABLE
    {
        printf 'result=%s\nreason=%s\nexit_code=%s\n' "$RESULT" "$REASON" "$rc"
        printf 'completed_cycles=%s\nrequested_cycles=%s\nfinal_pid=%s\n' "$COMPLETED" "$RUNS" "${FINAL_PID:-NONE}"
        printf 'boot_id_start=%s\nboot_id_end=%s\n' "${BOOT0:-UNAVAILABLE}" "$boot1"
    } >"$SUMMARY"
    find "$OUT" -type f ! -name SHA256SUMS | LC_ALL=C sort >"$list"
    (cd "$OUT" || exit; while IFS= read -r f; do
        f=".${f#"$OUT"}"; if command -v shasum >/dev/null; then shasum -a 256 "$f"; else sha256sum "$f"; fi
    done <"$list" >SHA256SUMS)
    rm -f "$list"
    exit "$rc"
}
trap finish EXIT
trap 'die_i "interrupted by signal"' INT TERM

# pidof -s is broken on the target Android build. Plain pidof must return zero
# or one numeric field; nonzero with a live adb transport means "not running".
read_pid() {
    local text rc words
    set +e; text="$(adb shell pidof "$PKG" 2>/dev/null)"; rc=$?; set -e
    text="$(printf '%s' "$text" | tr -d '\r' | awk '{$1=$1;print}')"
    if [[ $rc -ne 0 ]]; then need_online; PID_VALUE=; return; fi
    [[ -n "$text" ]] || { PID_VALUE=; return; }
    read -r -a words <<<"$text"
    [[ ${#words[@]} -eq 1 && "${words[0]}" =~ ^[0-9]+$ ]] || die_i "ambiguous pidof output: $text"
    PID_VALUE="${words[0]}"
}

read_ticks() {
    local text rc
    set +e; text="$(adb shell cat "/proc/$1/stat" 2>/dev/null)"; rc=$?; set -e
    [[ $rc -eq 0 ]] || { need_online; die_i "cannot read /proc/$1/stat"; }
    START_TICKS="$(printf '%s\n' "$text" | tr -d '\r' | awk '{print $22}')"
    [[ "$START_TICKS" =~ ^[0-9]+$ ]] || die_i "malformed /proc/$1/stat"
}

read_metric() {
    local text rc
    set +e; text="$(adb shell "test -r '$1' && cat '$1'" 2>/dev/null)"; rc=$?; set -e
    text="$(printf '%s' "$text" | tr -d '\r' | awk '{$1=$1;print}')"
    [[ $rc -eq 0 && "$text" =~ ^[0-9]+$ ]] || { need_online; die_i "missing metric: $1"; }
    METRIC="$text"
}

marker() {
    need_logcat
    adb shell log -t PanvkHarness "$*" >/dev/null 2>&1 || { need_online; die_i "cannot write marker: $*"; }
    note "$*"
}
pid_log() { need_logcat; awk -v p="$1" '$2 == p' "$LOGCAT" >"$2"; }

trace_gate() {
    local line="$1" where="$2" a f e
    a="$(printf '%s\n' "$line" | sed -n 's/.*atom_allocated=\([0-9][0-9]*\).*/\1/p')"
    f="$(printf '%s\n' "$line" | sed -n 's/.*atom_freed=\([0-9][0-9]*\).*/\1/p')"
    e="$(printf '%s\n' "$line" | sed -n 's/.*failed_events=\([0-9][0-9]*\).*/\1/p')"
    [[ -n "$a" && -n "$f" && -n "$e" ]] || die_i "$where malformed kbase trace"
    [[ "$e" == 0 ]] || die_f "$where failed_events=$e"
    [[ "$a" == "$f" ]] || die_f "$where unbalanced atoms allocated=$a freed=$f"
}

diagnostics() {
    local file="$1" phase="$2" global mem rc
    set +e
    global="$(adb shell "cat '${KPRCS%/kprcs}/total_gpu_mem'" 2>/dev/null)"; rc=$?
    [[ $rc -eq 0 ]] || { online || { set -e; die_i 'disconnect during GPU diagnostics'; }; global=UNAVAILABLE; }
    mem="$(adb shell cat /proc/meminfo 2>/dev/null)"; rc=$?
    [[ $rc -eq 0 ]] || { online || { set -e; die_i 'disconnect during memory diagnostics'; }; mem=UNAVAILABLE; }
    set -e
    { printf 'phase=%s timestamp=%s global_total_gpu_mem=%s\n--- /proc/meminfo ---\n' "$phase" "$(stamp)" "$(printf '%s' "$global" | tr -d '\r')";
      printf '%s\n' "$(printf '%s' "$mem" | tr -d '\r')"; } >>"$file"
}

record_counters() {
    local n="$1" pid="$2" file
    file="$OUT/launch-$(printf '%02d' "$1")-process-gpu-mem.txt"
    read_metric "$KPRCS/$pid/total_gpu_mem"; printf 'total_gpu_mem=%s\n' "$METRIC" >"$file"
    read_metric "$KPRCS/$pid/dma_buf_gpu_mem"; printf 'dma_buf_gpu_mem=%s\n' "$METRIC" >>"$file"
}

launch() {
    local n="$1" deadline pid log missing m trace prefix
    prefix="$OUT/launch-$(printf '%02d' "$1")"
    local marks=(
      'buffer-copy[panvk]: PASS' 'compute[panvk]: PASS' 'lifecycle-churn[panvk]: SUMMARY result=PASS_STABLE'
      'jm-async[panvk]: PASS mode=waves' 'jm-async[panvk]: PASS mode=single-submit-chain'
      'jm-async[panvk]: PASS mode=binary-semaphore' 'image-transfer[panvk]: PASS' 'sample[panvk]: PASS'
      'indexed[panvk]: PASS' 'blend[panvk]: PASS' 'depth[panvk]: PASS' 'ready extent=' 'first frame presented'
    )
    read_pid; [[ -z "$PID_VALUE" ]] || die_f "launch $n began with package PID $PID_VALUE"
    marker "launch=$n phase=start"; deadline=$(( $(date +%s) + 30 ))
    adb shell am start -W -n "$COMPONENT" >"$prefix-am-start.txt" 2>&1 || { need_online; die_f "launch $n am start failed"; }
    pid=
    while [[ $(date +%s) -le $deadline ]]; do need_online; need_logcat; read_pid; [[ -z "$PID_VALUE" ]] || { pid="$PID_VALUE"; break; }; sleep 0.1; done
    [[ -n "$pid" ]] || die_i "launch $n produced no PID within 30s"
    grep -Fxq "$pid" "$OUT/seen-pids.txt" && die_i "launch $n reused PID $pid"
    printf '%s\n' "$pid" >>"$OUT/seen-pids.txt"; printf '%s\n' "$pid" >"$prefix-pid.txt"
    read_ticks "$pid"; printf '%s\n' "$START_TICKS" >"$prefix-start-ticks.txt"
    LAUNCH_PID="$pid"; LAUNCH_TICKS="$START_TICKS"; log="$prefix-pid-log.txt"
    while [[ $(date +%s) -le $deadline ]]; do
        need_online; need_logcat; read_pid; [[ "$PID_VALUE" == "$pid" ]] || die_i "launch $n PID vanished or changed"
        read_ticks "$pid"; [[ "$START_TICKS" == "$LAUNCH_TICKS" ]] || die_i "launch $n PID generation changed"
        pid_log "$pid" "$log"
        grep -Eq '(buffer-copy|compute|lifecycle-churn|jm-async|image-transfer|sample|indexed|blend|depth)\[panvk\]: FAIL|VK_ERROR_DEVICE_LOST' "$log" && die_f "launch $n structured gate failed"
        grep -Eq '\[panvk\]: INCONCLUSIVE|presenter containment.*INCONCLUSIVE' "$log" && die_i "launch $n evidence is inconclusive"
        missing=; for m in "${marks[@]}"; do grep -Fq -- "$m" "$log" || { missing="$m"; break; }; done
        trace="$(grep -F 'panvk: kbase trace atom_allocated=' "$log" | tail -n 1 || true)"
        if [[ -z "$missing" && -n "$trace" ]]; then
            trace_gate "$trace" "launch $n"; printf '%s\n' "$trace" >"$prefix-final-kbase-trace.txt"
            marker "launch=$n pid=$pid start_ticks=$LAUNCH_TICKS phase=gates-passed"; return
        fi
        sleep 0.2
    done
    [[ -n "$trace" ]] || missing="${missing:+$missing, }final kbase trace"; die_i "launch $n timed out waiting for $missing"
}

home_quiesce() {
    local n="$1" pid="$2" ticks="$3" prefix launchlog base deadline phase trace active_total total dma stable='' stable_n=0
    prefix="$OUT/cycle-$(printf '%02d' "$1")"
    launchlog="$OUT/launch-$(printf '%02d' "$1")-pid-log.txt"
    record_counters "$n" "$pid"; read_metric "$KPRCS/$pid/total_gpu_mem"; active_total="$METRIC"; pid_log "$pid" "$launchlog"
    base="$(wc -l <"$launchlog" | awk '{print $1}')"; [[ "$base" =~ ^[0-9]+$ ]] || die_i "cycle $n cannot delimit HOME logs"
    marker "cycle=$n pid=$pid phase=home-request"
    adb shell input keyevent KEYCODE_HOME >"$prefix-home-command.txt" 2>&1 || { need_online; die_f "cycle $n HOME failed"; }
    deadline=$(( $(date +%s) + 30 )); : >"$prefix-home-counters.txt"; phase="$prefix-home-log.txt"
    while [[ $(date +%s) -le $deadline ]]; do
        need_online; need_logcat; read_pid; [[ "$PID_VALUE" == "$pid" ]] || die_i "cycle $n process died across HOME"
        read_ticks "$pid"; [[ "$START_TICKS" == "$ticks" ]] || die_i "cycle $n PID generation changed across HOME"
        pid_log "$pid" "$launchlog"; tail -n "+$((base + 1))" "$launchlog" >"$phase"
        trace=; grep -Fq APP_CMD_TERM_WINDOW "$phase" && trace="$(awk 'seen {print} /APP_CMD_TERM_WINDOW/ {seen=1}' "$phase" | grep -F 'panvk: kbase trace atom_allocated=' | tail -n 1 || true)"
        read_metric "$KPRCS/$pid/total_gpu_mem"; total="$METRIC"; read_metric "$KPRCS/$pid/dma_buf_gpu_mem"; dma="$METRIC"
        printf 'timestamp=%s total_gpu_mem=%s dma_buf_gpu_mem=%s\n' "$(stamp)" "$total" "$dma" >>"$prefix-home-counters.txt"
        if [[ "$total" == "$stable" ]]; then stable_n=$((stable_n + 1)); else stable="$total"; stable_n=1; fi
        if [[ -n "$trace" && "$dma" == 0 && "$total" -lt "$active_total" && $stable_n -ge 3 ]]; then
            trace_gate "$trace" "cycle $n post-TERM_WINDOW"; printf '%s\n' "$trace" >"$prefix-post-term-final-kbase-trace.txt"
            printf 'pid=%s\nstart_ticks=%s\nactive_total_gpu_mem=%s\nsettled_total_gpu_mem=%s\ndma_buf_gpu_mem=%s\nstable_samples=%s\n' "$pid" "$ticks" "$active_total" "$total" "$dma" "$stable_n" >"$prefix-quiescent-state.txt"
            marker "cycle=$n pid=$pid phase=home-quiescence-passed total_gpu_mem=$total dma_buf_gpu_mem=$dma"; return
        fi
        sleep 0.2
    done
    grep -Fq APP_CMD_TERM_WINDOW "$phase" || die_i "cycle $n missing APP_CMD_TERM_WINDOW"
    [[ -n "$trace" ]] || die_i "cycle $n missing post-TERM_WINDOW kbase trace"
    [[ "$dma" == 0 ]] || die_f "cycle $n dma_buf_gpu_mem did not reach zero"; die_f "cycle $n total_gpu_mem did not settle"
}

stop_cycle() {
    local n="$1" old="$2" ticks="$3" prefix deadline pstate=PRESENT proc=PRESENT kp=PRESENT now rc stat exit0
    prefix="$OUT/cycle-$(printf '%02d' "$1")"
    diagnostics "$prefix-resource-diagnostics.txt" before-force-stop; marker "cycle=$n pid=$old start_ticks=$ticks phase=force-stop"
    adb shell am force-stop "$PKG" >"$prefix-force-stop.txt" 2>&1 || { need_online; die_f "cycle $n force-stop failed"; }
    deadline=$(( $(date +%s) + 10 ))
    while [[ $(date +%s) -le $deadline ]]; do
        need_online; need_logcat; read_pid; [[ -z "$PID_VALUE" ]] && pstate=ABSENT || pstate="PRESENT:$PID_VALUE"
        set +e; stat="$(adb shell "if [ -r /proc/$old/stat ]; then cat /proc/$old/stat; else echo ABSENT; fi" 2>/dev/null | tr -d '\r')"; rc=$?; set -e
        [[ $rc -eq 0 ]] || { need_online; die_i "cycle $n cannot inspect old /proc"; }
        if [[ "$stat" == ABSENT ]]; then now=ABSENT; else now="$(printf '%s\n' "$stat" | awk '{print $22}')"; fi
        if [[ "$now" == ABSENT ]]; then proc=ABSENT
        elif [[ "$now" =~ ^[0-9]+$ && "$now" != "$ticks" ]]; then die_i "cycle $n PID $old was reused"
        elif [[ "$now" == "$ticks" ]]; then proc=PRESENT_SAME_GENERATION
        else die_i "cycle $n malformed /proc evidence"; fi
        set +e; kp="$(adb shell "if [ -e '$KPRCS/$old' ]; then echo PRESENT; else echo ABSENT; fi" 2>/dev/null | tr -d '\r')"; rc=$?; set -e
        [[ $rc -eq 0 && ( "$kp" == PRESENT || "$kp" == ABSENT ) ]] || { need_online; die_i "cycle $n cannot inspect kprcs cleanup"; }
        printf 'timestamp=%s package=%s proc=%s kprcs=%s\n' "$(stamp)" "$pstate" "$proc" "$kp" >>"$prefix-cleanup-poll.txt"
        [[ "$pstate" == ABSENT && "$proc" == ABSENT && "$kp" == ABSENT ]] && break; sleep 0.1
    done
    [[ "$pstate" == ABSENT ]] || die_f "cycle $n package PID survived force-stop"
    [[ "$proc" == ABSENT ]] || die_f "cycle $n old /proc generation survived force-stop"
    [[ "$kp" == ABSENT ]] || die_f "cycle $n kprcs/$old survived force-stop"
    deadline=$(( $(date +%s) + 5 )); exit0=
    while [[ $(date +%s) -le $deadline ]]; do
        adb shell dumpsys activity exit-info "$PKG" >"$prefix-exit-info.txt" 2>&1 || { need_online; die_i "cycle $n missing exit-info"; }
        exit0="$(awk '/ApplicationExitInfo #0:/{take=1; next} /ApplicationExitInfo #1:/{exit} take{print}' "$prefix-exit-info.txt")"
        if grep -Eq "[[:space:]]pid=$old([[:space:]]|$)" <<<"$exit0" &&
           grep -Fq 'reason=10 (USER REQUESTED) subreason=21 (FORCE STOP) status=0' <<<"$exit0"; then break; fi
        sleep 0.1
    done
    [[ -n "$exit0" ]] || die_i "cycle $n empty newest exit-info record"
    grep -Eq "[[:space:]]pid=$old([[:space:]]|$)" <<<"$exit0" || die_f "cycle $n newest exit-info PID does not match $old"
    grep -Fq 'reason=10 (USER REQUESTED) subreason=21 (FORCE STOP) status=0' <<<"$exit0" || die_f "cycle $n newest exit-info is not a successful force-stop"
    sleep 1
    diagnostics "$prefix-resource-diagnostics.txt" after-settle; marker "cycle=$n old_pid=$old phase=cleanup-passed"
}

note "suite phase=preconditions runs=$RUNS evidence=$OUT"
set +e; DEVICES="$($ADB_BIN devices 2>&1)"; rc=$?; set -e; [[ $rc -eq 0 ]] || die_i 'adb devices failed'
ROWS="$(printf '%s\n' "$DEVICES" | awk 'NR>1 && NF>=2 {print $1 " " $2}')"
[[ "$(printf '%s\n' "$ROWS" | awk 'NF{n++} END{print n+0}')" == 1 && "$(printf '%s\n' "$ROWS" | awk '$2=="device"{n++} END{print n+0}')" == 1 ]] || die_i 'need exactly one authorized adb device'
SERIAL="$(printf '%s\n' "$ROWS" | awk '$2=="device"{print $1}')"; need_online
adb shell pm path "$PKG" >"$OUT/package-path.txt" 2>&1 || die_i "$PKG is not installed"
PACKAGE_PATH="$(sed -n 's/^package://p' "$OUT/package-path.txt" | tr -d '\r')"; [[ -n "$PACKAGE_PATH" && "$(printf '%s\n' "$PACKAGE_PATH" | wc -l | awk '{print $1}')" == 1 ]] || die_i "$PKG has no unique base APK"
[[ -f "$APK" ]] || die_i "local APK missing: $APK"; APK_SHA="$(sha_host "$APK")" || die_i 'no SHA-256 utility'; HARNESS_SHA="$(sha_host "$OUT/harness.sh")"
set +e; ILINE="$(adb shell "toybox sha256sum '$PACKAGE_PATH'" 2>/dev/null | tr -d '\r')"; rc=$?; set -e
[[ $rc -eq 0 ]] || { need_online; die_i "cannot hash installed APK: $PACKAGE_PATH"; }; INSTALLED_APK_SHA="$(printf '%s\n' "$ILINE" | awk '{print $1}')"
[[ "$INSTALLED_APK_SHA" =~ ^[0-9a-fA-F]{64}$ ]] || die_i 'malformed installed APK SHA-256'; [[ "$INSTALLED_APK_SHA" == "$APK_SHA" ]] || die_i 'installed APK does not match local checkpoint build'
set +e; DLINE="$(adb shell "toybox sha256sum '$DRIVER' 2>/dev/null || sha256sum '$DRIVER' 2>/dev/null" | tr -d '\r')"; rc=$?; set -e
[[ $rc -eq 0 ]] || { need_online; die_i "driver missing: $DRIVER"; }; DSHA="$(printf '%s\n' "$DLINE" | awk '{print $1}')"; [[ "$DSHA" =~ ^[0-9a-fA-F]{64}$ ]] || die_i 'malformed driver SHA-256'
set +e; KLIST="$(adb shell "for d in /sys/devices/platform/*.mali/kprcs; do [ -d \"\$d\" ] && [ -r \"\$d\" ] && echo \"\$d\"; done" 2>/dev/null | tr -d '\r')"; rc=$?; set -e
[[ $rc -eq 0 ]] || { need_online; die_i 'cannot discover kprcs'; }; [[ "$(printf '%s\n' "$KLIST" | awk 'NF{n++} END{print n+0}')" == 1 ]] || die_i 'expected one readable Mali kprcs directory'
KPRCS="$(printf '%s\n' "$KLIST" | awk 'NF{print;exit}')"; BOOT0="$(adb shell cat /proc/sys/kernel/random/boot_id | tr -d '\r')"; [[ -n "$BOOT0" ]] || die_i 'missing boot ID'
FINGERPRINT="$(adb shell getprop ro.build.fingerprint | tr -d '\r')"; KERNEL="$(adb shell uname -a | tr -d '\r')"; COMMIT="$(git -C "$ROOT" rev-parse HEAD)" || die_i 'missing git commit'; git -C "$ROOT" status --short >"$OUT/git-status.txt"
printf 'serial=%s\nboot_id=%s\nfingerprint=%s\nkernel=%s\ngit_commit=%s\nharness_sha256=%s\napk=%s\napk_sha256=%s\ninstalled_apk=%s\ninstalled_apk_sha256=%s\ndevice_driver=%s\ndevice_driver_sha256=%s\ncounted_cycles=%s\nkprcs=%s\n' \
  "$SERIAL" "$BOOT0" "$FINGERPRINT" "$KERNEL" "$COMMIT" "$HARNESS_SHA" "$APK" "$APK_SHA" "$PACKAGE_PATH" "$INSTALLED_APK_SHA" "$DRIVER" "$DSHA" "$RUNS" "$KPRCS" >"$OUT/metadata.txt"

adb shell am force-stop "$PKG" >"$OUT/precondition-force-stop.txt" 2>&1 || die_i 'initial force-stop failed'
deadline=$(( $(date +%s) + 10 )); while [[ $(date +%s) -le $deadline ]]; do read_pid; [[ -z "$PID_VALUE" ]] && break; sleep 0.1; done
[[ -z "$PID_VALUE" ]] || die_f 'initial package process survived force-stop'; adb logcat -b all -c || die_i 'cannot clear logcat'
: >"$LOGCAT"; "$ADB_BIN" -s "$SERIAL" logcat -b all -v epoch >"$LOGCAT" 2>&1 & LOGCAT_PID=$!; sleep 0.3; need_logcat; marker "suite phase=capture-start boot_id=$BOOT0"

for ((n=1; n<=RUNS; n++)); do
    launch "$n"; home_quiesce "$n" "$LAUNCH_PID" "$LAUNCH_TICKS"; stop_cycle "$n" "$LAUNCH_PID" "$LAUNCH_TICKS"; COMPLETED="$n"
done
launch "$((RUNS+1))"; FINAL_PID="$LAUNCH_PID"; record_counters "$((RUNS+1))" "$FINAL_PID"; diagnostics "$OUT/final-resource-diagnostics.txt" final-recovery-running
BOOT1="$(adb shell cat /proc/sys/kernel/random/boot_id 2>/dev/null | tr -d '\r')"; [[ -n "$BOOT1" ]] || die_i 'missing final boot ID'; [[ "$BOOT1" == "$BOOT0" ]] || die_i 'device rebooted during suite'
marker "suite phase=final-recovery-passed pid=$FINAL_PID"; sleep 0.2; need_logcat
FAULTS="$OUT/fault-scan.txt"
PID_ALT="$(paste -sd'|' "$OUT/seen-pids.txt")"
grep -Ei "VK_ERROR_DEVICE_LOST|Mali[^[:cntrl:]]*(fault|hang|reset)|kbase[^[:cntrl:]]*(fault|hang|reset)|MMU[^[:cntrl:]]*fault|job[^[:cntrl:]]*fault|GPU[^[:cntrl:]]*(reset|hang)|\([[:space:]]*($PID_ALT)\)[^[:cntrl:]]*Fatal signal|ANR in $PKG|>>> $PKG <<<" "$LOGCAT" >"$FAULTS" || true
awk -v pids="$(paste -sd, "$OUT/seen-pids.txt")" 'BEGIN{n=split(pids,a,","); for(i=1;i<=n;i++) target[a[i]]=1} target[$2] && /Fatal signal/' "$LOGCAT" >>"$FAULTS"
[[ ! -s "$FAULTS" ]] || die_f 'fault scan matched device loss, GPU fault, fatal signal, tombstone, or ANR'; printf 'clean: no configured fault patterns matched the captured interval\n' >"$FAULTS"
RESULT=PASS; REASON="$RUNS HOME-quiesce/force-stop cycles and final recovery passed; final process intentionally left running"; note "PASS cycles=$RUNS final_pid=$FINAL_PID final_process=left-running"
