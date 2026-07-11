#!/usr/bin/env bash
# Live monitor for the MsPacman Rainbow ablation (run_ablation.sh).
#
# Every configured run is shown with its state:
#   waiting   queued in the pipeline, not launched yet (no log)
#   running   log is being written to right now
#   FROZEN    was progressing but its log has not advanced in $STALE_SECS s
#   done      finished cleanly (exit 0)
#   FAILED    exited non-zero
#
# Run from hw3/:  ./watch_progress.sh
# One-shot (no live refresh):  ./watch_progress.sh --once
# Tuning:  STALE_SECS=90 ./watch_progress.sh   (freeze threshold, seconds)
cd "$(dirname "$0")"
export STALE_SECS=${STALE_SECS:-45}

# Keep in sync with CONFIGS in run_ablation.sh. Order = display order within a
# status group. "config:scenario" — scenario is the pretty label.
export ABL_RUNS="mspacman_rainbow:rainbow
mspacman_abl_no_double:no_double
mspacman_abl_no_dueling:no_dueling
mspacman_abl_no_distributional:no_distributional
mspacman_abl_no_noisy:no_noisy
mspacman_abl_no_per:no_per
mspacman_abl_no_nd:no_nd
mspacman:dqn"

# Seeds in the sweep. Every config is shown once per seed (7 x 3 = 21 rows).
export ABL_SEEDS=${ABL_SEEDS:-"1 2 3"}

render() {
  local esc reset bold dim red green cyan
  esc=$(printf '\033')
  reset="${esc}[0m"; bold="${esc}[1m"; dim="${esc}[90m"
  red="${esc}[1;31m"; green="${esc}[32m"; cyan="${esc}[36m"
  local BAR_W=12 now hr rows='' line rem_total=0 rates=''
  local n_run=0 n_froz=0 n_wait=0 n_done=0 n_fail=0
  now=$(date +%s)
  hr=$(printf '%.0s─' $(seq 1 138))

  printf '%s%s  MsPacman Rainbow ablation — live monitor%s   %s%s%s\n' \
    "$bold" "$cyan" "$reset" "$dim" "$(date '+%a %H:%M:%S')" "$reset"
  printf '%s%s%s\n' "$dim" "$hr" "$reset"
  printf '%s%-17s %-5s %-8s %-15s %-8s %-10s %-7s %-7s %-7s %-7s %-17s %s%s\n' \
    "$bold" "SCENARIO" "SEED" "STATUS" "STEPS" "IT/S" "ETA" "MEAN_R" "MEAN5" "MAX_R" "LAST_R" "PROGRESS" "EVAL_TREND" "$reset"

  local cfgidx=0
  while IFS=: read -r cfg scenario; do
    [ -z "$cfg" ] && continue
    local sd
    for sd in $ABL_SEEDS; do
      local log="logs/${cfg}_sd${sd}.log" exitf="logs/${cfg}_sd${sd}.exit"
      # legacy fallback: in-flight seed-1 jobs wrote unsuffixed log names
      if [ ! -f "$log" ] && [ "$sd" = "1" ] && [ -f "logs/${cfg}.log" ]; then
        log="logs/${cfg}.log"; exitf="logs/${cfg}.exit"
      fi
      local pct steps rate eta status color key mtime age line
      local filled empty bar i row cur den rmean rroll rmax rlast rspark expd rr beststep bs d

      pct=0; steps='-'; rate='-'; eta='-'
      if [ -f "$log" ]; then
        line=$(tr '\r' '\n' < "$log" | grep -E '^[[:space:]]*[0-9]+%\|' | tail -1)
        if [ -n "$line" ]; then
          pct=$(echo "$line"  | grep -oE '^[[:space:]]*[0-9]+%' | tr -d ' %')
          steps=$(echo "$line" | grep -oE '[0-9]+/[0-9]+')
          rate=$(echo "$line"  | grep -oE '[0-9.]+it/s' | tail -1)
          eta=$(echo "$line"   | grep -oE '<[0-9:]+' | tr -d '<')
        fi
      fi
      [ -z "$pct" ] && pct=0

      # most-progressed run dir for this (config, seed): a completed run here wins
      # over any stray restart/log/exit state, so finished results aren't hidden.
      expd=''; beststep=-1
      for d in exp/MsPacman_${scenario}_sd${sd}_*/; do
        [ -f "${d}log.csv" ] || continue
        bs=$(tail -1 "${d}log.csv" | awk -F, '{printf "%d",$6+0}')
        if [ "${bs:-0}" -gt "$beststep" ]; then beststep=$bs; expd=$d; fi
      done

      if [ "$beststep" -ge 990000 ]; then
        status='done'; color="$green"; key=4; pct=100; steps="${beststep}/1000000"; rate='-'; eta='-'; n_done=$((n_done+1))
      elif [ -f "$exitf" ] && [ "$(cat "$exitf" 2>/dev/null || echo 1)" != "0" ]; then
        status='FAILED'; color="$red"; key=1; n_fail=$((n_fail+1))
      elif [ ! -f "$log" ]; then
        status='waiting'; color="$dim"; key=3; n_wait=$((n_wait+1))
      else
        mtime=$(stat -c %Y "$log" 2>/dev/null || echo "$now")
        age=$((now - mtime))
        if [ "$age" -ge "$STALE_SECS" ]; then
          status='FROZEN'; color="$red"; key=0; eta="stalled ${age}s"; n_froz=$((n_froz+1))
        else
          status='running'; color="$cyan"; key=2; n_run=$((n_run+1))
        fi
      fi

      # returns so far from this run's log.csv: latest mean eval, best max eval,
      # and latest training-episode return (updates far more often than evals).
      # Skip waiting runs so a stale dir from a prior aborted run isn't shown.
      rmean='-'; rroll='-'; rmax='-'; rlast='-'; rspark='-'
      if [ "$status" != 'waiting' ]; then
        if [ -n "$expd" ] && [ -f "${expd}log.csv" ]; then
          # ms=latest eval mean, r5=rolling mean of last 5 evals, xs=best eval max,
          # ts=latest train-episode return, sp=sparkline of the rolling-5 series
          # (last 20 points, normalised per-run so the shape/trend is visible).
          rr=$(awk -F, 'NR>1{if($1!="")a[++n]=$1; if($3!=""&&(!s||$3>x)){x=$3;s=1}; if($7!="")t=$7}
                        END{ms=(n?sprintf("%d",a[n]):"-");
                             if(n){c=0;su=0;for(i=n;i>=1&&c<5;i--){su+=a[i];c++};r5=sprintf("%d",su/c)}else r5="-";
                             xs=(s?sprintf("%d",x):"-"); ts=(t==""?"-":sprintf("%d",t));
                             sp="-";
                             if(n>=2){
                               for(i=1;i<=n;i++){cc=0;ss=0;for(j=i;j>=1&&cc<5;j--){ss+=a[j];cc++};sm[i]=ss/cc}
                               W=20; st=(n>W?n-W+1:1); lo=1e18; hi=-1e18;
                               for(i=st;i<=n;i++){if(sm[i]<lo)lo=sm[i]; if(sm[i]>hi)hi=sm[i]}
                               nb=split("▁ ▂ ▃ ▄ ▅ ▆ ▇ █",B," "); sp=""; rng=hi-lo;
                               for(i=st;i<=n;i++){k=(rng<=0?4:int((sm[i]-lo)/rng*7+0.5)); if(k<0)k=0; if(k>7)k=7; sp=sp B[k+1]}
                             }
                             printf "%s %s %s %s %s",ms,r5,xs,ts,sp}' "${expd}log.csv")
          read rmean rroll rmax rlast rspark <<< "$rr"
        fi
      fi

      # sweep estimate: steps still to run + aggregate it/s of running rows
      if [ "$steps" != '-' ]; then cur=${steps%/*}; den=${steps#*/}; else cur=0; den=1000000; fi
      [ -z "$den" ] && den=1000000
      [ "$status" != 'done' ] && rem_total=$(( rem_total + den - cur ))
      [ "$status" = 'running' ] && [ "$rate" != '-' ] && rates="${rates}${rate%it/s} "

      filled=$(( pct * BAR_W / 100 )); [ "$filled" -gt "$BAR_W" ] && filled=$BAR_W
      empty=$(( BAR_W - filled )); bar=''
      i=0; while [ "$i" -lt "$filled" ]; do bar="${bar}█"; i=$((i+1)); done
      i=0; while [ "$i" -lt "$empty"  ]; do bar="${bar}░"; i=$((i+1)); done

      row=$(printf '%-17s %-5s %-8s %-15s %-8s %-10s %-7s %-7s %-7s %-7s %s %3s%% %s' \
            "$scenario" "sd${sd}" "$status" "$steps" "$rate" "$eta" "$rmean" "$rroll" "$rmax" "$rlast" "$bar" "$pct" "$rspark")
      # Sort in run order: seed-major (seed 1 finishes first), then launch order
      # within a seed (config index). key/status only drive colour + counts now.
      rows+="${sd}${cfgidx}"$'\t'"${color}${row}${reset}"$'\n'
    done
    cfgidx=$((cfgidx+1))
  done <<< "$ABL_RUNS"

  printf '%s' "$rows" | sort -n -k1 | cut -f2-

  printf '%s%s%s\n' "$dim" "$hr" "$reset"
  printf '%swaiting %d%s · %s%srunning %d%s · %sFROZEN %d%s · %sdone %d%s · %sFAILED %d%s   %s(stale after %ss)%s\n' \
    "$dim" "$n_wait" "$reset" \
    "$bold" "$cyan" "$n_run" "$reset" \
    "$red" "$n_froz" "$reset" \
    "$green" "$n_done" "$reset" \
    "$red" "$n_fail" "$reset" \
    "$dim" "$STALE_SECS" "$reset"

  # Whole-sweep ETA: total steps left / current aggregate throughput, projected
  # to a wall-clock finish. Assumes concurrency holds at the current level.
  local agg rem_m eta_sec etah finish
  agg=$(echo $rates | awk '{for(i=1;i<=NF;i++)s+=$i; printf "%.1f", s+0}')
  rem_m=$(awk -v r="$rem_total" 'BEGIN{printf "%.1f", r/1000000}')
  if awk -v a="$agg" 'BEGIN{exit !(a>0)}'; then
    eta_sec=$(awk -v r="$rem_total" -v a="$agg" 'BEGIN{printf "%d", r/a}')
    etah=$(awk -v s="$eta_sec" 'BEGIN{printf "%.1f", s/3600}')
    finish=$(date -d "@$(( now + eta_sec ))" '+%a %d %b %H:%M' 2>/dev/null)
    printf '%ssweep ETA:%s %s~%sh%s at %s it/s aggregate  (%sM steps left)   %s→ done ~%s%s\n' \
      "$bold" "$reset" "$bold" "$etah" "$reset" "$agg" "$rem_m" "$dim" "$finish" "$reset"
  else
    printf '%ssweep ETA:%s %s(no runs active — %sM steps left)%s\n' \
      "$bold" "$reset" "$dim" "$rem_m" "$reset"
  fi
}

if [ "${1:-}" = "--once" ]; then
  render
  exit 0
fi

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
exec watch -c -t -n 2 "bash '$SELF' --once"
