#!/bin/zsh
# Waits until the ODI (or any match given as $1) has finished, then switches the driver LaunchAgent from shadow to LIVE mode
# (predictions + Fact-o-Meter to Telegram for T20Is). Usage: ops/switch_to_live.sh 172681
set -e
ROOT=/Users/susha/Downloads/cricket_predictor
LOG=$ROOT/data/live/console-$1.log
until grep -aq "MATCH COMPLETE" "$LOG"; do sleep 20; done
cp $ROOT/ops/com.cricketbaba.driver.live.plist ~/Library/LaunchAgents/com.cricketbaba.driver.plist
launchctl bootout gui/$(id -u)/com.cricketbaba.driver || true
sleep 3
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cricketbaba.driver.plist
echo "switched to live at $(date)" >> $ROOT/data/live/switch-to-live.log
