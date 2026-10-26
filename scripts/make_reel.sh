#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
FF=$(.venv/Scripts/python.exe -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
OUT=demo/build; mkdir -p $OUT
card() { "$FF" -y -loglevel error -loop 1 -t 3.5 -i "$1" -vf "fade=in:0:12,fade=out:st=3.0:d=0.5,format=yuv420p" -r 30 -c:v libx264 -crf 18 -preset medium "$2"; }
interp() { "$FF" -y -loglevel error $3 -i "$1" -vf "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,format=yuv420p" -c:v libx264 -crf 18 -preset medium "$2"; }
card demo/cards/00_title.png $OUT/c0.mp4
card demo/cards/01_opus.png $OUT/c1.mp4
card demo/cards/02_best.png $OUT/c2.mp4
card demo/cards/03_relay.png $OUT/c3.mp4
interp artifacts/demo3_opus_programs_robot.mp4 $OUT/v1.mp4 ""
interp artifacts/demo2_skills_best_run.mp4 $OUT/v2.mp4 ""
interp artifacts/demo1_relay_teacher.mp4 $OUT/v3.mp4 "-t 30"
printf "file 'c0.mp4'\nfile 'c1.mp4'\nfile 'v1.mp4'\nfile 'c2.mp4'\nfile 'v2.mp4'\nfile 'c3.mp4'\nfile 'v3.mp4'\n" > $OUT/list.txt
"$FF" -y -loglevel error -f concat -safe 0 -i $OUT/list.txt -c:v libx264 -crf 19 -preset medium -r 30 -movflags +faststart demo/demo_reel_30fps.mp4
echo "REEL DONE $(date +%H:%M)"
