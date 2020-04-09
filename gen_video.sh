#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: gen_video.sh /path/to/samples.npy"
    exit 1
fi
# Generate samples.
samples=$1
basename=$(basename -- "$samples")
basename="${basename%.*}"
# Delete list.txt, if it exists.
rm -f list.txt
# Number of frames per process.
nframes=1800
# Loop over and start 20 processes in background.
# The number of processes can be changed depending on the machine.
for (( k = 0; k < 30; ++k )); do
  a=$(( nframes*k ))  
  python make_video.py --samples $samples --start $a --n $nframes &
  # Save PID to wait on it later.
  pids[${k}]=$!
  temp[${k}]="$basename$a.mp4"
  # Write the filename to list.txt. 
  # This file will be used by ffmpeg to combine the videos.
  echo "file '$basename$a.mp4'" >> list.txt
done
# Wait for all processes to finish.
for pid in ${pids[*]}; do
    wait $pid
done
# Combine the videos using ffmpeg.
ffmpeg -y -f concat -safe 0 -i list.txt -c copy "$basename.mp4"
# Delete all the temporary mp4 files.
for tm in ${temp[*]}; do
    rm -f $tm
done
rm -f list.txt
