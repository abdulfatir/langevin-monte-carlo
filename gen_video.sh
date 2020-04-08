python ula.py
rm -f list.txt
for (( k = 0; k < 20; ++k )); do
  a=$(( 2500*k ))  
  python make_video.py --start $a --n 2500 &
  pids[${k}]=$!
  temp[${k}]="heart$a.mp4"
  echo "file 'heart$a.mp4'" >> list.txt
done

for pid in ${pids[*]}; do
    wait $pid
done

ffmpeg -y -f concat -safe 0 -i list.txt -c copy heart.mp4

for tm in ${temp[*]}; do
    rm -f $tm
done
