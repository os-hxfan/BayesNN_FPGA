for i in {1..5}
do
  echo "$i LeNet"
  nohup ./diff_scale.py $i LeNet &
done