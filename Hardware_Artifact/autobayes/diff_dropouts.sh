for i in {1..8}
do
  echo "$i ResNet18"
  nohup ./diff_dropouts.py $i ResNet18 &
  if [ $i -le 7 ]
  then
    echo "$i LeNet"
    nohup ./diff_dropouts.py $i LeNet &
  fi
  # echo "$i VGG11"
  # nohup ./diff_dropouts.py $i VGG11 &
done