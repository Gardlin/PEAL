
for epoch in $(seq 10 20); do
  for n in $(seq 0 6); do
      python test.py --test_epoch=$epoch --benchmark=$1 --iteration=$n

      python eval.py --test_epoch=$epoch --benchmark=$1 --method=lgr --iteration=$n
#      for k in 250 500 1000 2500 5000; do
#  #      i=0
#  #      len=3
#        for m in $(seq 1 3);  do
#            python eval.py --test_epoch=$epoch --benchmark=$1  --num_corr=$k --method=ransac --iteration=$n
#          done
#      done
  done

done








