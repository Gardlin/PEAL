for epoch in $(seq 7 17); do
  for n in $(seq 0 0); do
      python test.py --test_epoch=$epoch --benchmark=$1
      for k in 250 500 1000 2500 5000; do
  #      i=0
  #      len=3
        for m in $(seq 1 5);  do
            python eval_o3d12.py --test_epoch=$epoch --benchmark=$1  --num_corr=$k --method=ransac --iteration=$n
          done
      done
      python eval_o3d12.py --test_epoch=$epoch --benchmark=$1 --method=lgr --iteration=$n

  done
done









