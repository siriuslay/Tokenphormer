python train.py --dataset cora --device 0 --batch_size 2000 --dropout 0.1 --hidden_dim 512 \
          --n_heads 1 --n_layers 1 --pe_dim 3 --peak_lr 0.005  --weight_decay=1e-05 \
          --hop_num 3 \
          --t_nums 100 --w_len 4 --uniformRWRate 0.3 --nonBackRWRate 0.05 --nJumpRate 0.6 \
          --seed 0 \
          --sgpm_token