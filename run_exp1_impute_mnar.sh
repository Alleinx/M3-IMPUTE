python train_mdi.py --epochs 40000 --repeat_exp_num 5 --init_epsilon 1e-4 --apply_attr --apply_peer --sample_peer_size 5 --sample_strategy cos-similarity --node_dim 128 --edge_dim 128 --impute_hidden 128 --log_dir EXP1_impute_mnar --corrupt mnar --known 0.5 --mar_rate_obs 0.5 --mar_rate_missing 0.42 --mnar_known_mask 0.875 uci --train_edge 0.7 --data yacht
python train_mdi.py --epochs 40000 --repeat_exp_num 5 --init_epsilon 1e-4 --apply_attr --apply_peer --sample_peer_size 5 --sample_strategy cos-similarity --node_dim 128 --edge_dim 128 --impute_hidden 128 --log_dir EXP1_impute_mnar --corrupt mnar --known 0.5 --mar_rate_obs 0.5 --mar_rate_missing 0.37 --mnar_known_mask 0.875 uci --train_edge 0.7 --data wine
python train_mdi.py --epochs 40000 --repeat_exp_num 5 --init_epsilon 1e-4 --apply_attr --apply_peer --sample_peer_size 5 --sample_strategy cos-similarity --node_dim 128 --edge_dim 128 --impute_hidden 128 --log_dir EXP1_impute_mnar --corrupt mnar --known 0.5 --mar_rate_obs 0.5 --mar_rate_missing 0.36 --mnar_known_mask 0.875 uci --train_edge 0.7 --data housing
python train_mdi.py --epochs 40000 --repeat_exp_num 5 --init_epsilon 1e-4 --apply_attr --apply_peer --sample_peer_size 5 --sample_strategy cos-similarity --node_dim 128 --edge_dim 128 --impute_hidden 128 --log_dir EXP1_impute_mnar --corrupt mnar --known 0.5 --mar_rate_obs 0.5 --mar_rate_missing 0.38 --mnar_known_mask 0.875 uci --train_edge 0.7 --data concrete
python train_mdi.py --epochs 40000 --repeat_exp_num 5 --init_epsilon 1e-4 --apply_attr --apply_peer --sample_peer_size 5 --sample_strategy cos-similarity --node_dim 128 --edge_dim 128 --impute_hidden 128 --log_dir EXP1_impute_mnar --corrupt mnar --known 0.5 --mar_rate_obs 0.5 --mar_rate_missing 0.38 --mnar_known_mask 0.875 uci --train_edge 0.7 --data energy
# Very Large Dataset:
python train_mdi.py --epochs 20000 --repeat_exp_num 3 --very_large_dataset --init_epsilon 1e-4 --apply_attr --apply_peer --sample_peer_size 5 --sample_strategy cos-similarity --node_dim 128 --edge_dim 128 --impute_hidden 128 --log_dir EXP1_impute_mnar --corrupt mnar --known 0.5 --mar_rate_obs 0.5 --mar_rate_missing 0.41 --mnar_known_mask 0.875 uci --train_edge 0.7 --data power
python train_mdi.py --epochs 20000 --repeat_exp_num 3 --very_large_dataset --init_epsilon 1e-4 --apply_attr --apply_peer --sample_peer_size 5 --sample_strategy cos-similarity --node_dim 128 --edge_dim 128 --impute_hidden 128 --log_dir EXP1_impute_mnar --corrupt mnar --known 0.5 --mar_rate_obs 0.5 --mar_rate_missing 0.41 --mnar_known_mask 0.875 uci --train_edge 0.7 --data kin8nm
python train_mdi.py --epochs 20000 --repeat_exp_num 3 --very_large_dataset --init_epsilon 1e-4 --apply_attr --apply_peer --sample_peer_size 5 --sample_strategy cos-similarity --node_dim 128 --edge_dim 128 --impute_hidden 128 --log_dir EXP1_impute_mnar --corrupt mnar --known 0.5 --mar_rate_obs 0.5 --mar_rate_missing 0.40 --mnar_known_mask 0.875 uci --train_edge 0.7 --data naval