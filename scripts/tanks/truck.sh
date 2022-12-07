python train.py --exp_name truck \
    --root_dir ../datasets/TanksAndTempleBG/Truck --dataset_name tnt \
    --batch_size 1024 --scale 8.0 --num_epochs 43 --render_path --render_train --render_rgb \
    --render_depth --embed_a --embed_a_len 6 --use_skybox \
    --ckpt_load last.ckpt # --val_only 


# Testing view
python render.py --config configs/Truck.txt --exp_name truck \
    --root_dir ../datasets/TanksAndTempleBG/Truck --dataset_name tnt \
    --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb  --render_depth \
    --embed_a --embed_a_len 8 --use_skybox \
    --ckpt_load ckpts/tnt/truck/last.ckpt 