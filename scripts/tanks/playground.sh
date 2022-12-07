# train
python train.py --config configs/Playground.txt --exp_name playground \
    --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
    --batch_size 2048 --scale 8.0 --num_epochs 70 --render_path --render_train --render_rgb \
    --render_depth --embed_a --embed_a_len 8 --use_skybox 


# Testing view
python render.py --config configs/Playground.txt --exp_name playground \
    --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
    --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
    --embed_a --embed_a_len 8 --use_skybox \
    --ckpt_load ckpts/tnt/playground/last.ckpt \
    --render_depth 