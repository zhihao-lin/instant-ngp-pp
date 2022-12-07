python render_panorama.py \
    --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt --exp_name playground \
    --pano_hw 512 1024 --pano_radius 0.1 \
    --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_train --render_rgb \
    --render_depth --embed_a --embed_a_len 6 --use_skybox \
    --val_only --ckpt_path ckpts/tnt/playground/epoch=59_slim.ckpt 