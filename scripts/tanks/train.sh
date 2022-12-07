# python train.py --exp_name train-no \
#     --root_dir ../datasets/TanksAndTempleBG/Train --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 50 --render_path --render_train --render_rgb \
#     --render_depth --embed_a --embed_a_len 6 --use_skybox \
#      --ckpt_load last.ckpt # --val_only 

# python train.py --exp_name train-smog --climate smog \
#     --root_dir ../datasets/TanksAndTempleBG/Train --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 140 --render_path --render_train --render_rgb \
#     --render_depth --embed_a --embed_a_len 6 --use_skybox \
#     --ckpt_load last.ckpt --val_only 

# python train.py --exp_name train-flood --climate flood \
#     --root_dir ../datasets/TanksAndTempleBG/Train --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 140 --render_path --render_train --render_rgb \
#     --render_depth --embed_a --embed_a_len 6 --use_skybox \
#     --ckpt_load last.ckpt --val_only 


# Testing view
# python render.py --config configs/Train.txt --exp_name train-no \
#     --root_dir ../datasets/TanksAndTempleBG/Train --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb  --render_depth \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/train-no/epoch=149_slim.ckpt 

# Smog
# python render.py --config configs/Train.txt --exp_name train-smog-sparse \
#     --root_dir ../datasets/TanksAndTempleBG/Train --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/train-no/epoch=149_slim.ckpt \
#     --depth_path results/tnt/train-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 0.5 --rgb 0.925 0.906 0.758 

# Render panorama
# python render_panorama.py --config configs/Train.txt --exp_name train-no \
#     --root_dir ../datasets/TanksAndTempleBG/Train --dataset_name tnt \
#     --pano_hw 512 1024 --pano_radius 0.2 \
#     --batch_size 1024 --use_skybox \
#     --ckpt_load ckpts/tnt/train-no/epoch=149_slim.ckpt 

# Flood
# wave param: plane_len= 1.0, ampl_const= 2e6, freq=5.0
python render.py --config configs/Train.txt --exp_name train-flood \
    --root_dir ../datasets/TanksAndTempleBG/Train --dataset_name tnt  \
    --batch_size 1024 --use_skybox --render_path --render_rgb --render_mask \
    --ckpt_load ckpts/tnt/train-no/epoch=149_slim.ckpt \
    --simulate water --water_height 0.0 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
    --plane_path ../datasets/TanksAndTempleBG/Train/plane.npy \
    --gl_theta 0.008 --gl_sharpness 500 --wave_len 1.0 --wave_ampl 2000000 \
    --anti_aliasing_factor 1--chunk_size 600000

# gan+ngp: smog
# python train.py --config configs/Train.txt --exp_name train-smog-gan-ngp \
#     --root_dir ../datasets/TanksAndTempleBG/Train --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 10 --render_path --render_rgb \
#     --use_skybox --climate smog \