# train
# python train.py --config configs/Playground.txt --exp_name playground-no \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 2048 --scale 8.0 --num_epochs 70 --render_path --render_train --render_rgb \
#     --render_depth --embed_a --embed_a_len 8 --use_skybox 


# Testing view
# python render.py --config configs/Playground.txt --exp_name playground-no \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --render_depth 

# Smog
# python render.py --config configs/Playground.txt --exp_name playground-smog-sparse \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --depth_path results/tnt/playground-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 0.5 --rgb 0.925 0.906 0.758 

# Render panorama
# python render_panorama.py --config configs/Playground.txt --exp_name playground-no \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --pano_hw 512 1024 --pano_radius 0.2 \
#     --batch_size 1024 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt 

# Flood
# wave param: plane_len=0.2, ampl_const=5e5
python render.py --config configs/Playground.txt --exp_name playground-flood \
    --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt  \
    --batch_size 1024 --use_skybox --render_path --render_rgb --render_mask --render_depth \
    --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
    --simulate water --water_height 0.0 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
    --plane_path ../datasets/TanksAndTempleBG/Playground/plane.npy \
    --gl_theta 0.008 --gl_sharpness 500 --wave_len 0.2 --wave_ampl 500000 \
    --anti_aliasing_factor 1 --chunk_size 600000

# gan+ngp: smog
# python train.py --config configs/Playground.txt --exp_name playground-smog-gan-ngp \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 50 --render_path --render_rgb \
#     --use_skybox --climate smog \
#     --ckpt_load ckpts/tnt/playground-smog-gan-ngp/last.ckpt #--val_only