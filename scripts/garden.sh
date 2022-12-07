# Testing view
# python render.py --config configs/Garden.txt --exp_name garden-no \
#     --root_dir ../datasets/360_v2/garden --use_skybox \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb --render_depth \
#     --ckpt_load ckpts/tnt/garden-no/epoch=49_slim.ckpt 

# Smog
# python render.py --config configs/Garden.txt --exp_name garden-smog-sparse \
#     --root_dir ../datasets/360_v2/garden --use_skybox \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_rgb \
#     --ckpt_load ckpts/tnt/garden-no/epoch=49_slim.ckpt \
#     --depth_path results/colmap/garden-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 0.5 --rgb 0.925 0.906 0.758 

# Estimate plane
# python -m utility.vanishing_point \
#     -dataset colmap -root_dir ../datasets/360_v2/garden \
#     -repeat 10 -align_dim 1 -align_neg \
#     -downsample 0.25

# control min:-0.45, max:-0.1
# Flood
# wave param: plane_len= 2.0, ampl_const= 2e6, freq=5.0
python render.py --config configs/Garden.txt --exp_name garden-flood \
    --root_dir ../datasets/360_v2/garden --dataset_name colmap  \
    --batch_size 1024 --use_skybox --render_path --render_rgb \
    --ckpt_load ckpts/tnt/garden-no/epoch=49_slim.ckpt \
    --simulate water --water_height -0.2 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
    --plane_path ../datasets/360_v2/garden/plane.npy \
    --gl_theta 0.008 --gl_sharpness 500 --wave_len 2.0 --wave_ampl 2000000 \
    --downsample 0.25 --anti_aliasing_factor 2 --chunk_size 100000

# gan+ngp: smog
# python train.py --config configs/Garden.txt --exp_name garden-flood-gan-ngp \
#     --root_dir ../datasets/360_v2/garden --dataset_name colmap \
#     --batch_size 1024 --num_epochs 50 --render_path --render_rgb \
#     --use_skybox --climate flood \