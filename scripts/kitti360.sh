# python render.py --config configs/kitti360_1908.txt --exp_name kitti360-1908 \
#     --root_dir ../datasets/KITTI-360 --dataset_name kitti  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb --render_depth \
#     --ckpt_load ckpts/kitti360/1908-1971/epoch=49_slim_1908-1971.ckpt \


# # Smog
# python render.py --config configs/kitti360_1908.txt --exp_name kitti360-1908-smog \
#     --root_dir ../datasets/KITTI-360 --dataset_name kitti  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb  \
#     --ckpt_load ckpts/kitti360/1908-1971/epoch=49_slim_1908-1971.ckpt \
#     --depth_path results/kitti/kitti360-1908/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 2.0 --rgb 0.925 0.906 0.758 

# Flood
python render.py --config configs/kitti360_1538.txt --exp_name kitti360-1538-flood \
    --root_dir ../datasets/KITTI-360 --dataset_name kitti  \
    --batch_size 1024 --use_skybox --render_path --render_rgb \
    --ckpt_load ckpts/kitti360/1538-1601/epoch=9_slim_1538-1601.ckpt \
    --simulate water --water_height -0.01 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
    --plane_path ../datasets/KITTI-360/planes/plane-1538.npy \
    --gl_theta 0.008 --gl_sharpness 500 --wave_len 1.0 --wave_ampl 2000000 \
    --anti_aliasing_factor 2 --chunk_size 500000


# python render.py --config configs/kitti360_1728.txt --exp_name kitti360-1728-flood \
#     --root_dir ../datasets/KITTI-360 --dataset_name kitti  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb \
#     --ckpt_load ckpts/kitti360/1728-1791/epoch=9_slim_1728-1791.ckpt \
#     --simulate water --water_height -0.01 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
#     --plane_path ../datasets/KITTI-360/planes/plane-1538.npy \
#     --gl_theta 0.008 --gl_sharpness 500 --wave_len 1.0 --wave_ampl 2000000 \
#     --anti_aliasing_factor 2 --chunk_size 500000

# python render.py --config configs/kitti360_1908.txt --exp_name kitti360-1908-flood \
#     --root_dir ../datasets/KITTI-360 --dataset_name kitti  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb  \
#     --ckpt_load ckpts/kitti360/1908-1971/epoch=49_slim_1908-1971.ckpt \
#     --simulate water --water_height -0.01 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
#     --plane_path ../datasets/KITTI-360/planes/plane-1538.npy \
#     --gl_theta 0.008 --gl_sharpness 500 --wave_len 1.0 --wave_ampl 2000000 \
#     --anti_aliasing_factor 2 --chunk_size 500000
