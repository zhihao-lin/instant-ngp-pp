# train
# python train.py --config configs/Building.txt --exp_name building-no \
#     --batch_size 1024 --num_epochs 100 --render_path --render_rgb --render_depth \
#     --downsample 0.25 --chunk_size 500000 \
#     --mega_frame_start 753 --mega_frame_end 846

# Render
# python render.py --config configs/Building.txt --exp_name building-no \
#     --batch_size 1024 --num_epochs 100 --render_path --render_rgb --render_depth \
#     --ckpt_load ckpts/mega/building-no/last.ckpt \
#     --downsample 0.25 \
#     --mega_frame_start 754 --mega_frame_end 846

# Smog
# python render.py --config configs/Building.txt --exp_name building-smog \
#     --batch_size 1024 --num_epochs 100 --render_path --render_rgb \
#     --ckpt_load ckpts/mega/building-no/last.ckpt \
#     --downsample 0.25 \
#     --mega_frame_start 754 --mega_frame_end 846 \
#     --depth_path results/mega/building-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 3.0 --rgb 0.925 0.906 0.758

# Flood
python render.py --config configs/Building.txt --exp_name building-flood \
    --batch_size 1024 --num_epochs 100 --render_path --render_rgb \
    --ckpt_load ckpts/mega/building-no/last.ckpt \
    --downsample 0.25 \
    --mega_frame_start 754 --mega_frame_end 846 \
    --simulate water --water_height 0.165 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
    --plane_path results/mega/building-no/plane.npy \
    --gl_theta 0.008 --gl_sharpness 500 --chunk_size 600000