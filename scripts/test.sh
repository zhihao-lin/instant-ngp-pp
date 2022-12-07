# bash scripts/tanks/family.sh
# bash scripts/tanks/horse.sh
# bash scripts/tanks/playground.sh
# bash scripts/tanks/train.sh
# bash scripts/tanks/truck.sh

# bash scripts/garden.sh

# python render.py --config configs/Playground.txt --exp_name playground-flood-height \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb --render_mask \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --simulate water --water_height 0.01 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
#     --plane_path ../datasets/TanksAndTempleBG/Playground/plane.npy \
#     --pano_path results/tnt/playground-no/panorama/rgb/0.png --v_forward 1 0 0 --v_down 0 1 0 --v_right 0 0 1 \
#     --gl_theta 0.008 --gl_sharpness 500

# python render.py --config configs/Playground.txt --exp_name playground-flood-height \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb --render_mask \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --simulate water --water_height -0.0 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
#     --plane_path ../datasets/TanksAndTempleBG/Playground/plane.npy \
#     --pano_path results/tnt/playground-no/panorama/rgb/0.png --v_forward 1 0 0 --v_down 0 1 0 --v_right 0 0 1 \
#     --gl_theta 0.008 --gl_sharpness 500

# python render.py --config configs/Playground.txt --exp_name playground-flood-height \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb --render_mask \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --simulate water --water_height -0.01 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
#     --plane_path ../datasets/TanksAndTempleBG/Playground/plane.npy \
#     --pano_path results/tnt/playground-no/panorama/rgb/0.png --v_forward 1 0 0 --v_down 0 1 0 --v_right 0 0 1 \
#     --gl_theta 0.008 --gl_sharpness 500

# python render.py --config configs/Playground.txt --exp_name playground-flood-height \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb --render_mask \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --simulate water --water_height -0.02 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
#     --plane_path ../datasets/TanksAndTempleBG/Playground/plane.npy \
#     --pano_path results/tnt/playground-no/panorama/rgb/0.png --v_forward 1 0 0 --v_down 0 1 0 --v_right 0 0 1 \
#     --gl_theta 0.008 --gl_sharpness 500

# python render.py --config configs/Playground.txt --exp_name playground-flood-height \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb --render_mask \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --simulate water --water_height -0.03 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
#     --plane_path ../datasets/TanksAndTempleBG/Playground/plane.npy \
#     --pano_path results/tnt/playground-no/panorama/rgb/0.png --v_forward 1 0 0 --v_down 0 1 0 --v_right 0 0 1 \
#     --gl_theta 0.008 --gl_sharpness 500

# python render.py --config configs/Playground.txt --exp_name playground-flood-height \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb --render_mask \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --simulate water --water_height -0.04 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
#     --plane_path ../datasets/TanksAndTempleBG/Playground/plane.npy \
#     --pano_path results/tnt/playground-no/panorama/rgb/0.png --v_forward 1 0 0 --v_down 0 1 0 --v_right 0 0 1 \
#     --gl_theta 0.008 --gl_sharpness 500


# python render.py --config configs/Playground.txt --exp_name playground-smog-density \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --depth_path results/tnt/playground-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 0.2 --rgb 0.925 0.906 0.758 

# python render.py --config configs/Playground.txt --exp_name playground-smog-density \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --depth_path results/tnt/playground-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 0.5 --rgb 0.925 0.906 0.758

# python render.py --config configs/Playground.txt --exp_name playground-smog-density \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --depth_path results/tnt/playground-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 1.0 --rgb 0.925 0.906 0.758


# python render.py --config configs/Playground.txt --exp_name playground-smog-density \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --depth_path results/tnt/playground-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 1.5 --rgb 0.925 0.906 0.758

# python render.py --config configs/Playground.txt --exp_name playground-smog-density \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --depth_path results/tnt/playground-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 2.0 --rgb 0.925 0.906 0.758

# python render.py --config configs/Playground.txt --exp_name playground-smog-density \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --depth_path results/tnt/playground-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 2.5 --rgb 0.925 0.906 0.758

# python render.py --config configs/Playground.txt --exp_name playground-smog-density \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --depth_path results/tnt/playground-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 4.0 --rgb 0.925 0.906 0.758


## anti-aliasing
python render.py --config configs/Playground.txt --exp_name playground-flood-anti \
    --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt  \
    --batch_size 1024 --use_skybox --render_path --render_rgb --render_mask \
    --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
    --simulate water --water_height -0.0 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
    --plane_path ../datasets/TanksAndTempleBG/Playground/plane.npy \
    --pano_path results/tnt/playground-no/panorama/rgb/0.png --v_forward 1 0 0 --v_down 0 1 0 --v_right 0 0 1 \
    --gl_theta 0.008 --gl_sharpness 500 --anti_aliasing --chunk_size 552384