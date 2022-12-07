# Testing view
python render.py --config configs/Garden.txt --exp_name garden-no \
    --root_dir ../datasets/360_v2/garden --use_skybox \
    --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb --render_depth \
    --ckpt_load ckpts/tnt/garden-no/epoch=49_slim.ckpt 
