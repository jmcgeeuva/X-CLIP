# ViT-B/32
job_name="xclip_edu_vit32"
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    main_xclip.py \
    --do_eval \
    --init_model ./ckpts_dsw/xclip_edu_vit32/pytorch_model.bin.14 \
    --num_thread_reader=4 \
    --epochs=20 \
    --batch_size=8 \
    --n_display=10 \
    --data_path ./lists/ \
    --features_path ./frames/ \
    --output_dir ckpts_dsw/${job_name} \
    --lr 1e-4 \
    --max_words 64 \
    --max_frames 64 \
    --batch_size_val 24 \
    --datatype edu \
    --feature_framerate 1 \
    --coef_lr 1e-3 \
    --freeze_layer_num 0  \
    --slice_framepos 2 \
    --loose_type \
    --linear_patch 2d \
    --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}