model = dict(
    type='PSENet',
    backbone=dict(
        type='resnet18',
        pretrained=True ##True
    ),
    neck=dict(
        type='FPN_r18',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        type='PSENet_Head',
        in_channels=1024,
        hidden_dim=256,
        num_classes=7,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=0.7
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.3
        )
    )
)
data = dict(
    batch_size=8,##设备限制，只能调整到8
    train=dict(
        type='PSENET_CTW',
        split='train',
        is_transform=True,
        img_size=640,
        short_size=640,
        kernel_num=7,
        min_scale=0.7,
        read_type='cv2'
    ),
    test=dict(
        type='PSENET_CTW',
        split='test',
        short_size=640,##640
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule=(200, 400,),
    epoch=600,
    optimizer='Adam',
    pretrain='checkpoints/psenet_r18_synth/checkpoint.pth.tar'
)
test_cfg = dict(
    min_score=0.85,
    min_area=16,
    kernel_num=7,
    bbox_type='poly',
    result_path='outputs/submit_ctw.zip'
)
