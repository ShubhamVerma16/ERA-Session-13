import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(),
        A.CoarseDropout(
            min_holes=1,
            max_holes=1,
            min_height=8,
            min_width=8,
            max_height=8,
            max_width=8,
            fill_value=[0.49139968*255, 0.48215827*255 ,0.44653124*255],  # type: ignore
            p=0.5,
        ),
        A.Normalize((0.49139968, 0.48215827, 0.44653124),
                    (0.24703233, 0.24348505, 0.26158768)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize((0.49139968, 0.48215827, 0.44653124),
                    (0.24703233, 0.24348505, 0.26158768)),
        ToTensorV2(),
    ]
)
