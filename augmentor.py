import torchio as tio

spatial_augment = {
    tio.RandomAffine(degrees=15, p=0.5),
    # tio.RandomFlip(axes=(0), flip_probability=0.5)
}

intensity_augment = {
    tio.RandomNoise() ,
    tio.RandomBiasField() ,
    tio.RandomBlur(std=(0,1.5)),
    tio.RandomMotion(),
}

vit_transforms = tio.Compose([
    tio.Compose(spatial_augment),
    tio.OneOf(intensity_augment, p=0.75),
    tio.RescaleIntensity(out_min_max=(0, 1))
])