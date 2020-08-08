# Convert torch to pytorch (forked from [here](https://github.com/clcarwin/convert_torch_to_pytorch))
Convert torch t7 model to pytorch model and source.

## Updates:
- Merged code from [here](https://github.com/clcarwin/convert_torch_to_pytorch/pull/43/files)
- Added `environment.yml` file for conda configuration

## Installation
- Clone Repository
- Create conda env from `environment.yml`

## Usage
- Run `python convert_torch.py -m <model_path>/<model_name>.t7` from inside of the conda env
- Two files will be created:
  - Model state dict: `<model_path>/<model_name>.pth`
  - Model initialization code: `<model_path>/<model_name>.py`
- `<model_path>/<model_name>.py` should include something like this:
    ```python
    /home/michael/Documents/Github/Michael/vgg_face_descriptor/original_code/VGG_FACE = nn.Sequential( # Sequential,
        nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),

        ... more model layers ...

        nn.ReLU(),
        nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
        Lambda(lambda x: x.view(x.size(0),-1)), # View,
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(25088,4096)), # Linear,
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,2622)), # Linear,
        nn.Softmax(),
    )
    ```
- To load this model, remove everything after the final ReLU - the final layers are to reshape the tensor and perform classification.  Only the actual features are needed for fine-tuning the model
- Additionally, turn the path into a normal python variable so the model itself can be easily loaded:
    ```python
    # Example:
    model = nn.Sequential( # Sequential,
        nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
        nn.ReLU(),

        ... more model layers ...

        nn.ReLU(),
        nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    )
    ```
- The model can now be loaded like this:
    ```python
    model = ...
    model.load_state_dict(torch.load('<model_path>/<model_name>.pth'))
    model.eval()
    ```

## Validated
All the models in this table can be converted and the results have been validated.

| Network             | Download |
| ------------------- | -------- |
| AlexNet | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| Inception-V1 | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| VGG-16 | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| VGG-19 | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| ResNet-18 | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| ResNet-200 | [cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks) |
| ResNeXt-50 (32x4d) | [ResNeXt](https://github.com/facebookresearch/ResNeXt) |
| ResNeXt-101 (32x4d) | [ResNeXt](https://github.com/facebookresearch/ResNeXt) |
| ResNeXt-101 (64x4d) | [ResNeXt](https://github.com/facebookresearch/ResNeXt) |
| DenseNet-264 (k=32) | [DenseNet](https://github.com/liuzhuang13/DenseNet#results-on-imagenet-and-pretrained-models) |
| DenseNet-264 (k=48) | [DenseNet](https://github.com/liuzhuang13/DenseNet#results-on-imagenet-and-pretrained-models) |
