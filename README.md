# srcnn
Super Resolution for Satellite Imagery
<br />
Applying super resolution strategies to sattelite imagery


![](https://github.com/WarrenGreen/srcnn/blob/master/results/05261_combined.jpg)
![](https://github.com/WarrenGreen/srcnn/blob/master/results/05454_combined.jpg)
![](https://github.com/WarrenGreen/srcnn/blob/master/results/06006_combined.jpg)


Based on: https://arxiv.org/pdf/1501.00092.pdf

## Usage

**Train:**

For training, training imagery should be stored under <data_path>/images. These images will automatically be cropped and processed for training/testing. There is an example image already in this directory and an easy way to accumulate more is using Google Maps.

```python srcnn.py --action train --data_path data --epoch 50 --batch_size=128 --model_path models/custom_weights.h5```

**Evaluate:**
```python srcnn.py --action test --data_path data --model_path models/custom_weights.h5```


**Run: (Currently Broken)**
```python srcnn.py --action run --data_path evaluate --model_path models/custom_weights.h5 --output_path results```
