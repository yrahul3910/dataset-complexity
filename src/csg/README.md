# Cumulative Spectral Gradient (CSG)

Please see the [CVPR paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Branchaud-Charron_Spectral_Metric_for_Dataset_Complexity_Assessment_CVPR_2019_paper.pdf) for details, and the [GitHub implementation](https://github.com/Dref360/spectral-metric).

## Usage

```sh
KERAS_BACKEND=torch python3 -u <filename> | tee ../../output/csg/<filename>
```

Of course, you can change the backend as you prefer; the default is TensorFlow.
