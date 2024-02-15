# USPM
These are the source codes for the USPM model and its corresponding data.
Paper: Profiling Urban Streets: A Semi-Supervised Prediction Model Based on Street View Imagery and Spatial Topology
## Pretrain
The pretrained image model can be obtained by the following steps:
```
python pretrain.py
```
## Run
  For street function predction task:
```
python main.py --task function
```
  For socioeconomic indicator prediction task:
```
python main.py --task poi
```
