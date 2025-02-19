# MoERad: Mixture of Experts for Radiology Report Generation

This is the implementation of MoERad that was submitted as part of [ReXrank](https://rexrank.ai/) Leaderboard. We have trained and evaluated the model on IU-Xray dataset. The model's prediction on IU-Xray test split and its corresonding CXR-Metrics scores can be found in `results/iu_xray`. The model generates only the "Findings" part of the report.

## Download Model Checkpoints
The code requires the following checkpoints to generate reports: `MoE_model.pt` and `pretrained_BCL.pt`.

You can download the checkpoints from this [Google Drive link](https://drive.google.com/drive/folders/1CtzPf39mpvpW6l3l6UcfbP6ghGVkpshX?usp=sharing). Once downloaded, please place those in the `models` folder.

## IU-Xray Test Split Evaluation Score
| BLEU (↑) | BERT Score (↑) | Semb Score (↑) | RadGraph Combined (↑) | RadCliq-v0 (↓) | RadCliq-v1 (↓) | 
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.2704 | 0.5421 | 0.6355 | 0.3397 | 2.3472 | 0.5000 |

## Run Inference
Please update the defaults of the following Args in the `inference.py` file as per your preference:

`--input_json_file`, `--save_json_file`, and `--img_root_dir`

`inference.py` also supports ArgumentParser. You can also pass the above Args during the file run. Either way will work. 

Once everything is done, please run the `inference.py` file to generate reports for CXR Images. This would result in a `MoERad.json` file that contains predicted reports. Once this output JSON file is created, please run the `evaluation_script.py` file to create `gts.csv` and `preds.csv` files with sample-wise ground truth and predicted reports, respectively. All these output files will be placed in the `results/iu_xray` folder.

## MoERad Architecture
![](./figures/MoERad_Archi.png) 

## Acknowledgement
This work inherits code from [BCL](https://github.com/FlamieZhu/Balanced-Contrastive-Learning) and [Build nanoGPT](https://github.com/karpathy/build-nanogpt). We extend our gratitude to them.