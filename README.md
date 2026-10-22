# MMA3001 - Pork rasher packaging defects project
# by Gemma Micomonaco

This project trains two object detection models (YOLOv8) to find packaging defects in photos of pork rasher trays, like loose meat, twisted meat, unsealed packs and wrinkles. It then tests how accurate and fast each model is, to see if a small fast model is good enough for a production line.

## What it does
- Downloads the dataset from Roboflow
- Checks the labels before training
- Checks if near-identical video frames are in both the training and test sets
- Trains two models: YOLOv8n (small and fast, the baseline) and YOLOv8s (bigger, the main model)
- Tests both on photos they haven't seen and compares accuracy and speed

## Inputs and outputs
- Input: one photo of a pork rasher tray (720 x 540 pixels, JPG)
- Output: a box around each defect found, with the defect type and a confidence score from 0 to 1

## How to run it
Everything runs in Google Colab, so nothing needs to be installed.
1. Make a free Roboflow account and copy your Private API Key (Settings, then API Keys).
2. Open notebooks/pork_rasher_defects.ipynb in Google Colab (File, Open notebook, GitHub tab, then paste this repo's link).
3. Runtime, Change runtime type, T4 GPU.
4. Runtime, Run all, then paste your Roboflow key when asked.

It takes about 25 minutes on a free Colab GPU (with 15 epochs).

## Environment and dependencies
- Python 3 on Google Colab with a T4 GPU
- The libraries and their versions are listed in requirements.txt. The notebook installs them automatically.

## Folders
- notebooks - the main notebook
- src - helper functions (pork_inspect.py)
- tests - tests for the helper functions
- docs - HTML documentation (made with pdoc) and the test report
- results - tables and charts from the final run

## Data
Pork Rasher Error (Packaging) dataset, version 4, from Roboflow Universe:
https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging

It is licensed CC BY 4.0. The photos are not stored in this repo. The notebook downloads them.

## Tests and docs
Run the tests: python -m pytest -v tests/
Rebuild the docs: pdoc src/pork_inspect.py -o docs/html

## Results
Results from the full run (15 epochs). The test set has 80 photos.

- The small model (YOLOv8n) did better than the bigger one: mAP50 of 0.37 vs 0.28.
- The bigger model (YOLOv8s) stopped early at epoch 11 (its best was epoch 6), probably because the validation set is small and its scores jump around.
- Unsealed packs were found well: the small model found 88% of them (49 in the test set).
- Loose and twisted meat only have 3 or 4 boxes in the test set, and wrinkle has none, so those can't be judged properly.
- About 2 in 3 test photos have a training photo within 5 video frames, so the scores are probably better than real use would be.
- Both models are fast enough for 60 trays per minute, even on a CPU (small model: 12 ms per tray on GPU, 92 ms on CPU).

Full tables and charts are in the results folder.

## AI use
Most of the coding and formatting/layout such as this .MD file for this project was drafted by myself and Anthropic's flagship Claude Opus 5.5 model at Max effort inside the Claude Desktop app. Claude and I fully understand the ins and outs of this project. Admittedly Claude did drastically help with the starter code. Each file I have reviewed and manually uploaded myself, Claude drafted comments but I went into detail with Claude what a code block actually means rewrote the comments so I actually understand whats going on. Full details about my AI use are in the AI reflection section in my report.

## Licence
MIT, see LICENSE.
