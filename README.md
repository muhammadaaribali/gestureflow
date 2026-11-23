# GestureFlow

GestureFlow is a university AI project that uses a webcam to control playback and system volume on a Linux desktop. The Flask dashboard shows the hand landmarks, gesture probabilities and action log. You can also upload a hand photo for a single prediction.

## Team

| Name | Roll number |
|---|---|
| Muhammad Aarib Ali | 24p-3036 |
| Muhammad Shafay Naeem | 24p-3089 |
| Muhammad Abdullah | 24p-3041 |

[Project report](docs/report.md) · Demo video pending.

## Quick start on Linux

The project was tested with Python 3.11.16. The repository includes the trained classifier and MediaPipe hand landmarker. From a regular desktop terminal with [uv](https://docs.astral.sh/uv/) installed, run:

```bash
git clone https://github.com/muhammadaaribali/gestureflow.git
cd gestureflow
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python app.py
```

Open [the local dashboard](http://127.0.0.1:5000). For real playback control on Ubuntu, install `playerctl` with `sudo apt install playerctl` and start an MPRIS-compatible player such as Spotify or VLC. Browser media sessions can also appear in the player list. System volume uses `wpctl`, with `pactl` or `amixer` as fallbacks. During development, the VS Code Snap terminal restricted access to Spotify over D-Bus. A regular terminal worked.

## Using the dashboard

Select the hand you are using. Enable **Media control** to preview actions in the log. Enable **LIVE** when you want gestures to affect playback. Use the player selector and refresh button to target a running player. **Try an image instead** accepts a photo when you want a prediction without a webcam.

| Gesture | Default action |
|---|---|
| Open palm | Play or pause |
| Close (fist) | Mute or unmute |
| Pointer (index finger extended) | Raise volume while held |
| Ok (thumb and index form a ring) | Lower volume while held |
| Nice (thumbs-up, forearm roughly horizontal) | Next track |
| Idle (relaxed hand) | No action |

[config.json](config.json) defines the mapping, preferred player (Spotify), cooldowns and volume increment (3%). For the desktop interface, run `python live_predict.py --player spotify`. Press **c** to toggle control, **d** to toggle dry-run mode, **m** to change hands, **p** to select a player and **q** to quit.

## From hand landmarks to actions

MediaPipe locates 21 hand landmarks. GestureFlow subtracts the wrist position, drops the two resulting zero wrist coordinates and divides the remaining 40 values by the largest absolute offset. This reduces the effect of hand position and apparent size. Left-hand mode mirrors the horizontal coordinates.

A multilayer perceptron (MLP) with one hidden layer of 64 ReLU units predicts six class probabilities. It was trained with Adam. A gesture needs at least five qualifying votes in a seven-frame window at confidence 0.70 or above. The winning gesture must remain stable for 0.4 seconds before it triggers. The stabiliser tolerates brief dropouts with a 0.4-second release time. Pointer and Ok repeat their volume actions while held.

## Dataset and model selection

The five-class benchmark uses the static subset of *Hand Gesture Recognition Dataset: Static & Dynamic Landmarks*, collected by Yefri Estiven Vera Morales and Jesus Camilo Perafan Ledezma (Universidad del Cauca), version 1, 2026. The source is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Zenodo record and DOI 10.5281/zenodo.20032888](https://doi.org/10.5281/zenodo.20032888).

The checked-in, headerless static CSV contains 15,029 rows (the record description lists 15,028). Neighbouring rows have nearly identical coordinates, so splitting individual frames would risk leakage. We kept chunks of up to 100 consecutive same-gesture rows together for the train/test split and for three-fold training cross-validation.

| Model | CV macro-F1 | Held-out test accuracy |
|---|---:|---:|
| MLP | 99.35% | 98.53% |
| Random Forest | 98.98% | 99.59% |
| SVM | 98.87% | 98.53% |
| KNN | 98.83% | 98.87% |

The MLP won on the prespecified CV metric. Random Forest scored highest on the test set. The deployed model adds a sixth class, Idle, using labelled webcam recordings and the selected MLP hyperparameters. It achieved 98.4% gesture accuracy on the Zenodo test set. See the [model comparison](docs/model_comparison.md), [EDA figures and findings](docs/eda_findings.md), [webcam notes](docs/live_findings.md) and [earlier silhouette dataset evaluation](docs/rejected_dataset.md).

## Reproducing the analysis

The source CSVs, webcam landmark recordings and saved models are included. For retraining, use a separate checkout. The four-model script writes the five-class winner to `models/best_model.joblib`, the same path the dashboard reads.

```bash
python preprocess.py
python eda.py
python train_models.py
cp models/best_model.joblib models/model_v1.joblib
python train_v2.py
cp models/model_v2.joblib models/best_model.joblib
```

## Checks and evaluation tools

```bash
python check_features.py
python check_smoothing.py
python check_actions.py
python check_app.py
```

The checks run without a camera. Use `python replay_v2.py` to score saved sessions, or `python inspect_segment.py reports/live_frames_aarib_run2_right.csv CLOSE` to inspect a single hold. For new recordings, `live_check.py --person NAME --hand right` runs the prompted evaluation and `record_samples.py --person NAME` collects labelled training takes. Run both scripts with Python.

## Project structure

| Path | Purpose |
|---|---|
| `app.py`, `templates/` | Flask app and dashboard |
| `features.py`, `smoothing.py` | Landmark conversion and gesture confirmation |
| `actions.py`, `backends.py`, `config.json` | Action timing, Linux media controls and configuration |
| `preprocess.py`, `eda.py`, `train_models.py`, `train_v2.py` | Dataset preparation, figures, model selection and six-class training |
| `models/` | Hand landmarker, trained classifiers and held-out take IDs |
| `data/`, `reports/`, `docs/` | Datasets, evaluation outputs, figures and findings |
| `live_predict.py`, `live_check.py`, `record_samples.py` | Desktop prediction, webcam evaluation and sample collection |
| `replay_v2.py`, `inspect_segment.py`, `check_*.py` | Offline replay, segment inspection and checks |

## Known limitations

All webcam evaluations used one participant. In the fresh two-hand session, all ten deliberate holds fired once and relaxed hands produced zero triggers. Casual fidgeting produced nine false triggers in 30 seconds, including at a confidence threshold of 0.80. Keep the hand relaxed or out of view between commands. On an earlier right-hand fist, a 0.77-second prediction gap caused a second mute trigger. Left-hand use requires manual selection of mirrored features.
