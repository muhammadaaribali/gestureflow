# Model comparison and selection (requirement 6)

## Evaluation design

The Zenodo dataset contains 15,029 static hand poses across five classes: Open, Close, Pointer, Ok and Nice. Each row contains 40 wrist-relative coordinates scaled to the range −1 to 1. Within each gesture, adjacent rows differed by 0.0164 per coordinate on average, compared with 0.1584 for randomly paired rows. A random frame split could therefore place very similar poses on both sides of the split.

We kept chunks of up to 100 consecutive same-gesture rows together when assigning the 12,097 training and 2,932 test examples. We also kept chunks together during three-fold stratified cross-validation on the training set. The selection metric was macro-F1, which gives each gesture equal weight when averaging its F1 score. The training script selected the winner before scoring the held-out test set.

## Four-model comparison

| Model | Training CV macro-F1 | Held-out test accuracy |
|---|---:|---:|
| MLP | 99.35% | 98.53% |
| Random Forest | 98.98% | 99.59% |
| SVM | 98.87% | 98.53% |
| KNN | 98.83% | 98.87% |

The MLP won the prespecified selection metric by 0.37 percentage points over Random Forest. Random Forest achieved the highest test accuracy. Using that result to change the winner would turn the test set into another selection set. The 2,932 test frames include correlated observations within chunks, so the accuracy difference alone gives limited evidence about performance on a new person.

## How the selected MLP works

For one detected hand, the model receives the horizontal and vertical positions of 20 landmarks relative to the wrist. Each of its 64 hidden units multiplies these 40 numbers by learned weights, sums the products and adds a bias. A ReLU activation keeps positive values and sets negative values to zero. Combining these responses lets the model distinguish hand shapes using several joints together. For example, thumb and index fingertip proximity helps distinguish Ok from Open even when both poses have several extended fingers.

The output layer combines the 64 responses into five gesture scores. Softmax converts them into probabilities that sum to one. During training, cross-entropy penalises low probability for the labelled gesture. Backpropagation computes the gradients and Adam uses them to update the weights. The app uses the largest class probability as a confidence score, then combines votes over time before confirming a gesture. A high score can still occur for an unfamiliar pose.

## Error analysis

The selected five-class MLP correctly classified 2,889 of 2,932 test frames. Of its 43 mistakes, 34 were Ok frames labelled Open. This direction matters: all 600 actual Open frames were classified correctly. The two gestures had the closest mean landmark configurations (Euclidean distance 1.19), and their clusters overlap in the two-component PCA plot. These observations help explain why this pair is difficult. See the [EDA findings](eda_findings.md) and [test confusion matrix](../reports/mlp_confusion.csv).

## From the five-class benchmark to the six-class app

The original model had to pick one of five gestures even when the person was resting their hand. In the earlier left-hand webcam run, it called the relaxed pose Close with about 99% mean confidence. Left-hand input also required feature mirroring to match the trained orientation. The deployed model uses the selected MLP hyperparameters with a sixth output, Idle, which the stabilizer treats as no action.

The webcam recordings supplied 1,518 deliberate-gesture frames and 2,010 Idle frames. Training used the gesture frames and 1,502 Idle frames alongside the original Zenodo training data. Three complete Idle takes (508 frames) were held out. Keeping a take together prevents adjacent frames from that take appearing in both training and evaluation.

| Evaluation | Five-class MLP | Deployed six-class MLP |
|---|---:|---:|
| Zenodo test gesture accuracy | 98.5% | 98.4% |
| Earlier webcam gesture accuracy, right / left | 100% / 99.3% | 100% / 99.7% |
| Earlier webcam relaxed Idle recall | 0% (no Idle output) | 100% |
| Held-out Idle recall (508 frames) | Not applicable | 88.6% |

## Fresh webcam results and limitations

At the default confidence of 0.70 and release time of 0.4 s, both hands completed five deliberate holds each. All ten triggered exactly once. Relaxed Idle produced zero triggers over two 15-second intervals. Both webcam runs operated at about 31 frames per second.

- Casual fidgeting produced six false triggers on the right and three on the left in 30 seconds total. Raising confidence to 0.80 in replay left all nine. Relaxed-hand performance therefore overstates reliability during everyday hand movement. The app starts with control off.
- In an earlier right-hand Close hold, recorded predictions had a 0.77 s gap. The stabilizer released Close and fired it again after predictions resumed. A 0.80 s release prevented this duplicate in offline replay. The deployed release remains 0.4 s.
- All project webcam recordings came from one participant. Results for another person's hand shape, movement and lighting remain unknown.

The [four-model results](../reports/model_comparison.csv), [right-hand live summary](../reports/live_check_aarib_run3_right.json) and [left-hand live summary](../reports/live_check_aarib_run3_left.json) contain the underlying scores.

## Hyperparameters of the selected MLP

| Parameter | Saved value |
|---|---|
| `hidden_layer_sizes` | `(64,)` |
| `activation` | `relu` |
| `solver` | `adam` |
| `alpha` | `0.0001` |
| `learning_rate_init` | `0.001` |
| `max_iter` | `400` |
| `early_stopping` | `False` |
| `random_state` | `42` |
