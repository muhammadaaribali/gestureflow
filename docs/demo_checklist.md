# GestureFlow demo checklist (4 minutes 45 seconds)

## Before recording

- Start Spotify with a playlist at moderate volume. Open a YouTube video and have a hand photo ready for upload.
- Use even lighting and a plain background. Close heavy apps if the preview lags.
- From a regular terminal run `cd ~/gestureflow && .venv/bin/python app.py`. Open http://127.0.0.1:5000 and check for a visible feed and roughly 15 fps or better.
- Start with Media control and LIVE off, Right hand selected and Spotify selected. Keep the hand out of view between commands. Check microphone level if narrating live.

## Recording sequence

| Time | Show and explain |
|---|---|
| 0:00–0:30 | Introduce the team and show the problem: changing playback without reaching for a keyboard. |
| 0:30–1:15 | Follow one pose from webcam to MediaPipe landmarks, 40 coordinates, MLP probabilities, confirmed gesture and media command. |
| 1:15–2:00 | Show the average hand shapes and the four-model results. Explain why the MLP won on cross-validation even though random forest had the highest test accuracy. |
| 2:00–3:30 | Enable Media control in dry run. Show each gesture and its action log. Switch to LIVE. Pause and resume Spotify with two separate Open holds. Use Pointer and Ok for volume, Close for mute, then release and use Close again to unmute. Use Nice to skip. Show relaxed Idle and remove the hand. Select the browser player, pause YouTube with Open, then disable LIVE and show a photo prediction. |
| 3:30–4:15 | Explain why the sixth class was added. Report ten deliberate holds recognised once each, zero relaxed Idle triggers and nine fidget triggers. Mention the single-participant limitation. |
| 4:15–4:45 | Show the GitHub README, report and dataset citation. |

## After recording

Watch the recording with sound and check that the action log is readable. Upload the final video using the course-approved sharing setting. Add the working link to the README and project report, then verify it opens for a viewer.
