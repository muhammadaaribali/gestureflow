# Live webcam findings

In manual webcam testing, gestures looked correct with the right hand unmirrored and the left hand mirrored. With the right hand mirrored, the classifier called an OK sign Close (96%) and an open palm Pointer (84%). These were visual observations from a 640 × 480 preview.

During live testing, the automatic setting appeared reversed. The preview now starts in manual right-hand mode with feature mirroring off. Pressing m selects left-hand mode with feature mirroring on.

The desktop runs Wayland. GestureFlow uses playerctl for playback and track changes, and wpctl for volume and mute, with pactl and amixer as fallbacks. Control starts OFF in dry-run mode. Volume changes in 3% steps and wpctl caps increases at 100%.
