# Live webcam findings

In manual webcam testing, gestures looked correct with the right hand unmirrored and the left hand mirrored. With the right hand mirrored, the classifier called an OK sign Close (96%) and an open palm Pointer (84%). These were visual observations from a 640 × 480 preview.

Automatic mode uses MediaPipe’s hand label from the selfie-view frame. It mirrors horizontal landmark offsets for Left and uses the original offsets for Right. The live preview shows the detected side and the setting used, so the user can check the labels against their physical hands.
