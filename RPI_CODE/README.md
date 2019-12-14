# SafeDetect Audio Processing Program

This program is executed on our raspberry pi to perform our audio sampling
and ML on the edge. The main program creates an AudioProcessor class
which handles the audio sampling and predictions in order to detect gunshots.
When a gunshot is detected, an event is triggered in the main loop, and the
the prediction confidence such as time stamp is sent to the xdot which will
transmit it over LoRa.
