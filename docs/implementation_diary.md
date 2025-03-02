# Autonomous Drone System Implementation Journey

**_Day 1_**

- Installed airsim from the source using the github repository method and built it on the command line

- Installed unreal engine and made the necessary configurations to connect the airsim package with the unreal engine version I had which was the new one but it didn't work because airsim was not compatible witn the newest version of unreal engine. After some research online I discovered that version 4.27 was the most compatible one with airsim so I installed that version instead and it worked

- Installed PX4 firmware the software version to enable a bit of control on the drone in the simulator and it worked. I carried out a couple tests such as arming and disarming the drone, commanding the drone to take off, and commmanding the drone to land using the PX4 console

**_Day 2_**

- Created the structure of the project directory and went for a structure that helps me organise the different files that I'll be working with

- Installed the necessary packages, including airism and wrote the first script of code for this project which is a script that utilises airsim python apis to connect with the airsim server and to the drone itself to enable drone control through python code but that did not work

- Turns out the firmware I was using was causing an issue because it doesn't identify a gps home location of the drone which is required for safety reasons. (expand on it)

Dependencies:

numpy
pandas
msgpack-rpc-python
tensorflow
opencv-python
matplotlib
stable-baselines3[extra]
shapely
scikit-learn
imageio
pyyaml
gymnasium
wandb
pytest
tensorboard
