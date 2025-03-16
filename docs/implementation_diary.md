# Autonomous Drone System Implementation Journey

**_Day 1_**

- Installed airsim from the source using the github repository method and built it on the command line

- Installed unreal engine and made the necessary configurations to connect the airsim package with the unreal engine version I had which was the new one but it didn't work because airsim was not compatible witn the newest version of unreal engine. After some research online I discovered that version 4.27 was the most compatible one with airsim so I installed that version instead and it worked

- Installed PX4 firmware the software version to enable a bit of control on the drone in the simulator and it worked. I carried out a couple tests such as arming and disarming the drone, commanding the drone to take off, and commmanding the drone to land using the PX4 console

**_Day 2_**

- Created the structure of the project directory and went for a structure that helps me organise the different files that I'll be working with

- Installed the necessary packages, including airism and wrote the first script of code for this project which is a script that utilises airsim python apis to connect with the airsim server and to the drone itself to enable drone control through python code but that did not work

- Turns out the firmware I was using was causing an issue because it doesn't identify a gps home location of the drone which is required for safety reasons. (expand on it)


**_Day 4_**
- Finally got the python code to work and made the drone fly to the waypoints I set in the code. I also implemented the taking images function from the airsim library to take images of the environment and saved them in the data folder for data collection and for pre-processing and processing as well as feature extraction.
- I developed a training manager to manage the RL model training, which pulls some of the parameters that were set in the config folder (from the local_config file and cloud_config file) which contain the waypoints, the batch-size, learning-rate, number of epochs, the timesteps per epoch, and the max number of episodes
- Created a boilerplate of the custom environment for RL model training


**_Day 5_**
- Further expanded on the custom environment and included functions such as observation function, reward function, reset function, get position function, etc. 
- Used a pre-trained RCNNs fast rast-net model as a feature extractor for the images that my drone collects from the environment.
- Created a training loop using the stable-baselines3 package to set up my proximal policy optimisation algorithm. The ppo gets fed by the data outputted by the feature extractor and forms its policy based on what it received from it. 


**_Day 6_**
- Trained the model and it didn't work at first because the ppo architecture from stable-baselines3 was set up to receive data processed from a deep learning model that uses pytorch instead of tensorflow. And I could not find another pre-trained model that uses pytorch for feature-extracting (of course they do exist somewhere I was just not lucky enough to find it), so I created my own convolutional neural network model that uses pytorch.
- After some more debugging I finally managed to get the code to work and started training my model, the training took pretty long because I'm training using my local computer that has no gpu support so I had to rely on my cpu instead.
- After finishing the training loop I found out that the total reward my rl agent accumulated was in the high negatives which all though a horrible result is not really suprising considering the fact that it started training from scratch, although I the rewards had some tweaking to be done as the agent was heavily penalised for a lot of things it did which makes it harder for it to learn as it also needs to receive positive rewards for some of the things it does that helps it to move closer to the waypoint that was set. The drone was mostly moving around the same position it spawned in.



**_Day 7_**
- After further observation I changed my code so that my training agent won't get heavily penalised for some of the decisions it makes and also increase the value of some reward elements that encourage the drone to move reach its destination and avoid obstacles.
- I trained my model a couple more times and identified that my drone starts to avoid obstacles ahead of it but not in a good way. Instead of manouevering past obstacles by slalloming through it, it goes round the obstacles and keeps too much of a distance away from obstacles ahead to the point it keeps moving away from the goal which increases the distance between its position and the goal which also increases the amount of time taken for it to reach its goal. It moves too far off the plane.
- Another thing I noticed is that I'm only using depth images as the drone's eyes to identify potential obstacles ahead which might not be as accurate on its own and can often deceive the drones perception. Which is why I decided to edit my airsim settings.json file and included a couple more sensors such as LiDAR sensors for a much greater view that can be scaled up to 3D and IMU for the tracking of the orientation and the motion of the drone. Forming a fusion of all those perception sensors can drastically improve the drone's vision and make it more accurate which will improve and provide more options for path planning that avoid obstacles effectively and is energy saving.

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
