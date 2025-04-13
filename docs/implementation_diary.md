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








[[1741916328.7131274, 5000, 322.28533935546875], [1741920659.4643412, 10000, 531.9779052734375], [1741924723.5130894, 15000, 500.9539794921875], [1741928806.824395, 20000, 514.9216918945312], [1741957898.1771512, 15000, 511.7099609375], [1741962504.209884, 20000, 559.0969848632812], [1741968191.9410634, 25000, -1480.8482666015625], [1741973219.3088613, 30000, 520.55078125], [1741990597.0383587, 35000, 534.7280883789062], [1741996742.8517792, 50000, 526.4957885742188], [1742003509.1250465, 65000, -1273.1363525390625], [1742010330.4785666, 80000, -769.9580078125], [1742017133.996523, 95000, -957.8765258789062], [1742023974.9944174, 110000, -939.7770385742188], [1742030268.7476013, 125000, 539.9364013671875], [1742036802.6314104, 140000, 595.8408813476562], [1742069688.1088426, 145000, 527.1859130859375], [1742078243.77391, 150000, 369.0718688964844]]















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



Testing object detection:

0: 384x640 1 Tree, 1 Tree trunk, 1 Tree branch, 389.1ms
Speed: 9.0ms preprocess, 389.1ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.10706866532564163 meters  
Detected Tree branch at depth: 0.10706866532564163 meters
Detected Tree trunk at depth: 0.10706866532564163 meters

0: 384x640 1 Tree, 1 Tree branch, 453.3ms
Speed: 8.2ms preprocess, 453.3ms inference, 1.6ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: -0.10055679082870483 meters 
Detected Tree branch at depth: -0.10055679082870483 meters

0: 384x640 1 Tree, 781.4ms
Speed: 6.5ms preprocess, 781.4ms inference, 1.4ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.2646176218986511 meters   

0: 384x640 1 Tree, 494.8ms
Speed: 13.7ms preprocess, 494.8ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.3067082464694977 meters   

0: 384x640 1 Tree, 380.8ms
Speed: 5.6ms preprocess, 380.8ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.17842860519886017 meters  

0: 384x640 1 Tree, 1 Tree trunk, 424.2ms
Speed: 7.4ms preprocess, 424.2ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.3111247718334198 meters   
Detected Tree trunk at depth: 0.3111247718334198 meters

0: 384x640 1 Tree, 1 Tree trunk, 397.1ms
Speed: 7.7ms preprocess, 397.1ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.22166414558887482 meters  
Detected Tree trunk at depth: 0.22166414558887482 meters

0: 384x640 2 Trees, 1 Tree trunk, 356.3ms
Speed: 9.6ms preprocess, 356.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.18584507703781128 meters  
Detected Tree trunk at depth: 0.18584507703781128 meters
Detected Tree at depth: 0.18584507703781128 meters  

0: 384x640 2 Trees, 748.3ms
Speed: 6.3ms preprocess, 748.3ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.19685134291648865 meters  
Detected Tree at depth: 0.19685134291648865 meters  

0: 384x640 1 Tree, 1 Tree trunk, 483.2ms
Speed: 22.4ms preprocess, 483.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.21620707213878632 meters  
Detected Tree trunk at depth: 0.21620707213878632 meters

0: 384x640 1 Tree, 1 Tree branch, 364.3ms
Speed: 14.0ms preprocess, 364.3ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.5384756922721863 meters   
Detected Tree branch at depth: 0.5384756922721863 meters

0: 384x640 1 Tree, 1 Tree trunk, 954.4ms
Speed: 47.4ms preprocess, 954.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.3430006504058838 meters   
Detected Tree trunk at depth: 0.3430006504058838 meters

0: 384x640 1 Tree, 395.0ms
Speed: 9.1ms preprocess, 395.0ms inference, 1.4ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.03109830804169178 meters  

0: 384x640 1 Tree, 1 Tree trunk, 593.6ms
Speed: 13.5ms preprocess, 593.6ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.13045985996723175 meters  
Detected Tree trunk at depth: 0.13045985996723175 meters

0: 384x640 1 Tree, 1 Tree branch, 438.0ms
Speed: 8.9ms preprocess, 438.0ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.1654464602470398 meters   
Detected Tree branch at depth: 0.1654464602470398 meters

0: 384x640 1 Tree, 1 Tree trunk, 698.7ms
Speed: 8.3ms preprocess, 698.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.23288635909557343 meters  
Detected Tree trunk at depth: 0.23288635909557343 meters

0: 384x640 1 Tree, 1 Tree branch, 518.2ms
Speed: 10.5ms preprocess, 518.2ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.43046143651008606 meters  
Detected Tree branch at depth: 0.43046143651008606 meters

0: 384x640 1 Tree, 1 Tree branch, 761.7ms
Speed: 29.4ms preprocess, 761.7ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.13433288037776947 meters  
Detected Tree branch at depth: 0.13433288037776947 meters

0: 384x640 1 Tree, 1 Tree branch, 427.3ms
Speed: 10.2ms preprocess, 427.3ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.18992675840854645 meters  
Detected Tree branch at depth: 0.18992675840854645 meters

0: 384x640 1 Tree, 1 Tree branch, 438.5ms
Speed: 7.6ms preprocess, 438.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.05002982169389725 meters  
Detected Tree branch at depth: 0.05002982169389725 meters

0: 384x640 1 Tree, 1 Tree branch, 616.5ms
Speed: 23.2ms preprocess, 616.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.35968631505966187 meters  
Detected Tree branch at depth: 0.35968631505966187 meters

0: 384x640 1 Tree, 392.8ms
Speed: 8.2ms preprocess, 392.8ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.49564462900161743 meters  

0: 384x640 1 Tree, 1056.2ms
Speed: 19.3ms preprocess, 1056.2ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)     
Detected Tree at depth: 0.04947374388575554 meters  

0: 384x640 1 Tree, 1670.0ms
Speed: 20.5ms preprocess, 1670.0ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)     
Detected Tree at depth: 0.2676316201686859 meters   

0: 384x640 1 Tree, 1 Tree trunk, 1870.1ms
Speed: 53.2ms preprocess, 1870.1ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)     
Detected Tree at depth: 0.27850934863090515 meters  
Detected Tree trunk at depth: 0.27850934863090515 meters

0: 384x640 1 Tree, 1 Tree trunk, 197.3ms
Speed: 5.6ms preprocess, 197.3ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.21508437395095825 meters  
Detected Tree trunk at depth: 0.21508437395095825 meters

0: 384x640 1 Tree, 220.9ms
Speed: 8.7ms preprocess, 220.9ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.08023812621831894 meters  

0: 384x640 1 Tree, 1 Tree trunk, 221.8ms
Speed: 5.4ms preprocess, 221.8ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.33169832825660706 meters  
Detected Tree trunk at depth: 0.33169832825660706 meters

0: 384x640 1 Tree, 1 Tree trunk, 1 Tree branch, 252.1ms
Speed: 7.1ms preprocess, 252.1ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.2884408235549927 meters   
Detected Tree branch at depth: 0.2884408235549927 meters
Detected Tree trunk at depth: 0.2884408235549927 meters

0: 384x640 1 Tree, 1 Tree trunk, 230.0ms
Speed: 3.4ms preprocess, 230.0ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: -0.18799039721488953 meters 
Detected Tree trunk at depth: -0.18799039721488953 meters

0: 384x640 1 Tree, 1 Tree branch, 196.4ms
Speed: 12.9ms preprocess, 196.4ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.15192365646362305 meters  
Detected Tree branch at depth: 0.15192365646362305 meters

0: 384x640 1 Tree, 321.6ms
Speed: 10.7ms preprocess, 321.6ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.2182372510433197 meters   

0: 384x640 1 Tree, 191.1ms
Speed: 6.9ms preprocess, 191.1ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)       
Detected Tree at depth: 0.24642755091190338 meters  

0: 384x640 1 Tree, 1 Tree trunk, 234.5ms
Speed: 17.6ms preprocess, 234.5ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.3309643566608429 meters   
Detected Tree trunk at depth: 0.3309643566608429 meters

0: 384x640 1 Tree, 179.7ms
Speed: 16.2ms preprocess, 179.7ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)      
Detected Tree at depth: 0.32050418853759766 meters 