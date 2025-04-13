import os
import torch
import shutil
from torch.nn import Sequential
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
import gc

gc.collect()
torch.cuda.empty_cache()

# Use raw strings (r) to treat backslashes as literal characters
#model_path = r'C:\Users\uwabo\OneDrive\Documents\AIUAVProject\model\yolov10n.pt'
config_path = r'C:\Users\uwabo\OneDrive\Documents\AIUAVProject\data\local_config.yaml'
output_dir_in_drive = r'C:\Users\uwabo\OneDrive\Documents\AIUAVProject\model\TrainedWeights'
pre_trained_model = r'C:\Users\uwabo\OneDrive\Documents\AIUAVProject\model\TrainedWeights\best_yolov10n_obstacle.pt'

model = YOLO(pre_trained_model)
#model.load(model_path)

results = model.train(data=config_path, epochs=50, batch=8)

weights_path = str(results.save_dir) + '/weights/best.pt' # or 'last.pt'

os.makedirs(output_dir_in_drive, exist_ok=True)

destination_path = os.path.join(output_dir_in_drive, 'best_yolov10n_obstacle.pt')

try:
    shutil.copy(weights_path, destination_path)
    print(f"Trained model saved to: {destination_path}")
except FileNotFoundError:
    print(f"Error: Weights file not found at {weights_path}")
except Exception as e:
    print(f"An error occurred while saving the model: {e}")

results = model.val(data=config_path)

success = model.export(format="onnx")