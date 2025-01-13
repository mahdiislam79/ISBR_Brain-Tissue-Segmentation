import os
import argparse
import torch
import nibabel as nib
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from torchvision.transforms import functional as F
import scipy.ndimage
import pickle as pkl


# from preprocessing import Preprocessing
from metrics import *
# from attention_unet import AttentionUNet
from unet import UNet
# from sep_res_unet import UNet
from dataset import MRIDataset
from torch.utils.data import DataLoader

class Inference:
    def __init__(self, model_path, n_classes=4, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path, n_classes)
        self.dice_scores = {"csf": [], "gm": [], "wm": []}
        self.hausdorff_distances = {"csf": [], "gm": [], "wm": []}
        self.predictions = []

    def load_model(self, model_path, n_classes):
        model = UNet(in_channels=1, out_channels=n_classes)  # Replace with your model class
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def dice_coefficient(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        return (2. * intersection + 1e-6) / (y_true.sum() + y_pred.sum() + 1e-6)

    @staticmethod
    def hausdorff_distance(y_true, y_pred):
        y_true_points = np.argwhere(y_true)
        y_pred_points = np.argwhere(y_pred)
        return max(
            directed_hausdorff(y_true_points, y_pred_points)[0],
            directed_hausdorff(y_pred_points, y_true_points)[0]
        )
    
    @staticmethod    
    def get_test_dataloader(test_id, test_data_dir):
        test_dataset = MRIDataset(
            test_id, test_data_dir,
            slice_size=0, target_size=(256, 256), 
            n_classes=4, data_augmentation=False,
            mode="test" 
        )
        
        test_loader = DataLoader(
        test_dataset, 
        batch_size=len(test_dataset.samples) // len(test_dataset.volume_map), 
        shuffle=False
        )
        
        return test_loader
    
    @staticmethod
    def get_val_dataloader(val_id, val_data_dir):
        val_dataset = MRIDataset(
            val_id, val_data_dir,
            slice_size=0, target_size=(256, 256), 
            n_classes=4, data_augmentation=False,
            mode="val" 
        )
        
        val_loader = DataLoader(
        val_dataset, 
        batch_size=len(val_dataset.samples) // len(val_dataset.volume_map), 
        shuffle=False
        )
        
        return val_loader
            
    def evaluate_and_save(self, data_dir, set_type="val"):
        if set_type == "val":
            mean_dice_scores = []
            mean_hausdorff_distances = []
            case_ids = [['IBSR_11'], ['IBSR_12'], ['IBSR_13'], ['IBSR_14'], ['IBSR_17']]
            # unpickple the label metadata
            label_metadatas = pkl.load(open('code/valid_label_metadatas.pkl', 'rb'))
            
            for case_id in case_ids:
                print(f"Evaluating Validation case: {case_id[0]}")
                
                val_loader = Inference.get_val_dataloader(case_id, data_dir)
                # get the label metadata for the case
                case_affine = label_metadatas[case_id[0]]['affine']
                case_header = label_metadatas[case_id[0]]['header']
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        
                        # Remove any unnecessary dimensions (e.g., singleton dimensions)
                        batch_x = torch.squeeze(batch_x, dim=2)  # Remove the 3rd dimension if its size is 1
                        batch_x = torch.permute(batch_x, [1, 0, 2, 3])  # Change the order of dimensions
                        # print('label before squeeze', batch_y.shape)
                        batch_y = torch.squeeze(batch_y, dim=0)  # Remove the 1st dimension
                        # print('label after squeeze', batch_y.shape)
                        pred = self.model(batch_x)
                        # print('prediction shape', pred.shape)
                        pred = torch.argmax(pred, dim=1).cpu().numpy()
                        for tissue_type, label in [("csf", 1), ("gm", 2), ("wm", 3)]:
                            true_mask = (torch.argmax(batch_y, dim=1) == label).cpu().numpy()
                            # print('true mask shape', true_mask.shape)
                            predicted_mask = (pred == label)
                            # print('predicted mask shape', predicted_mask.shape)
                            dice_score = Inference.dice_coefficient(true_mask, predicted_mask)
                            hausdorff_dist = Inference.hausdorff_distance(true_mask, predicted_mask)
                            self.dice_scores[tissue_type].append(dice_score)
                            self.hausdorff_distances[tissue_type].append(hausdorff_dist)
                            print(f"Tissue: {tissue_type} - Dice Score: {dice_score:.4f}, Hausdorff Distance: {hausdorff_dist:.4f}")
                            print(f'{case_id[0]} Mean Dice Score: {np.mean(self.dice_scores[tissue_type]):.4f}, {case_id[0]} Mean Hausdorff Distance: {np.mean(self.hausdorff_distances[tissue_type]):.4f}')
                            mean_dice_scores.append(np.mean(self.dice_scores[tissue_type]))
                            mean_hausdorff_distances.append(np.mean(self.hausdorff_distances[tissue_type]))
                        print("\n")
                        self.predictions.append(pred)
                        pred = np.transpose(pred, [1, 0, 2])
                        # prediction_resampled = scipy.ndimage.zoom(
                        #     pred, 
                        #     zoom= np.array([1.0, 1.5, 1.0]) / np.array([1.0, 1.0, 1.0]), 
                        #     order=0)  # Nearest-neighbor interpolation for segmentation masks
                        pred_nifti = nib.Nifti1Image(pred, affine=case_affine, header=case_header)
                        nib.save(pred_nifti, os.path.join(data_dir, f"{case_id[0]}_pred.nii.gz"))
            print("\n")
            print('\n')
            print(f'Mean Dice Score: {np.mean(mean_dice_scores):.4f}, Mean Hausdorff Distance: {np.mean(mean_hausdorff_distances):.4f}')
                        
        elif set_type == "test":
            case_ids = [['IBSR_02'], ['IBSR_10'], ['IBSR_15']]
            label_metadatas = pkl.load(open('code/test_metadatas.pkl', 'rb'))
            for case_id in case_ids:
                print(f"Evaluating Test case: {case_id[0]}")
                test_loader = Inference.get_test_dataloader(case_id, data_dir)
                case_affine = label_metadatas[case_id[0]]['affine']
                case_header = label_metadatas[case_id[0]]['header']
                with torch.no_grad():
                    for batch_x in test_loader:
                        batch_x = batch_x.to(self.device)
                        batch_x = torch.squeeze(batch_x, dim=2)  # Remove the 3rd dimension if its size is 1
                        batch_x = torch.permute(batch_x, [1, 0, 2, 3])  # Change the order of dimensions
                        pred = self.model(batch_x)
                        pred = torch.argmax(pred, dim=1).squeeze(1).cpu().numpy()
                        self.predictions.append(pred)
                        pred = np.transpose(pred, [1, 0, 2])
                        pred_nifti = nib.Nifti1Image(pred.astype(np.uint16), affine=case_affine, header=case_header)
                        nib.save(pred_nifti, os.path.join(data_dir, f"{case_id[0]}_pred.nii.gz"))
                        
        
                        
                            
if __name__ == "__main__":        
    parser = argparse.ArgumentParser(description="Inference Script for Medical Image Segmentation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--set_type", type=str, choices=["val", "test"], default="val", help="Dataset type: 'val' or 'test'")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the inference on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    # Initialize Inference class
    inference = Inference(model_path=args.model_path, device=args.device)

    # Run evaluation
    inference.evaluate_and_save(data_dir=args.data_dir, set_type=args.set_type)

   
            
            
        
    # def preprocess_image(self, image_path):
    #     img = nib.load(image_path).get_fdata()
    #     img = self.preprocessing.normalize_intensity(img)
    #     img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # Shape: (1, 1, H, W)
    #     return img

    # def predict_image(self, img):
    #     with torch.no_grad():
    #         pred = self.model(img)  # Shape: (1, n_classes, H, W)
    #         pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()  # Shape: (H, W)
    #     return pred

    # def evaluate_case(self, case_id, image_path, mask_path):
    #     print(f"Evaluating case: {case_id}")

    #     img = self.preprocess_image(image_path)
    #     pred_mask = self.predict_image(img)

    #     mask = nib.load(mask_path).get_fdata().astype(np.uint8)  # Ground truth

    #     for tissue_type, label in [("csf", 1), ("gm", 2), ("wm", 3)]:
    #         true_mask = (mask == label).astype(np.uint8)
    #         predicted_mask = (pred_mask == label).astype(np.uint8)

    #         dice_score = self.dice_coefficient(true_mask, predicted_mask)
    #         self.dice_scores[tissue_type].append(dice_score)

    #         hausdorff_dist = self.hausdorff_distance(true_mask, predicted_mask)
    #         self.hausdorff_distances[tissue_type].append(hausdorff_dist)

    #         print(f"Tissue: {tissue_type} - Dice Score: {dice_score:.4f}, Hausdorff Distance: {hausdorff_dist:.4f}")

    #     return pred_mask

    # def predict_and_save(self, test_data_dir, output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    #     test_cases = [
    #         (case, os.path.join(test_data_dir, case, f"{case}.nii.gz"))
    #         for case in os.listdir(test_data_dir)
    #     ]

    #     for case_id, image_path in test_cases:
    #         pred_mask = self.predict_image(self.preprocess_image(image_path))
    #         pred_nifti = nib.Nifti1Image(pred_mask.astype(np.uint8), affine=np.eye(4))
    #         nib.save(pred_nifti, os.path.join(output_dir, f"{case_id}_pred.nii.gz"))

    #     print("Predictions saved.")

    # def summarize_results(self):
    #     print("\nEvaluation Summary:")
    #     for tissue_type in self.dice_scores.keys():
    #         mean_dice = np.mean(self.dice_scores[tissue_type])
    #         mean_hd = np.mean(self.hausdorff_distances[tissue_type])
    #         print(f"Tissue: {tissue_type} - Mean Dice: {mean_dice:.4f}, Mean Hausdorff Distance: {mean_hd:.4f}")

# if __name__ == "__main__":
#     # Example usage
#     model_path = "results/AttentionUNet/Best_Model.pth"
#     val_data_dir = "data/Validation_Set"
#     test_data_dir = "data/Test_Set"
#     output_dir = "results/Test_Predictions"

#     preprocessing = Preprocessing(data_aug=False, norm_intensity=True)  # Replace with your preprocessing class
#     inference_system = Inference(model_path, preprocessing)

#     # Run evaluation on validation set
#     for case_id in ["IBSR_11", "IBSR_12", "IBSR_13"]:
#         image_path = os.path.join(val_data_dir, case_id, f"{case_id}.nii.gz")
#         mask_path = os.path.join(val_data_dir, case_id, f"{case_id}_seg.nii.gz")
#         inference_system.evaluate_case(case_id, image_path, mask_path)

#     # Summarize results
#     inference_system.summarize_results()

#     # Predict on test set
#     inference_system.predict_and_save(test_data_dir, output_dir)
