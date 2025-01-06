import os
import torch
import nibabel as nib
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from torchvision.transforms import functional as F

class Inference:
    def __init__(self, model_path, preprocessing, n_classes=4, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path, n_classes)
        self.preprocessing = preprocessing
        self.dice_scores = {"csf": [], "gm": [], "wm": []}
        self.hausdorff_distances = {"csf": [], "gm": [], "wm": []}

    def load_model(self, model_path, n_classes):
        model = AttentionUNet(in_channels=1, out_channels=n_classes)  # Replace with your model class
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
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

    def preprocess_image(self, image_path):
        img = nib.load(image_path).get_fdata()
        img = self.preprocessing.normalize_intensity(img)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # Shape: (1, 1, H, W)
        return img

    def predict_image(self, img):
        with torch.no_grad():
            pred = self.model(img)  # Shape: (1, n_classes, H, W)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()  # Shape: (H, W)
        return pred

    def evaluate_case(self, case_id, image_path, mask_path):
        print(f"Evaluating case: {case_id}")

        img = self.preprocess_image(image_path)
        pred_mask = self.predict_image(img)

        mask = nib.load(mask_path).get_fdata().astype(np.uint8)  # Ground truth

        for tissue_type, label in [("csf", 1), ("gm", 2), ("wm", 3)]:
            true_mask = (mask == label).astype(np.uint8)
            predicted_mask = (pred_mask == label).astype(np.uint8)

            dice_score = self.dice_coefficient(true_mask, predicted_mask)
            self.dice_scores[tissue_type].append(dice_score)

            hausdorff_dist = self.hausdorff_distance(true_mask, predicted_mask)
            self.hausdorff_distances[tissue_type].append(hausdorff_dist)

            print(f"Tissue: {tissue_type} - Dice Score: {dice_score:.4f}, Hausdorff Distance: {hausdorff_dist:.4f}")

        return pred_mask

    def predict_and_save(self, test_data_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        test_cases = [
            (case, os.path.join(test_data_dir, case, f"{case}.nii.gz"))
            for case in os.listdir(test_data_dir)
        ]

        for case_id, image_path in test_cases:
            pred_mask = self.predict_image(self.preprocess_image(image_path))
            pred_nifti = nib.Nifti1Image(pred_mask.astype(np.uint8), affine=np.eye(4))
            nib.save(pred_nifti, os.path.join(output_dir, f"{case_id}_pred.nii.gz"))

        print("Predictions saved.")

    def summarize_results(self):
        print("\nEvaluation Summary:")
        for tissue_type in self.dice_scores.keys():
            mean_dice = np.mean(self.dice_scores[tissue_type])
            mean_hd = np.mean(self.hausdorff_distances[tissue_type])
            print(f"Tissue: {tissue_type} - Mean Dice: {mean_dice:.4f}, Mean Hausdorff Distance: {mean_hd:.4f}")

if __name__ == "__main__":
    # Example usage
    model_path = "results/AttentionUNet/Best_Model.pth"
    val_data_dir = "data/Validation_Set"
    test_data_dir = "data/Test_Set"
    output_dir = "results/Test_Predictions"

    preprocessing = Preprocessing(data_aug=False, norm_intensity=True)  # Replace with your preprocessing class
    inference_system = Inference(model_path, preprocessing)

    # Run evaluation on validation set
    for case_id in ["IBSR_11", "IBSR_12", "IBSR_13"]:
        image_path = os.path.join(val_data_dir, case_id, f"{case_id}.nii.gz")
        mask_path = os.path.join(val_data_dir, case_id, f"{case_id}_seg.nii.gz")
        inference_system.evaluate_case(case_id, image_path, mask_path)

    # Summarize results
    inference_system.summarize_results()

    # Predict on test set
    inference_system.predict_and_save(test_data_dir, output_dir)
