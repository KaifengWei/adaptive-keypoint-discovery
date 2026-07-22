import unittest

import cv2
import numpy as np

from phenotype_roi_basal_anchor import derive_phenotype_roi, select_learned_basal_anchor


class PhenotypeRoiBasalAnchorTest(unittest.TestCase):
    def synthetic(self, vertical: bool = False):
        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        shoot = np.zeros((180, 260), dtype=bool)
        root = np.zeros_like(shoot)
        if vertical:
            cv2.line(image, (130, 130), (130, 20), (35, 130, 45), 9)
            cv2.line(shoot.view(np.uint8), (130, 130), (130, 20), 1, 9)
            cv2.ellipse(image, (130, 145), (9, 14), 0, 0, 360, (145, 65, 35), -1)
            cv2.ellipse(root.view(np.uint8), (130, 145), (9, 14), 0, 0, 360, 1, -1)
            cv2.line(image, (130, 150), (170, 175), (95, 80, 55), 3)
            cv2.line(root.view(np.uint8), (130, 150), (170, 175), 1, 3)
        else:
            cv2.line(image, (80, 90), (235, 90), (35, 130, 45), 9)
            cv2.line(shoot.view(np.uint8), (80, 90), (235, 90), 1, 9)
            cv2.ellipse(image, (62, 90), (14, 9), 0, 0, 360, (145, 65, 35), -1)
            cv2.ellipse(root.view(np.uint8), (62, 90), (14, 9), 0, 0, 360, 1, -1)
            cv2.line(image, (50, 90), (10, 120), (95, 80, 55), 3)
            cv2.line(root.view(np.uint8), (50, 90), (10, 120), 1, 3)
        full = shoot | root
        return image, shoot, root, full

    def test_horizontal_roi_keeps_shoot_and_excludes_root_side(self):
        image, shoot, root, full = self.synthetic(False)
        result = derive_phenotype_roi(image, shoot, root, full)
        self.assertGreater(result["shoot_retention_ratio"], 0.90)
        self.assertLess(result["root_base_overlap_ratio"], 0.20)
        self.assertGreater(result["shoot_direction_xy"][0], 0.7)

    def test_vertical_orientation_uses_shoot_direction_not_screen_up_rule(self):
        image, shoot, root, full = self.synthetic(True)
        result = derive_phenotype_roi(image, shoot, root, full)
        self.assertGreater(result["shoot_retention_ratio"], 0.90)
        self.assertLess(result["shoot_direction_xy"][1], -0.7)

    def test_anchor_must_be_an_existing_model_point_inside_roi(self):
        image, shoot, root, full = self.synthetic(False)
        result = derive_phenotype_roi(image, shoot, root, full)
        points = [
            {"point_id": "root", "x": 40.0, "y": 105.0, "score": 0.9},
            {"point_id": "shoot_base", "x": 88.0, "y": 90.0, "score": 0.7},
        ]
        anchor, audited = select_learned_basal_anchor(
            points,
            result["phenotype_roi"],
            result["transition_center_xy"],
            result["bbox_diag"],
        )
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["point_id"], "shoot_base")
        self.assertEqual(len(audited), 2)


if __name__ == "__main__":
    unittest.main()
