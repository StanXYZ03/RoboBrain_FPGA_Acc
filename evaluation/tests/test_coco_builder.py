from __future__ import annotations

import unittest

from evaluation.build_coco_spatial_smoke import choose_annotations


class CocoBuilderTest(unittest.TestCase):
    @staticmethod
    def payload() -> dict:
        return {
            "images": [
                {"id": index, "width": 200, "height": 200, "file_name": f"{index}.jpg"}
                for index in range(1, 7)
            ],
            "categories": [{"id": 1, "name": "cup"}, {"id": 2, "name": "book"}],
            "annotations": [
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 1 if index <= 3 else 2,
                    "bbox": [20, 20, 60, 60],
                    "area": 3600,
                    "iscrowd": 0,
                }
                for index in range(1, 7)
            ],
        }

    def test_exclusion_and_category_cap(self) -> None:
        selected = choose_annotations(
            self.payload(),
            4,
            7,
            max_per_category=2,
            excluded_image_ids={1},
        )
        image_ids = [int(image["id"]) for image, _, _ in selected]
        categories = [category for _, _, category in selected]
        self.assertNotIn(1, image_ids)
        self.assertEqual(categories.count("cup"), 2)
        self.assertEqual(categories.count("book"), 2)


if __name__ == "__main__":
    unittest.main()
