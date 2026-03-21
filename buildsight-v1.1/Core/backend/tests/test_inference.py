import pytest
from ..services.inference import inference_service

def test_iou():
    # Perfect overlap
    box1 = [0, 0, 10, 10]
    assert inference_service._iou(box1, box1) == 1.0
    
    # No overlap
    box2 = [20, 20, 30, 30]
    assert inference_service._iou(box1, box2) == 0.0
    
    # Partial overlap (50%)
    # Box1 area = 100
    # Box3: 5, 0, 15, 10 -> Area 100. Intersection: 5x10 = 50. Union: 150. IoU = 50/150 = 0.33
    box3 = [5, 0, 15, 10]
    iou = inference_service._iou(box1, box3)
    assert 0.3 < iou < 0.35

def test_check_overlap():
    person_box = [0, 0, 100, 200]
    
    # Helmet box in top region (head)
    # Head region is top 35% -> y 0 to 70
    helmet_box = [20, 10, 80, 50] # Inside
    assert inference_service._check_overlap(person_box, [helmet_box], "head") == True
    
    # Helmet box way below
    helmet_box_low = [20, 150, 80, 190]
    assert inference_service._check_overlap(person_box, [helmet_box_low], "head") == False

    # Vest box in torso region
    # Torso is 10% to 80% (20 to 160)
    vest_box = [20, 50, 80, 150]
    assert inference_service._check_overlap(person_box, [vest_box], "torso") == True
