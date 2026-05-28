import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO


def main():
    a = YOLO('./SSCA-YOLO.yaml')
    a.train(
        data='./dataset.yaml', 
        epochs=300,  
        imgsz=640,  
        batch=16,  
        device=0,  
        degrees=15.0,  
        translate=0.1,  
        scale=0.5, 
        shear=5.0,  
        fliplr=0.5,  
        mosaic=1.0,  
        mixup=0.05,  
        copy_paste=0.2,    
        erasing=0.1,    

    )


if __name__ == '__main__':
    main()
