from ultralytics import YOLO

def entrenar_modelo():
    # 1. Cargar el modelo base preentrenado de segmentación YOLO26
    # Usamos la versión nano ('n') por velocidad, pero puedes usar 's', 'm', 'l' o 'x'
    model = YOLO("yolo26n-seg.pt")

    # 2. Iniciar el entrenamiento
    results = model.train(
        # https://universe.roboflow.com/mri-brain-tumor/brain-tumor-segmentation-8aek5/dataset/1
        
        data = "Brain tumor segmentation.v1i.yolov8-obb/data.yaml",
        epochs=200,                  # Cantidad de épocas de entrenamiento
        imgsz=640,                   # Tamaño de la imagen de entrada
        batch=16,                    # Tamaño del lote (ajustar según tu memoria VRAM)
        #device=0,                    # Usa 0 para la primera GPU de NVIDIA, o "cpu" si no tienes
        device="cpu",
        #workers=8,                   # Hilos del procesador para la carga de datos
        workers=0,                   # en ordenador personal deshabilitar multi`proceso (ocasianoa fallos)
        project="entrenamientos_yolo26", # Carpeta donde se guardará todo
        name="segmentacion_custom"   # Nombre de esta sesión específica
        
    )

if __name__ == "__main__":
    entrenar_modelo()
