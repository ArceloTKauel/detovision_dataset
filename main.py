import numpy as np
from PIL import Image

ORIGINAL_PATH = "Ejemplo explo original.jpeg"
OUTPUT_PATH = "recreated_explosion.jpg"


def load_as_grayscale(path: str) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.array(image, dtype=np.uint8)


def binarize(tensor: np.ndarray, threshold: int = 128) -> np.ndarray:
    # 0 = negro, 255 = blanco
    return np.where(tensor >= threshold, 255, 0).astype(np.uint8)


def tensor_to_image(tensor: np.ndarray, path: str) -> None:
    image = Image.fromarray(tensor, mode="L")
    image.save(path)
    print(f"Imagen guardada en: {path}")


def main():
    grayscale = load_as_grayscale(ORIGINAL_PATH)
    print(f"Dimensiones del tensor: {grayscale.shape}")
    print(f"Rango de valores: [{grayscale.min()}, {grayscale.max()}]")

    binary = binarize(grayscale)
    black_pixels = np.count_nonzero(binary == 0)
    white_pixels = np.count_nonzero(binary == 255)
    total = binary.size
    print(f"Pixeles negros (0): {black_pixels} ({black_pixels/total*100:.1f}%)")
    print(f"Pixeles blancos (255): {white_pixels} ({white_pixels/total*100:.1f}%)")

    tensor_to_image(binary, OUTPUT_PATH)


if __name__ == "__main__":
    main()
