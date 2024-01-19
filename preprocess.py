import os
import shutil
from pathlib import Path, PurePath

from PIL import Image
from sklearn.model_selection import train_test_split

from util import RAW_PATH, TRAIN_PATH, TEST_PATH, clean_mkdir, ROWS, COLS, IMAGES_PATH, TEST_LABELS_PATH, TRAIN_LABELS_PATH
Image.MAX_IMAGE_PIXELS = 1933120000


def split_image_into_grids(data_path, image_path, size=(400,400)):
    output_path = str(Path(data_path) / "tmp")
    filename = PurePath(image_path).stem
    img_src = Image.open(image_path)
    rows = img_src.height
    cols = img_src.width
    count = 0
    i_step = int(COLS / 2)
    j_step = int(ROWS / 2)
    img_arr = []
    for i in range(0, cols - COLS - 1, i_step):
        for j in range(0, rows - ROWS - 1, j_step):
            img_crop = img_src.crop((i, j, i + ROWS, j + COLS))
            img_crop_output = "{}/{}_{:05d}.png".format(output_path, filename, count)
            img_crop.save(img_crop_output)
            count += 1
            img_arr.append(img_crop_output)
    return img_arr

def clean_directories(data_path):
    raw_path = str(Path(data_path) / RAW_PATH)
    train_path = str(Path(data_path) / TRAIN_PATH)
    test_path = str(Path(data_path) / TEST_PATH)
    train_labels_path = str(Path(data_path) / TRAIN_LABELS_PATH)
    test_labels_path = str(Path(data_path) / TEST_LABELS_PATH)
    clean_mkdir(raw_path)
    clean_mkdir(train_path)
    clean_mkdir(test_path)
    clean_mkdir(train_labels_path)
    clean_mkdir(test_labels_path)


def generate_samples(image_path, data_path):
    """
    Split up a larger image into correctly sized chucks for the model.

    Args:
        image_path:
        data_path:

    """
    output_path = str(Path(data_path) / RAW_PATH)
    filename = PurePath(image_path).stem
    img_src = Image.open(image_path)
    rows = img_src.height
    cols = img_src.width
    count = 0
    i_step = int(COLS / 2)
    j_step = int(ROWS / 2)
    # iterate starting X
    for i in range(0, cols - COLS - 1, i_step):
        # iterate starting Y
        for j in range(0, rows - ROWS - 1, j_step):
            img_out = img_src.crop((i, j, i + ROWS, j + COLS))
            img_out.save("{}/{}_{:05d}.jpg".format(output_path, filename, count))
            count += 1
            img_out = img_out.rotate(90)
            img_out.save("{}/{}_{:05d}.jpg".format(output_path, filename, count))
            count += 1
            img_out = img_out.rotate(90)
            img_out.save("{}/{}_{:05d}.jpg".format(output_path, filename, count))
            count += 1
            img_out = img_out.rotate(90)
            img_out.save("{}/{}_{:05d}.jpg".format(output_path, filename, count))
            count += 1


def split_sets(data_path: Path):
    input_path = data_path / RAW_PATH
    train_path = data_path /  TRAIN_LABELS_PATH
    test_path = data_path / TEST_LABELS_PATH
    filenames = []
    for filename in os.listdir(str(input_path)):
        if filename.endswith(".jpg"):
            filenames.append(filename)

    train_filenames, test_filenames = train_test_split(
        filenames, train_size=0.8, test_size=0.2
    )

    for filename in train_filenames:
        shutil.copyfile(str(input_path / filename), str(train_path / filename))

    for filename in test_filenames:
        shutil.copyfile(str(input_path / filename), str(test_path / filename))


def generate_dirty(data_path: Path):
    """
    Generate the X values by downsampling clean imagery.

    Args:
        data_path:

    """
    input_train_path = data_path / TRAIN_LABELS_PATH
    input_test_path = data_path / TEST_LABELS_PATH
    output_train_path = data_path / TRAIN_PATH
    output_test_path = data_path / TEST_PATH
    for file in os.listdir(str(input_train_path)):
        if file.split(".")[-1] not in ["jpg", "png"]:
            continue
        img = Image.open(str(input_train_path / file))
        temp = img.resize((int(ROWS / 2), int(COLS / 2)), Image.BILINEAR)
        temp = temp.resize((ROWS, COLS), Image.BILINEAR)
        temp.save(str(output_train_path / file))

    for file in os.listdir(str(input_test_path)):
        if file.split(".")[-1] not in ["jpg", "png"]:
            continue
        img = Image.open(str(input_test_path / file))
        temp = img.resize((int(ROWS / 2), int(COLS / 2)), Image.BILINEAR)
        temp = temp.resize((ROWS, COLS), Image.BILINEAR)
        temp.save(str(output_test_path / file))


def preprocess_dataset(data_path):
    data_path = Path(data_path)
    images_path = data_path / IMAGES_PATH
    clean_directories(data_path)
    for filename in os.listdir(str(images_path)):
        generate_samples(str(images_path / filename), data_path)
    split_sets(data_path)
    generate_dirty(data_path)
