import os.path
from glob import glob
import numpy as np


def load_examples(no_of_ex):
    examples_path = []

    folders_path = os.path.join('data', '*')
    folders_name = glob(folders_path)

    for folder in folders_name:
        files_path = os.path.join(folder, '*')
        files_name = glob(files_path)
        for i in range(no_of_ex // len(folders_name)):
            random_example = np.random.randint(0, len(files_name))
            examples_path.append(files_name[random_example])
    return examples_path


def examples(no_of_ex):
    texts = []
    examples_path = load_examples(no_of_ex)
    for path in examples_path:
        class_name = path.split(os.sep)[1]
        with open(path, 'r', encoding="utf8") as file:
            text = file.read()[:100]
            texts.append([text, class_name])
    return texts
