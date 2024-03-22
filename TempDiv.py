import os
import shutil
import random

imgs = list(filter(lambda d: d.endswith("jpg"),os.listdir("datasets")))

print("Total de imagens: {}".format(len(imgs)))

test = random.sample(imgs,k=10000)
print("Conjunto de testes tem: {}".format(len(test)))
print("Movendo {} imagens para conjunto de testes".format(len(test)))
for f in test:
    try:
        shutil.move(f"datasets/{f}","datasets/LeandroGuns/test/images")
        shutil.move("datasets/{}.txt".format(f[:-4]),"datasets/LeandroGuns/test/labels")
    except FileNotFoundError as ex:
        print(ex)

imgs = list(filter(lambda d: d.endswith("jpg"),os.listdir("datasets")))
val = random.sample(imgs,k=500)
print("Movendo {} imagens para conjunto de validacao (remanescentes {}).".format(len(val),len(imgs)-500))

for f in val:
    try:
        shutil.move(f"datasets/{f}","datasets/LeandroGuns/valid/images")
        shutil.move("datasets/{}.txt".format(f[:-4]),"datasets/LeandroGruns/valid/labels")
    except FileNotFoundError as ex:
        print(ex)

imgs = list(filter(lambda d: d.endswith("jpg"),os.listdir("datasets")))
print("Movendo {} imagens para conjunto de treinamento.".format(len(imgs)))
for f in imgs:
    try:
        shutil.move(f"datasets/{f}","datasets/LeandroGuns/train/images")
        shutil.move("datasets/{}.txt".format(f[:-4]),"datasets/LeandroGuns/train/labels")
    except FileNotFoundError as ex:
        print(ex)
