import argparse
import os
import yaml
import json
import random
import shutil

def Merge(config):
    """
    Merge the datasets
    """

    if not os.path.isdir(config.base):
        print(f"Not a directory:{config.base}")
        return
    if not os.path.isdir(config.target):
        print(f"Not a directory: {config.target}")
        return

    ybase = list(filter(lambda d: d.endswith("yaml"),os.listdir(config.base)))
    ytarget = list(filter(lambda d: d.endswith("yaml"),os.listdir(config.target)))

    with open(os.path.join(config.base,ybase[0]),'r') as f:
        base_service = yaml.safe_load(f)
    with open(os.path.join(config.target,ytarget[0]),'r') as f:
        target_service = yaml.safe_load(f)

    for d in ['test','train','valid']:
        if config.verbose:
            print("***** STARTING ANALYSIS ({}) *****".format(d))
        timages = {i:os.path.join(config.target,d,'labels',"{}.{}".format(i[:-4],'txt')) for i in os.listdir(os.path.join(config.target,d,'images'))}
        if config.perc < 1.0:
            rk = random.choices(list(timages.keys()),k=round(config.perc*len(timages)))
            timages = {z:timages[z] for z in rk}
        destination = os.path.join(config.base,d,"images")

        #Selected images will be added to base dataset if they belong to the desired class (in config.synms)
        for image in timages:
            with open(timages[image],"r") as ann:
                annotations = ann.readlines()
            dest_annotations = None
            for line in annotations:
                sl = line.split(' ')
                cl = target_service["names"][int(sl[0])].lower()
                if cl in config.classes:
                    if dest_annotations is None:
                        dest_annotations = open(os.path.join(config.base,d,"labels",os.path.basename(timages[image])),"w")

                    if config.verbose:
                        print("Converting class {} from target to class {} in base set".format(cl,config.classes[cl]))
                    dest_cl = base_service["names"].index(config.classes[cl])
                    sl[0] = str(dest_cl) #Assign class index from base set to the synonym
                    dest_annotations.write(' '.join(sl))
                    shutil.copy(os.path.join(config.target,d,"images",image),os.path.join(config.base,d,"images")) #copy image to base set
            if not dest_annotations is None:
                dest_annotations.close()


if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    synms = {"guns":"Gun","pistol":"Gun","handgun":"Gun","rifle":"Gun"}

    parser = argparse.ArgumentParser(description='Merge YoLo format datasets into one\
        dataset.')
    parser.add_argument('-p', dest='perc', type=float,
        help='Merge X% of the target dataset into the base set (Default: 100%)', default=1.0,required=False)
    parser.add_argument('-bs', dest='base', type=str,
        help='Base dataset path, to which target will be merged into', default='',required=True)
    parser.add_argument('-target', dest='target', type=str, default='',
        help='Target dataset path.',required=True)
    parser.add_argument('-classes', dest='classes', type=json.loads, default=synms,
        help='Class names equivalency (should be a dictionary formatted as string, targets are lower case, base is as writen in base set YAML).',required=False)
    parser.add_argument('-v', action='store_true', dest='verbose',
        help='Output merging messages.',default=False)

    config, unparsed = parser.parse_known_args()

    Merge(config)
