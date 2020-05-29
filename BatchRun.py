#importing the libraries
import os
os.environ['PATH']=os.environ['PATH']+":/usr/local/lib64/python3.7/site-packages"

import glob




InputFiles = glob.glob("data/*.txt")


#Check for the most recent light curves
Location2Check="../PhotDB"


for File in InputFiles:
    print("*"*50)
    print("Now running::", File)
    OutputDirName = File.split("/")[1][:-4]
    print("The name of the output directory is given by::", OutputDirName)
    Command = "python3 -W ignore Launch.py -m 2 -o %s -i %s" %(OutputDirName, File)

    #input("Starting after this")

    Status = os.system(Command)
    print("The value of Status is::", Status)
    #input("Checl on status")
    if not(os.path.exists("OldData")):
        os.system("mkdir OldData")

    if Status == 1:
        os.system("mv %s OldData" %File)
