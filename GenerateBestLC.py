import numpy as np
import matplotlib.pyplot as plt
import glob

BaseLocation ="Output_LCs"
AllFileNames = np.array(glob.glob(BaseLocation+"/*.txt"))
TargetNames = list(set([Item.split("/")[-1][:9] for Item in AllFileNames]))

for Target in TargetNames:
    print("The value of target is given by:", Target)
    SelectIndex = np.zeros(len(AllFileNames)).astype(np.bool)

    for counter, FileName in enumerate(AllFileNames):
        SelectIndex[counter] = Target in FileName
    SelectedFiles = AllFileNames[SelectIndex]

    #In order to save file under a single file
    AllData = []

    for FileItem in SelectedFiles:
        try:
            CurrentData = np.loadtxt(FileItem)
        except:
            CurrentData = np.loadtxt( FileItem, delimiter=",")

        AllData.extend(CurrentData)


    #Now save the data after re-arranging the ascending order to time
    AllData = np.array(AllData)

    Time = AllData[:,0]
    ArrangeIndex = np.argsort(Time)
    AllData = AllData[ArrangeIndex]

    np.savetxt("data/"+Target+".txt", AllData)
