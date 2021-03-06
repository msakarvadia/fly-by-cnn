import vtk
import numpy as np 
import os
import glob
import argparse
from vtk.util.numpy_support import vtk_to_numpy

def main(args):
    shapes_arr = []
    if(args.surf):
        shapes_arr.append(args.surf)    

    for vtkfilename in shapes_arr:

        #Read the surface
        print("Reading: ", vtkfilename)
        vtkfilename = vtkfilename.rstrip()
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtkfilename)
        reader.Update()

        #Extract the polydata
        shapedata = reader.GetOutput()
        
        pd = shapedata.GetPointData()
        for i in range(pd.GetNumberOfArrays()):
            arr = pd.GetArray(i)
            name = arr.GetName()
            if(name == "normals"):
                np_array = vtk_to_numpy(arr)
                print(name)
                print(type(np_array))
                print(np_array)
                #np.savetxt("normals.txt", np_array, delimiter = " ")
                np.savetxt(args.out, np_array, delimiter = " ")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes maximum magnitude/scaling factor using bounding box and appends to file')
    parser.add_argument('--surf', type=str, default=None, help='Target surface or mesh')
    parser.add_argument('--out', type=str, default="norm.vtk", help='Output filename')
    #parser.add_argument('--property_file', type=str, default=None, help='.txt file with surface property')
    #parser.add_argument('--property_name', type=str, default="Property", help='Name of property')
    

    args = parser.parse_args()


    main(args)
