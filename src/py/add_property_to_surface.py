import vtk
import numpy as np 
import os
import glob
import argparse

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
        
        #Get the number of points in the polydata
        shapedatapoints = shapedata.GetPoints()
        idNumPointsInFile = shapedatapoints.GetNumberOfPoints()
        
        #Add property to each point
        prop = vtk.vtkFloatArray()
        prop.SetNumberOfComponents(1)
        prop.SetName(args.property_name)

        with open(args.property_file) as property_file:
            for line in property_file:
                    point_val = float(line[:-1])
                    prop.InsertNextValue(point_val)

        shapedata.GetPointData().AddArray(prop)

        #Write the updated surface to a file
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(shapedata)
        writer.SetFileName(args.out)
        writer.Write()
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Computes maximum magnitude/scaling factor using bounding box and appends to file')
    #TODO: MAKE AT LEAST ONE OF THE DIRECTRY OR SURF REQUIRED
    parser.add_argument('--surf', type=str, default=None, help='Target surface or mesh')
    parser.add_argument('--out', type=str, default="surf.vtk", help='Output filename')
    parser.add_argument('--property_file', type=str, default=None, help='.txt file with surface property')
    parser.add_argument('--property_name', type=str, default="Property", help='Name of property')
    

    args = parser.parse_args()


    main(args)
