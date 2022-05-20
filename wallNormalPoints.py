#GVivarelli 01-02-2021
#Script to produce normal points to wall from leading edge

#Modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as mh

#Set to 1 to plot all graps
view = 0

#Read in plate coordinates
coord = np.asanyarray(pd.read_csv("wallCoordinates.csv",skiprows=[0],header=None))

#Remove duplicate values
coord = np.unique(coord,axis=0)
x = coord[:,0]
y = coord[:,1]

#Order the nodes clockwise
##################################################################################################
#Function from :https://stackoverflow.com/questions/58377015/counterclockwise-sorting-of-x-y-data#
##################################################################################################
dist2 = lambda a,b: (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1])

z = list(zip(x, y)) # get the list of coordinate pairs
z.sort()            # sort by x coordinate

cw = z[0:1]  # first point in clockwise direction
ccw = z[1:2] # first point in counter clockwise direction
# reverse the above assignment depending on how first 2 points relate
if z[1][1] > z[0][1]: 
    cw = z[1:2]
    ccw = z[0:1]

for p in z[2:]:
    # append to the list to which the next point is closest
    if dist2(cw[-1], p) < dist2(ccw[-1], p):
        cw.append(p)
    else:
        ccw.append(p)

cw.reverse()
result = cw + ccw

#Convert to separate x,y vetcors
temp = np.array(result)
x = temp[:,0]
y = temp[:,1]
#####
#End#
#####

#Order staring from (0,0) -- i.e. LE
indexX = np.where(x==0.0)
indexY = np.where(y==0.0)
index = np.intersect1d(indexX,indexY)
x = np.roll(x,-index-1)
y = np.roll(y,-index-1)

#Remove the pressure side
#index = np.where(y>=0.0)
#x = x[index]
#y = y[index]

if view==1:
      plt.plot(x,y)
      plt.show()

#Determine the normal of each edge
norms = np.zeros((len(x)-1,2))
for i in range(0,len(x)-1):
     norms[i,1] = -(x[i+1]-x[i])  #y 
     norms[i,0] = (y[i+1]-y[i])   #x
     mag = np.sqrt(norms[i,0]**2+norms[i,1]**2)
     norms[i,0] = norms[i,0]/mag
     norms[i,1] = norms[i,1]/mag 

#Determine magnitude of each edge
length = np.zeros(len(x)-1)
for i in range(0,len(x)-1):
     dx = x[i+1]-x[i]
     dy = y[i+1]-y[i]
     length[i] = np.sqrt(dx**2+dy**2)

#Determine the norm of each node using edge length weighting
pointNorm = np.zeros((len(x),2))
for i in range(1,len(x)-1):
     accLength = length[i-1]+length[i]
     ratio1 = length[i-1]/accLength
     ratio2 = length[i]/accLength
     pointNorm[i,0] = ratio1*norms[i-1,0]+ratio2*norms[i,0]
     pointNorm[i,1] = ratio1*norms[i-1,1]+ratio2*norms[i,1]
     mag = np.sqrt(pointNorm[i,0]**2+pointNorm[i,1]**2)
     pointNorm[i,0] = pointNorm[i,0]/mag
     pointNorm[i,1] = pointNorm[i,1]/mag     
#Set the 1st and last points' norm
pointNorm[0,0] = norms[0,0]
pointNorm[0,1] = norms[0,1]
pointNorm[len(x)-1,0] = norms[len(x)-2,0]
pointNorm[len(x)-1,1] = norms[len(x)-2,1]

#Plot the norms
if view==1:
     plt.quiver(x, y, pointNorm[:,0], pointNorm[:,1], color=['k'])
     plt.plot(x,y,'b')
     plt.plot(x,y,'ro')
     plt.show() 

#Determine the lines normal to the wall surface and rotate them by the angle with the norm
nNormalPoints = 8000
yRange = 0.8
coords = np.zeros((len(x)*nNormalPoints,2))
rotatedCoords = np.zeros((len(x)*nNormalPoints,2))
dY = yRange/nNormalPoints
theta = np.zeros(len(x))
for i in range(0,len(x)-1):
     angle = mh.acos(pointNorm[i,1]) #As all lines are perfectly vertical norm is (0,1) therefore only take y component of nodal norm
     theta[i] = np.degrees(angle)
     coords[i*nNormalPoints:(i+1)*nNormalPoints,0] = x[i]
#     dY = (y[i]+yRange)/nNormalPoints
     for j in range(0,nNormalPoints):
         coords[i*nNormalPoints+j,1] = y[i]+j*dY      
     #Apply rotation matrix
     if pointNorm[i,0]>=0.0: #and pointNorm[i,1]>0.0:
          rotatedCoords[i*nNormalPoints:(i+1)*nNormalPoints,0] = (coords[i*nNormalPoints:(i+1)*nNormalPoints,0]-x[i])*mh.cos(angle)+(coords[i*nNormalPoints:(i+1)*nNormalPoints,1]-y[i])*mh.sin(angle)+x[i]
          rotatedCoords[i*nNormalPoints:(i+1)*nNormalPoints,1] = -(coords[i*nNormalPoints:(i+1)*nNormalPoints,0]-x[i])*mh.sin(angle)+(coords[i*nNormalPoints:(i+1)*nNormalPoints,1]-y[i])*mh.cos(angle)+y[i]
     else:
          rotatedCoords[i*nNormalPoints:(i+1)*nNormalPoints,0] = (coords[i*nNormalPoints:(i+1)*nNormalPoints,0]-x[i])*mh.cos(angle)-(coords[i*nNormalPoints:(i+1)*nNormalPoints,1]-y[i])*mh.sin(angle)+x[i]
          rotatedCoords[i*nNormalPoints:(i+1)*nNormalPoints,1] = (coords[i*nNormalPoints:(i+1)*nNormalPoints,0]-x[i])*mh.sin(angle)+(coords[i*nNormalPoints:(i+1)*nNormalPoints,1]-y[i])*mh.cos(angle)+y[i]         

if view==1:
     ax = plt.axes(projection='3d')
     ax.plot3D(x,y,theta, 'gray')
     plt.show()

if view==1:
     plt.plot(rotatedCoords[:,0],rotatedCoords[:,1],'k.')
    # plt.plot(coords[:,0],coords[:,1],'r.')
     plt.plot(x,y,'b')
     plt.show()     

with open("points.csv", 'wb') as outFile:
    outFile.write(b'x,y\n')
    np.savetxt(outFile, rotatedCoords, delimiter=",")