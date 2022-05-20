import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate
from scipy.interpolate import UnivariateSpline

########################
#User-defined quantities
########################

aoa = np.radians(0.0)#-9.0)   #Angle of attack

#Dimensional conversion/physical quantities
lRef = 1.0               #[m]
uRef = 1.0               #[m/s]
Re = 1.5e5               #Reynolds number
nu = 1/Re                #Kinematic non-dim. incompressible visc. (non-dim.)

#Calculation constants
coeffThwaites = 0.45     #Literature also indicates 0.47
n = 0.99                 #Ratio of simulation to inviscid axial velocity to determine boundary layer limit
xMax = 1.0               #Consider half the chord length for Thwaites theta and Blackwelder

#Plot viewing option
plot = 0                 #Plot true/false (0/1)
save = 0                 #Display (0) or save (1) image

#############
#Read in data
#############
flow = np.asanyarray(pd.read_csv("reStressesBoundaryProfile.csv"))                  #Sampling points
coord = np.asanyarray(pd.read_csv("wallCoordinates.csv",skiprows=[0],header=None))  #Surface geometry

######################
#Clean and format data
######################
#Remove duplicate values
coord = np.unique(coord,axis=0)
x = coord[:,0]
y = coord[:,1]

#NOTE: only need to rotate the geometry surface coordinates, 
#the ones from the flow file will have been rotated already
#when generating the sampling lines 

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

#1st node index should be the LE
index = np.argmin(x)
x = np.roll(x, -index)
y = np.roll(y, -index)

#Determine interpolation data size
nAxial = int(len(x))
nNormal = int(len(flow[:,0])/nAxial)

#Sort the flow data array into a 3D matrix
#Order of data:  x,y,u,v,w,p,uu,uv,uw,vv,vw,ww
flow = np.reshape(flow,(nAxial,nNormal,len(flow[0]))) 

#Shift the array to have the LE as the 1st element
index = np.argmin(flow[:,0,0])
flow = np.roll(flow, -index, axis=0)

#Remove all the pressure side -- not really of interest
flow = np.delete(flow, np.where(flow[:,0,1]<0.0), axis=0)
x = x[np.where(y>=0.0)]
y = y[np.where(y>=0.0)]
#Remove first and last point for simplicity
flow = flow[1:-1,:,:]
#Number of axial position has now changed, normal has not
nAxial = flow.shape[0]

print("\n\n##############################")
print("Sampling points are %d x %d " % (nAxial,nNormal))
print("##############################\n")

#Rotate the aerofoil
a1 = np.cos(aoa)
a2 = np.sin(aoa)

tempX = x
tempY = y
x = tempX*a1-tempY*a2
y = tempX*a2+tempY*a1

#Rotate the entire domain
tempFlow = np.copy(flow)
#Rotate the 1st sampling point (i.e. on the aerofoil surface)
for i in range(0,nAxial):
     for j in range(0,nNormal):
           flow[i,j,0] = tempFlow[i,j,0]*a1-tempFlow[i,j,1]*a2
           flow[i,j,1] = tempFlow[i,j,0]*a2+tempFlow[i,j,1]*a1
           flow[i,j,2] = tempFlow[i,j,2]*a1-tempFlow[i,j,3]*a2 
           flow[i,j,3] = tempFlow[i,j,2]*a2+tempFlow[i,j,3]*a1

if plot==1:
    plt.plot(x,y,"r.")
    plt.plot(flow[:,0,0],flow[:,0,1],"k--") 
    plt.title("Aerofoil & Sampling Points")  
    plt.grid()
    if save==1:
         plt.savefig('geometrySamplingPoints.pdf')     
    else:
         plt.show()
    plt.close()

#Make static P positive
minP = np.min(flow[:,:,5])
flow[:,:,5] = flow[:,:,5]+abs(minP)

###################################################
#Now decompose the velocity into wall normal and tangential direction
#Recover the normal to the surface from the point distribution
norm = np.zeros((nAxial,2))
norm[:,0] = flow[:,1,0]-flow[:,0,0]           
norm[:,1] = flow[:,1,1]-flow[:,0,1]
magnitude = np.sqrt(norm[:,0]**2+norm[:,1]**2)
#magnitude[np.where(magnitude==0.0)] = 1e-10
norm[:,0] = norm[:,0]/magnitude
norm[:,1] = norm[:,1]/magnitude
##Recover tangent
#tan = np.zeros((nAxial,2))
#for i in range(1,flow.shape[0]):
#     tan[i,0] = -(flow[i,0,0]-flow[i-1,0,0])  #Minus due to reversed axial order
#     tan[i,1] = -(flow[i,0,1]-flow[i-1,0,1])
#tan[0,0] = tan[1,0]
#tan[0,1] = tan[1,1]
#magnitude = np.sqrt(tan[:,0]**2+tan[:,1]**2)
##magnitude[np.where(magnitude==0.0)] = 1e-10
#tan[:,0] = tan[:,0]/magnitude
#tan[:,1] = tan[:,1]/magnitude

#Determine angle
theta = np.arctan(norm[:,0]/norm[:,1])
cos = np.cos(theta)
sin = np.sin(theta)

#Rotate
tempFlow = np.copy(flow)
#for i in range(0,nAxial):
#    for j in range(0,nNormal):
#         flow[i,j,2] = tempFlow[i,j,2]*cos[i]-tempFlow[i,j,3]*sin[i]
#         flow[i,j,3] = tempFlow[i,j,2]*sin[i]+tempFlow[i,j,3]*cos[i]

#FROM NOW ON THE U and V ARE THE ROTATED QUANTITIES, I.E. U PARALLEL TO SURFACE
#AND U PERPENDICULAR TO LOCAL SURFACE POSITION. W DOES NOT NEED ROTATION
###################################################

#----------------------1 Boundary layer height and velocity (100%)

#Determine the boundary layer limit according to the method in:
#"General method for determining the boundary layer thickness in nonequilibrium flow", Griffin, Fu and Moin 2021
#Determine the total pressure profile along every vertical line 
pTot = np.zeros((nAxial,nNormal,1))
pTot[:,:,0] = flow[:,:,5]+0.5*(flow[:,:,2]**2+flow[:,:,3]**2)
flow = np.append(flow,pTot, axis=2)  

#Now determine the ratio of the actual axial velocity to the inviscid one
ratio = np.zeros(nAxial)
UI2 = np.zeros(nAxial)
blHeight = np.zeros(nAxial)
blIDHeight = np.zeros(nAxial,dtype=int)
blURef = np.zeros(nAxial) 
blVRef = np.zeros(nAxial) 
blX = np.zeros(nAxial)    
blY = np.zeros(nAxial)  
for i in range(0,nAxial):
     maxp0 = max(flow[i,:,-1])
     U2 = flow[i,:,2]**2
     V2 = flow[i,:,3]**2
     W2 = flow[i,:,4]**2
     P = flow[i,:,5]
     UI2 = 2*(maxp0-P)-V2-W2
     UI2[np.where(UI2 == 0)] = 1e-10     #To avoid dividing by zero plug a small value -- this occurs at the TE lines at the node on the wall
     ratio = np.sqrt(np.abs(U2/UI2))     #To avoid square-root of negative values -- this occurs at the LE where the interpolation has collapsed all nodes onto the surface      
     location = np.argmin(abs(ratio-n))
     #Round-off the ratio to 2 decimal places
#     location = np.argmin(abs(np.round(ratio,2)-n))
     blHeight[i] = np.sqrt((flow[i,location,1]-flow[i,0,1])**2+(flow[i,location,0]-flow[i,0,0])**2)
     blURef[i] = flow[i,location,2]
     blVRef[i] = flow[i,location,3]
     blX[i] = flow[i,location,0]
     blY[i] = flow[i,location,1]
     blIDHeight[i] = location

if plot!=0:
     plt.plot(blX,blHeight,'r.',linewidth=5,label="Suction Side")
     plt.xlabel('Axial Position', fontsize=20)
     plt.ylabel('Boundary Layer Height', fontsize=20)
     plt.legend()
     plt.grid()
     plt.xticks(fontsize=20)
     plt.yticks(fontsize=20)
     if save==1:
           fig = plt.gcf()
           fig.set_size_inches(30, 7)
           plt.savefig('boundaryLayerHeight_n'+str(n*100)+'.pdf')
     else:
           plt.show()
     plt.close()

     plt.plot(blX,blURef,'r.',linewidth=5,label="Suction Side")
     plt.xlabel('Axial Position', fontsize=15)
     plt.ylabel('Local reference velocity', fontsize=15)
     plt.legend()
     plt.grid()
     plt.xticks(fontsize=15)
     plt.yticks(fontsize=15)
     if save==1:
           fig = plt.gcf()
           fig.set_size_inches(30, 7)
           plt.savefig('localReferenceVelocity_n'+str(int(n*100))+'.pdf')
     else:
           plt.show()
     plt.close()

print("\nDone boundary layer height and reference axial velocity\n")

#Write out the boundary layer height and reference velocity
output = np.zeros((nAxial,5))
for i in range(0,nAxial):
     output[i,0] = blX[i]*lRef               #x
     output[i,1] = blY[i]*lRef               #y 
     output[i,2] = blHeight[i]*lRef          #Height magnitude
     output[i,3] = blURef[i]*uRef            #Reference velocity - wall tangential
     output[i,4] = blVRef[i]*uRef            #Reference velocity - wall normal
nameOut = "blHeightUref_n"+str(int(n*100))+".csv"
csv_header = "x,y,height,uRef,vRef"
file_header = csv_header.split(',')
outData = pd.DataFrame(output)
outData.to_csv(nameOut,index=False,header=file_header)

print("\nWritten dimensional estimated boundary layer height and axial velocity to .csv\n")

#----------------------2 Boundary layer displacement thickness, momentum thickness, momentum thickness Re and shape factor

#Displacement thickness
integrand = np.zeros(nNormal)
deltaStar = np.zeros(nAxial)
#Use last two points to determine the integration length, 
#ought to consider x and y contribution as not aligned 
#with carteisan axis but the normal direction
integrationLength = np.sqrt((y[len(y)-1]-y[len(y)-2])**2+(x[len(x)-1]-x[len(x)-2])**2) #Points are equidistant
for i in range(0,nAxial):
     for j in range(0,blIDHeight[i]):
         integrand[j] = 1.0-(flow[i,j,2]/blURef[i])
     if blIDHeight[i]>0:
         height = np.sqrt((flow[i,0:blIDHeight[i],0]-flow[i,0,0])**2+(flow[i,0:blIDHeight[i],1]-flow[i,0,1])**2) #Remember to consider the x,y contributions as sampling lines not aligned with cartesian axis
         deltaStar[i] = scipy.integrate.simps(integrand[0:blIDHeight[i]],height,dx=integrationLength)
     else:
         deltaStar[i] = 0.0

plottingX = flow[:,0,0]
plottingY = flow[:,0,1]

if plot!=0:
     plt.plot(plottingX,deltaStar,'r.',linewidth=5,label="Suction Side")
     plt.xlabel('Normalised Axial Position')
     plt.ylabel('$\delta^*$')
     plt.title('Displacement Thickness')
     plt.grid()
     plt.legend()
     if save==1:
           plt.savefig('deltaStar_n'+str(int(n*100))+'.pdf') 
     else:
           plt.show()
     plt.close()           

print("\nDone Displacement Thickness\n")

#Momentum thickness
integrand = np.zeros(nNormal)
theta = np.zeros(nAxial)
for i in range(0,nAxial):
     for j in range(0,blIDHeight[i]):
         integrand[j] = (1-(flow[i,j,2]/blURef[i]))*(flow[i,j,2]/blURef[i])
     if blIDHeight[i]>0:
         height = np.sqrt((flow[i,0:blIDHeight[i],0]-flow[i,0,0])**2+(flow[i,0:blIDHeight[i],1]-flow[i,0,1])**2) #Remember to consider the x,y contributions as sampling lines not aligned with cartesian axis
         theta[i] = scipy.integrate.simps(integrand[0:blIDHeight[i]],height,dx=integrationLength)
     else:
         theta[i] = 0.0
   
if plot!=0:
     plt.plot(plottingX,theta,'r.',linewidth=5,label="Suction Side")
     plt.legend()
     plt.xlabel('Normalised Axial Position')
     plt.ylabel(r'$\theta$')
     plt.title('Momentum Thickness')
     plt.grid()
     if save==1:
           plt.savefig('theta_n'+str(int(n*100))+'.pdf')  
     else:
           plt.show()
     plt.close()     

print("\nDone Momentum Thickness\n")

#Momentum thickness Reynolds number
thetaRe = np.zeros(nAxial)
thetaRe = theta*blURef/nu

if plot!=0: 
     plt.plot(plottingX,thetaRe,'r.',linewidth=5,label="Suction Side")
     plt.xlabel('Normalised Axial Position')
     plt.ylabel(r'$Re_\theta$')
     plt.title('Momentum Thickness Reynolds number')
     plt.grid()    
     plt.legend()
     if save==1:
           plt.savefig('thetaRe_n'+str(int(n*100))+'.pdf')  
     else:
           plt.show()
     plt.close()

print("\nDone Momentum Thickness Reynolds number\n")     

#Shape factor  
#Set to 0.0 values where theta is very small
#Make sure to use the absolute value as I get
#negative values for the pressure side 
#(negative y no other physical reason)
index = np.where(abs(theta)<1e-10)
theta[index] = 1e-10
H = deltaStar/theta

if plot!=0:
     plt.plot(plottingX,H,'r.',linewidth=5,label="Suction Side") 
     plt.xlabel('Normalised Axial Position')
     plt.ylabel('H')
     plt.legend()
     plt.title('Shape Factor')
     plt.grid()
     if save==1:
           plt.savefig('H_n'+str(int(n*100))+'.pdf')  
     else:
           plt.show()
     plt.close()             

print("\nDone Shape Factor\n")

#Write out the displacement thickness, momentum thickness and shape factor
dataOut = np.zeros((len(theta),5))
for i in range(0,len(theta)):
     dataOut[i,0] = flow[i,0,0]*lRef
     dataOut[i,1] = deltaStar[i]*lRef
     dataOut[i,2] = theta[i]*lRef
     dataOut[i,3] = thetaRe[i]
     dataOut[i,4] = H[i]
     
file_out = 'dispMomentThickReHFact_n'+str(int(n*100))+'.csv'
csv_header = "x,deltaStar,Theta,ReTheta,H"
file_header = csv_header.split(',')
outData = pd.DataFrame(dataOut)
outData.to_csv(file_out,index=False,header=file_header)     

print("\nWritten estimated dimensional boundary layer displacement thickness, momentum thickness, momentum thickness Re and shape factor to .csv\n")

#----------------------3 Boundary layer Thwaites' momentum thickness

xMin = min(x)
index = np.intersect1d(np.where(flow[:,0,0]>=xMin),np.where(flow[:,0,0]<xMax))
pointX = flow[index,0,0]
pointY = flow[index,0,1]
blURefTemp = blURef[index]

#Spline can only be used for increasing x
index = np.argsort(pointX)
pointX = pointX[index]
pointY = pointY[index]
blURefTemp = blURefTemp[index]

if plot!=0:
     plt.plot(pointX,pointY,'b',label="Suction Side")
     plt.legend()
     plt.xlabel("Normalised Axial Position")
     plt.ylabel("LE to Mid-Chord Aerofoil")
     plt.title("LE to Mid-Chord")
     plt.grid()
     if save==1:
           plt.savefig("leToMidChord_n"+str(int(n*100))+".pdf")
     else:
           plt.show()
     plt.close() 

#Reconstruct velocity Uref with spline
fitSplineSS = UnivariateSpline(pointX[:],blURefTemp,s=0.0)

#Determine the x-integration momentum thickness 
axialPositions = np.arange(0.0,xMax,0.001)
axialThetaSS = np.zeros(len(axialPositions))
for i in range(0,len(axialPositions)):
     index = np.arange(0,i+1)
     #Suction side
     integralSS = scipy.integrate.simps(fitSplineSS(axialPositions[index])**5,axialPositions[index])
     numeratorSS = coeffThwaites*integralSS*nu
     denominatorSS = fitSplineSS(axialPositions[i])**6
     axialThetaSS[i] = np.sqrt(numeratorSS/denominatorSS)
     
if plot!=0:
     plt.close()
     plt.plot(axialPositions,axialThetaSS,"b",label="Suction Side")
     plt.grid()
     plt.xlabel('Axial Position')
     plt.ylabel('Thwaites Momentum Thickness')
     plt.legend()
     if save==1:
           plt.savefig('ThwaitesMomentumThickness_n'+str(int(n*100))+'.pdf')
     else:
           plt.show()      

print("\nDone Thwaites Momentum Thickness\n")

#Write out Thwaites momentum thickness and Thwaites Re
thwaitesThetaRe = np.zeros((len(axialPositions),3))
for i in range(0,len(axialPositions)):
     thwaitesThetaRe[i,0] = axialPositions[i]*lRef
     thwaitesThetaRe[i,1] = axialThetaSS[i]*lRef
     thwaitesThetaRe[i,2] = axialThetaSS[i]*fitSplineSS(axialPositions[i])/nu
file_out = 'thwaitesThetaRe_n'+str(int(n*100))+'.csv'
csv_header = "x,ThetaSS,ReThetaSS"
file_header = csv_header.split(',')
outData = pd.DataFrame(thwaitesThetaRe)
outData.to_csv(file_out,index=False,header=file_header)

print("\nWritten estimated dimensional boundary layer Thwaites momentum thickness and Thwaites momentum thickness Re to .csv\n")
print("\n\n##############################\n")

#---------------------------------------------------------------------------
#Adding the Blackwelder calculation based on averaged turbulent shear stresses
#NOTE: I have to convert the averaged shear stresses from simple averaging to RMS
print("\n\nDetermining the Blackwelder parameter\n")

#Remove the negative values (don't know why Nektar++ outputs negative u'u', how is that possible?)
#square and then square-root to remove negative values
flow[:,:,6] = np.sqrt(flow[:,:,6]**2)    #uu
flow[:,:,9] = np.sqrt(flow[:,:,9]**2)    #vv
flow[:,:,11] = np.sqrt(flow[:,:,11]**2)  #ww

#Convert from standard average (Nektar++) to RMS
flow[:,:,6] = np.sqrt(flow[:,:,6])
flow[:,:,9] = np.sqrt(flow[:,:,9])
flow[:,:,11] = np.sqrt(flow[:,:,11])

avgTurbTau = np.zeros((nAxial,nNormal,1))
avgTurbTau[:,:,0] = 0.5*(flow[:,:,6]+flow[:,:,9])
flow = np.append(flow,avgTurbTau, axis=2)  

blackwelder = np.zeros(nAxial)
height = np.zeros(nNormal)
X = np.ones(nAxial)*-1e10
Y = np.zeros(nAxial)
for i in range(0,nAxial):
     if flow[i,0,0]>xMax or blHeight[i]<1e-10:        #Plot between LE and xMax, avoid the nodes with 0 boundary layer height
           continue
     height = np.sqrt((flow[i,0:blIDHeight[i],0]-flow[i,0,0])**2+(flow[i,0:blIDHeight[i],1]-flow[i,0,1])**2) #Remember to consider the x,y contributions as sampling lines not aligned with cartesian axis
     limits = height[0:blIDHeight[i]]
     param = flow[i,0:blIDHeight[i],-1]
     integral=scipy.integrate.simps(param,limits)
     blackwelder[i] = integral/(blURef[i])**2
     X[i] = flow[i,0,0]
     Y[i] = flow[i,0,1]

#Remove values where x>0.5
index = np.where(X!=-1e10)
X = X[index]
Y = Y[index]
blackwelder = blackwelder[index]

if plot!=0:
     plt.plot(X,blackwelder,"r",linewidth=5,label="Suction Side")     
     plt.xlabel('Axial Position', fontsize=20)
     plt.ylabel('Blackwelder\'s Parameter', fontsize=20)
     plt.grid()
     plt.xticks(fontsize=20)
     plt.yticks(fontsize=20)
     if save==1:
           fig = plt.gcf()
           fig.set_size_inches(30, 7)
           plt.savefig('blackwelderParameter_n'+str(int(n*100))+'.pdf')
     else:
           plt.show()
     plt.close()           

#Write out Blackwelder's parameter
blackwelderOut = np.zeros((len(blackwelder),2))
for i in range(0,len(blackwelder)):
     blackwelderOut[i,0] = X[i]
     blackwelderOut[i,1] = blackwelder[i]
file_out = "blackwelder_n"+str(int(n*100))+".csv"
csv_header = "x,blackwelder"
file_header = csv_header.split(',')
outData = pd.DataFrame(blackwelderOut)
outData.to_csv(file_out,index=False,header=file_header)