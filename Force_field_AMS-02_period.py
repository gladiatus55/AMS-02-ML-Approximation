import math


# References
# [1] https://cosmicrays.oulu.fi/phi/phi.html
# [2] http://cc.oulu.fi/~usoskin/personal/Owens_SP_Phi_2024.pdf
# [3] http://cc.oulu.fi/~usoskin/personal/Vaisanen_JGR_2023.pdf


m0 = 1.67261e-27  # proton rest mas [kg]
q = 1.60219e-19   # Elementary charge [C]
c = 2.99793e8     # speed of light [m/s]

T0w = m0 * c * c           # [Joule]
T0 = m0 * c * c/(q*1e9);   # [GeV]


# leap year identification
def is_leap_year(year):
    """ if year is a leap year return True
        else return False """
    if year % 100 == 0:
        return year % 400 == 0
    return year % 4 == 0

# Day of the Year function
def doy(Y,M,D):
    """ given year, month, day return day of year
        Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7 """
    if is_leap_year(Y):
        K = 1
    else:
        K = 2
    N = int((275 * M) / 9.0) - K * int((M + 9) / 12.0) + D - 30
    return N


# Rigidity in GV, fi in GV
def force_field(Rig,fi):
    m0 = 1.67261e-27  # proton rest mas [kg]
    q = 1.60219e-19   # Elementary charge [C]
    c = 2.99793e8     # speed of light [m/s]
    T0w = m0 * c * c           # [Joule]
    T0 = m0 * c * c/(q*1e9);   # [GeV]
    Tr = 0.938  # GeV

    T = math.sqrt((T0*T0*q*q*1e9*1e9) + (q*q*Rig*Rig*1e9*1e9)) - (T0*q*1e9)       # Kinetic energy [Joule]
    T = T/(q*1e9)                                                                 # Kinetic energy [GeV]
    
    # evaluation of intensity in Force field approximation
    T = T + fi
 
    tt = T + Tr
    t2 = tt + Tr
    beta = math.sqrt(T*t2)/tt
    
    # equation (2) from [3]
    tem = (T + 0.67)/1.67
    tem = math.exp(-3.93*math.log(tem))
    JLIS = 2.7e3 * math.exp(1.12*math.log(T)) * tem
    JLIS = JLIS / (beta *beta)
    
    T = math.sqrt((T0*T0*q*q*1e9*1e9) + (q*q*Rig*Rig*1e9*1e9)) - (T0*q*1e9)       # Joule
    T = T/(q*1e9)

    # JGR 2005 and JGR 2023 [3]
    # equation (1) from [3]
    J = JLIS * (T*(T + (2.0*Tr)))
    J = J / (((T+fi)*(T+fi+(2.0*Tr))))    # J and T are given in (m2 s sr GeV/nuc)−1 and GeV/nuc, respectively. 
    
    # rebinning from GeV bins to GV bins
    factor = (T + T0)/(T*(T+(2*T0)))**0.5  # from flux per GeV to flux per GV 
    JGV = J/factor
    
    return J,JGV # J in (m2 s sr GeV/nuc)−1 ; JGV in (m2 s sr GV/nuc)−1


# AMS bins borders in GV

Nbins = 31

AMS_rig_min_bins_GV = [0]*Nbins

AMS_rig_min_bins_GV[0] = 1.00
AMS_rig_min_bins_GV[1] = 1.16
AMS_rig_min_bins_GV[2] = 1.33
AMS_rig_min_bins_GV[3] = 1.51
AMS_rig_min_bins_GV[4] = 1.71
AMS_rig_min_bins_GV[5] = 1.92
AMS_rig_min_bins_GV[6] = 2.15
AMS_rig_min_bins_GV[7] = 2.40
AMS_rig_min_bins_GV[8] = 2.67
AMS_rig_min_bins_GV[9] = 2.97
AMS_rig_min_bins_GV[10] = 3.29
AMS_rig_min_bins_GV[11] = 3.64
AMS_rig_min_bins_GV[12] = 4.02
AMS_rig_min_bins_GV[13] = 4.43
AMS_rig_min_bins_GV[14] = 4.88
AMS_rig_min_bins_GV[15] = 5.37
AMS_rig_min_bins_GV[16] = 5.90
AMS_rig_min_bins_GV[17] = 6.47
AMS_rig_min_bins_GV[18] = 7.09
AMS_rig_min_bins_GV[19] = 7.76
AMS_rig_min_bins_GV[20] = 8.48
AMS_rig_min_bins_GV[21] = 9.26
AMS_rig_min_bins_GV[22] = 10.10
AMS_rig_min_bins_GV[23] = 11.00
AMS_rig_min_bins_GV[24] = 13.00
AMS_rig_min_bins_GV[25] = 16.60
AMS_rig_min_bins_GV[26] = 22.80
AMS_rig_min_bins_GV[27] = 33.50
AMS_rig_min_bins_GV[28] = 48.50
AMS_rig_min_bins_GV[29] = 69.70
AMS_rig_min_bins_GV[30] = 100.0




import pandas as pd
import numpy as np

# modulation potential Phi, Oulu, from [1] and [2]

import matplotlib.pyplot as plt
# read the AMS-02 data set to data frame df
dfPhi = pd.read_csv(
    "Phi_daily_1964-2021_Vaisanen23.csv"
    )

yearPhi = np.array(dfPhi['Year'])
monthPhi = np.array(dfPhi['Month'])
dayPhi = np.array(dfPhi['Day'])
Phi = np.array(dfPhi['Modulation potential [MV]'])

N = len(yearPhi)
PhiDOYtem = [0]*N

Np1 = -1

Np = 3085  # number of days in AMS-02 measurements period

yearPhiAMSperiod = [0]*Np
monthPhiAMSperiod = [0]*Np
dayPhiAMSperiod = [0]*Np
doyPhiAMSperiod = [0]*Np
PhiAMSperiod = [0]*Np

f = open('Phi.csv', "w")
s = 'year,doy,phi\n'
f.write(s)
for i in range(0,N):
    PhiDOYtem[i] = doy(yearPhi[i],monthPhi[i],dayPhi[i])
    if yearPhi[i]>2010:
        if yearPhi[i]==2011:
            if PhiDOYtem[i]>139:
                Np1 = Np1 + 1
        if yearPhi[i]>2011:
            Np1 = Np1 + 1
        if Np1<Np:
            yearPhiAMSperiod[Np1] = yearPhi[i]
            monthPhiAMSperiod[Np1] = monthPhi[i]
            dayPhiAMSperiod[Np1] = dayPhi[i]
            doyPhiAMSperiod[Np1] = PhiDOYtem[i]
            PhiAMSperiod[Np1] = Phi[i]
            s = str(yearPhiAMSperiod[Np1]) + ',' + str(doyPhiAMSperiod[Np1]) + ',' + str(PhiAMSperiod[Np1]) + '\n'
            f.write(s)


f.close()            
print(Np1)
            
# results fields    
AMS02_force_field_GV = [[0 for i in range(Nbins)] for j in range(Np)]
AMS02_force_field_GeV = [[0 for i in range(Nbins)] for j in range(Np)]

f1 = open('AMS02_force_field_GV.csv', "w")
f2 = open('AMS02_force_field_GeV.csv', "w")

for i in range(0,Np):
    s1 = str(yearPhiAMSperiod[i]) + ',' + str(doyPhiAMSperiod[i]) 
    s2 = s1
    fi = PhiAMSperiod[i]/1000.
    for j in range(Nbins-1):
        # bin center 
        Rig = (AMS_rig_min_bins_GV[j]+AMS_rig_min_bins_GV[j+1])/2.0
        # bin min
        # Rig = AMS_rig_min_bins_GV[j]
        # bin max
        # Rig = AMS_rig_min_bins_GV[j+1]
        J, JGV = force_field(Rig,fi)
        AMS02_force_field_GV[i][j] = JGV
        s1 = s1 + ',' + str(AMS02_force_field_GV[i][j]) 
        AMS02_force_field_GeV[i][j] = J
        s2 = s2 + ',' + str(AMS02_force_field_GeV[i][j]) 
    s1 = s1 + '\n'
    s2 = s2 + '\n'
    f1.write(s1)
    f2.write(s2)

f1.close()
f2.close()



# To take in account bin withd and didide bins to subbins to improve estimation of intensity in AMS-02 bins

# results fields    
AMS02_force_field_GV_binfix = [[0 for i in range(Nbins)] for j in range(Np)]
AMS02_force_field_GeV_binfix = [[0 for i in range(Nbins)] for j in range(Np)]

f1 = open('AMS02_force_field_GV_binfix.csv', "w")

"""
AMS_rig_min_bins_GV[0] = 1.00
AMS_rig_min_bins_GV[1] = 1.16
...
"""

dR = 0.001  # GV    # 0.01 for faster run

for i in range(0,Np):  
    #print(i)
    if i%50==0:
        print("Day ",i)
    s1 = str(yearPhiAMSperiod[i]) + ',' + str(doyPhiAMSperiod[i]) 
    fi = PhiAMSperiod[i]/1000.  
    for j in range(Nbins-1):
        Nsubbins = ((AMS_rig_min_bins_GV[j+1]-AMS_rig_min_bins_GV[j])/dR)
        Nsubbins = round(Nsubbins)
        for k in range(Nsubbins):
            Rig = AMS_rig_min_bins_GV[j] + (k*dR)
            J, JGV = force_field(Rig,fi)
            AMS02_force_field_GV[i][j] = AMS02_force_field_GV[i][j] + JGV
            
        AMS02_force_field_GV[i][j] = AMS02_force_field_GV[i][j]/Nsubbins
        s1 = s1 + ',' + str(AMS02_force_field_GV[i][j]) 
        
    s1 = s1 + '\n'
    f1.write(s1)
    
f1.close()


