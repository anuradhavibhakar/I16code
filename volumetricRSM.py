import numpy as np
import math
from i16experiment import volumetricRSM as VRSM
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mypy import mypy
###these use the nexus files
def gethkl(filepath, h_ind, k_ind, l_ind):
    h5 = h5py.File(filepath, 'r')
    vol = h5.get('processed/reciprocal_space/volume')
    h = h5.get('processed/reciprocal_space/h-axis')
    k = h5.get('processed/reciprocal_space/k-axis')
    l = h5.get('processed/reciprocal_space/l-axis')
    weight = h5.get('processed/reciprocal_space/weight')
    hh, kk, ll = h[h_ind], k[k_ind], l[l_ind]
    I = vol[h_ind,k_ind,l_ind]
    # print("{:f}, {:f}, {:f}".format(h_ind,k_ind,l_ind)
    # print("{:f}, {:f}, {:f}".format(h[h_ind],k[k_ind],l[l_ind])
    # print("I = {:f}\n".format(vol[(h_ind,k_ind,l_ind)])
    return [hh,kk,ll]

def getslice(filepath, hval,kval,lval,slicingalong):
    h5 = h5py.File(filepath, 'r')
    vol = h5.get('processed/reciprocal_space/volume')
    h = h5.get('processed/reciprocal_space/h-axis')
    k = h5.get('processed/reciprocal_space/k-axis')
    l = h5.get('processed/reciprocal_space/l-axis')
    I = vol[:,:,:]

    hind = np.where(np.abs(h[:] - hval) == np.min(np.abs(h[:]-hval)))
    kind = np.where(np.abs(k[:] - kval) == np.min(np.abs(k[:]-kval)))
    lind = np.where(np.abs(l[:] - lval) == np.min(np.abs(l[:]-lval)))

    if slicingalong == 'h':
        xdata = h[:]
        ydata = I[:, kind, lind]

    elif slicingalong == 'k':
        xdata = k[:]
        ydata = I[hind, :, lind]

    elif slicingalong == 'l':
        xdata = l[:]
        ydata = I[hind,kind, :]

    return xdata, ydata.flatten()


def hklslices(filepath):
    h5 = h5py.File(filepath, 'r')
    vol = h5.get('processed/reciprocal_space/volume')
    h = h5.get('processed/reciprocal_space/h-axis')
    k = h5.get('processed/reciprocal_space/k-axis')
    l = h5.get('processed/reciprocal_space/l-axis')
    I = vol[:,:,:]

    I_h = I.sum(axis=1).sum(axis=1)
    I_k = I.sum(axis=0).sum(axis=1)
    I_l = I.sum(axis=0).sum(axis=0)

    fig,ax = plt.subplots(1,3, figsize=(20,5))

    Ih = []
    for hindex in np.arange(0,len(h[:]), 1):
        Ih += [np.sum(I[hindex,:,:])]

    Ik = []
    for kindex in np.arange(0,len(k[:]), 1):
        Ik += [np.sum(I[:,kindex,:])]

    Il = []
    for lindex in np.arange(0,len(l[:]), 1):
        Il += [np.sum(I[:,:,lindex])]

    ax[0].plot(h[:], Ih)
    ax[1].plot(k[:], Ik)
    ax[2].plot(l[:], Il)


def getsumI(magscans,direc, stringfname):
    allI = []
    for scan in magscans:
        filepath = "{:s}\{:d}-{:s}.nxs".format(direc, scan, stringfname)
        h5 = h5py.File(filepath, 'r')
        vol = h5.get('processed/reciprocal_space/volume')
        h = h5.get('processed/reciprocal_space/h-axis')
        k = h5.get('processed/reciprocal_space/k-axis')
        l = h5.get('processed/reciprocal_space/l-axis')
        weight = h5.get('processed/reciprocal_space/weight')
        I = vol[:,:,:]
        allI += [np.sum(I)]
    return allI

def getmax(filepath):
    h5 = h5py.File(filepath, 'r')
    vol = h5.get('processed/reciprocal_space/volume')
    h = h5.get('processed/reciprocal_space/h-axis')
    k = h5.get('processed/reciprocal_space/k-axis')
    l = h5.get('processed/reciprocal_space/l-axis')
    I = vol[:,:,:]
    Imax = np.max(I)
    hind, kind, lind = np.where(I==Imax)
    hmax, kmax, lmax = h[hind[0]], k[kind[0]], l[lind[0]]
    return hmax, kmax, lmax

def getmaxind(filepath):
    h5 = h5py.File(filepath, 'r')
    vol = h5.get('processed/reciprocal_space/volume')
    h = h5.get('processed/reciprocal_space/h-axis')
    k = h5.get('processed/reciprocal_space/k-axis')
    l = h5.get('processed/reciprocal_space/l-axis')
    I = vol[:,:,:]
    Imax = np.max(I)
    hind, kind, lind = np.where(I==Imax)
    return hind, kind, lind

def getIHKL(filepath):
    h5 = h5py.File(filepath, 'r')
    vol = h5.get('processed/reciprocal_space/volume')
    h = h5.get('processed/reciprocal_space/h-axis')
    k = h5.get('processed/reciprocal_space/k-axis')
    l = h5.get('processed/reciprocal_space/l-axis')
    I = vol[:,:,:]
    return I, h, k, l

#for h data3axis = 0, for k data3axis = 1, for l data3axis = 2
#for hk map data3axis = 2.
def dataforRSM_2d(filepath, data1label, data2label, data3axis):
    h5 = h5py.File(filepath, 'r')
    vol = h5.get('processed/reciprocal_space/volume')
    h = h5.get('processed/reciprocal_space/h-axis')
    k = h5.get('processed/reciprocal_space/k-axis')
    l = h5.get('processed/reciprocal_space/l-axis')
    weight = h5.get('processed/reciprocal_space/weight')
    I = vol[:,:,:]

    dict_hkl = {'h':h[:],'k':k[:],'l':l[:]}
    data1 = dict_hkl[data1label]
    data2 = dict_hkl[data2label]

    xdata, ydata = [], []
    i = 0
    for xval in data1:
        xdata += [data1[i]]*len(data2)
        i+=1
    ydata = np.asarray(data2.tolist()*len(data1))
    zdata = np.sum(I, axis=data3axis).flatten()

    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)

    X, Y = np.meshgrid(xi, yi)
    I = griddata((xdata, ydata), zdata, (X, Y))

    return X, Y, I


def dataforRSM_3d(filepath):
    h5 = h5py.File(filepath, 'r')
    vol = h5.get('processed/reciprocal_space/volume')
    h = h5.get('processed/reciprocal_space/h-axis')
    k = h5.get('processed/reciprocal_space/k-axis')
    l = h5.get('processed/reciprocal_space/l-axis')
    weight = h5.get('processed/reciprocal_space/weight')
    I = vol[:,:,:]

    # dict_hkl = {'h':h[:],'k':k[:],'l':l[:]}
    # data1 = dict_hkl[data1label]
    # data2 = dict_hkl[data2label]

    xdata, ydata, zdata = [], [], []
    i = 0
    for xval in data1:
        xdata += [data1[i]]*len(data2)
        i+=1
    ydata = np.asarray(data2.tolist()*len(data1))
    zdata = np.sum(I, axis=data3axis).flatten()

    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)

    X, Y = np.meshgrid(xi, yi)
    I = griddata((xdata, ydata), zdata, (X, Y))

    return X, Y, I




def dataforRSM_hk(filepath):
    h5 = h5py.File(filepath, 'r')
    vol = h5.get('processed/reciprocal_space/volume')
    h = h5.get('processed/reciprocal_space/h-axis')
    k = h5.get('processed/reciprocal_space/k-axis')
    l = h5.get('processed/reciprocal_space/l-axis')
    weight = h5.get('processed/reciprocal_space/weight')
    I = vol[:,:,:]

    dict_hkl = {'h':h[:],'k':k[:],'l':l[:]}
    data1 = dict_hkl['h']
    data2 = dict_hkl['k']

    xdata, ydata = [], []
    i = 0
    for xval in data1:
        xdata += [data1[i]]*len(data2)
        i+=1
    ydata = np.asarray(data2.tolist()*len(data1))
    zdata = np.sum(I, axis=2).flatten()

    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)

    X, Y = np.meshgrid(xi, yi)
    I = griddata((xdata, ydata), zdata, (X, Y))
    return X, Y, I

def allrawdataRSM(filepath):
    h5 = h5py.File(filepath, 'r')
    vol = h5.get('processed/reciprocal_space/volume')
    h = h5.get('processed/reciprocal_space/h-axis')
    k = h5.get('processed/reciprocal_space/k-axis')
    l = h5.get('processed/reciprocal_space/l-axis')
    weight = h5.get('processed/reciprocal_space/weight')
    I = vol[:,:,:]

    hallflat, kallflat, intensity = [], [], []

    for hind in np.arange(0, len(h), 1):
        for kind in np.arange(0, len(k), 1):
            intensity += [np.sum(vol[hind,kind,:])]
            hallflat += [h[hind]]
            kallflat += [k[kind]]

    zdata = intensity
    xdata, ydata = hallflat, kallflat

    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)

    X, Y = np.meshgrid(xi, yi)
    I = griddata((xdata,ydata), zdata, (X, Y))

    return X, Y, I, xdata, ydata, zdata


def extractsmallervolumeofVRSM(filepath, hklcentre, hrange, krange, lrange, data1label, data2label):
    I, h, k, l= getIHKL(filepath)
    ###getting centre of box
    X, Y, I = VRSM.dataforRSM_hk(filepath)
    I, h, k, l = VRSM.getIHKL(filepath)
    newhkl_cen_ind = []
    for oldcen, axis in zip(hklcentre, [h[:], k[:], l[:]]):
        diff = np.abs(np.abs(axis)-(np.abs(oldcen)))
        newcen = np.where(diff == np.min(diff))[0][0]
        newhkl_cen_ind += [newcen]
    hind, kind, lind = newhkl_cen_ind

    ####getting range
    hstep = h[1] - h[0]
    kstep = k[1] - k[0]
    lstep = l[1] - l[0]

    hrangeind = int(hrange/hstep)
    krangeind = int(krange/kstep)
    lrangeind = int(lrange/lstep)

    hnew = h[hind-hrangeind:hind+hrangeind]
    knew = k[kind-krangeind:kind+krangeind]
    lnew = l[lind-lrangeind:lind+lrangeind]
    Inew=I[hind-hrangeind:hind+hrangeind, kind-krangeind:kind+krangeind,lind-lrangeind:lind+lrangeind]

    dict_hkl = {'h':hnew, 'k':knew, 'l':lnew}

    data1allflat, data2allflat, intensity = [], [], []
    data1 = dict_hkl[data1label]
    data2 = dict_hkl[data2label]
    for data1ind in np.arange(0, len(data1), 1):
        for data2ind in np.arange(0, len(data2), 1):
            if data1label=='h' and data2label == 'k':
                intensity += [np.sum(Inew[data1ind,data2ind,:])]
            elif data1label =='h' and data2label == 'l':
                intensity += [np.sum(Inew[data1ind,:,data2ind])]
            elif data1label == 'k' and data2label == 'l':
                intensity += [np.sum(Inew[:,data1ind,data2ind])]
            data1allflat += [data1[data1ind]]
            data2allflat += [data2[data2ind]]

    zdata = intensity
    xdata, ydata = data1allflat, data2allflat

    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)

    X, Y = np.meshgrid(xi, yi)
    I = griddata((xdata,ydata), zdata, (X, Y))

    return X, Y, I

def maxpeakmagnifiervisualiser(filepath, hrange, krange, lrange):
    I, h, k, l= getIHKL(filepath)
    hmax, kmax, lmax = getmax(filepath)
    hindmax, kindmax, lindmax = np.where(I==np.max(I))
    hnew = h[hindmax[0]-hrange:hindmax[0]+hrange]
    knew = k[kindmax[0]-krange:kindmax[0]+krange]
    lnew = l[lindmax[0]-lrange:lindmax[0]+lrange]

    fig, ax = plt.subplots()
    X, Y, Z = dataforRSM_hk(filepath)
    img = ax.pcolor(X,Y,Z, cmap=plt.cm.seismic)
    ax.scatter(hmax, kmax, color='yellow')
    ax.plot([hnew[0], hnew[-1]], [knew[0], knew[0]], ls = '--',color='yellow')
    ax.plot([hnew[0], hnew[0]], [knew[0], knew[-1]], ls = '--',color='yellow')
    ax.plot([hnew[0], hnew[-1]], [knew[-1], knew[-1]], ls = '--',color='yellow')
    ax.plot([hnew[-1], hnew[-1]], [knew[0], knew[-1]], ls = '--',color='yellow')
    plt.show()
    Inew=I[hindmax[0]-hrange:hindmax[0]+hrange, kindmax[0]-krange:kindmax[0]+krange,lindmax[0]-lrange:lindmax[0]+lrange]

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ##hk plane
    hallflat, kallflat, lallflat, intensity = [], [], [], []
    for hind in np.arange(0, len(hnew),1):
        for kind in np.arange(0, len(knew),1):
            intensity += [np.sum(Inew[hind,kind,:])]
            hallflat += [hnew[hind]]
            kallflat += [knew[kind]]

    zdata = intensity
    xdata, ydata = hallflat, kallflat

    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)

    X, Y = np.meshgrid(xi, yi)
    Z = griddata((xdata,ydata), zdata, (X, Y))
    ax[0].pcolor(X,Y,Z, cmap=plt.cm.seismic)
    ax[0].set_xlabel('h')
    ax[0].set_ylabel('k')

    ##hl plane
    hallflat, kallflat, lallflat, intensity = [], [], [], []
    for hind in np.arange(0, len(hnew),1):
        for lind in np.arange(0, len(lnew),1):
            intensity += [np.sum(Inew[hind,:,lind])]
            hallflat += [hnew[hind]]
            lallflat += [lnew[lind]]

    zdata = intensity
    xdata, ydata = hallflat, lallflat

    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)

    X, Y = np.meshgrid(xi, yi)
    Z = griddata((xdata,ydata), zdata, (X, Y))
    ax[1].pcolor(X,Y,Z, cmap=plt.cm.seismic)
    ax[1].set_xlabel('h')
    ax[1].set_ylabel('l')
    ##kl plane
    hallflat, kallflat, lallflat, intensity = [], [], [], []
    for kind in np.arange(0, len(knew),1):
        for lind in np.arange(0, len(lnew),1):
            intensity += [np.sum(Inew[:,kind,lind])]
            kallflat += [knew[kind]]
            lallflat += [lnew[lind]]

    zdata = intensity
    xdata, ydata = kallflat, lallflat

    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)

    X, Y = np.meshgrid(xi, yi)
    Z = griddata((xdata,ydata), zdata, (X, Y))
    ax[2].pcolor(X,Y,Z, cmap=plt.cm.seismic)
    ax[2].set_xlabel('k')
    ax[2].set_ylabel('l')
    plt.show()

def volumetricfitter(direc, scan, savedir, centreofbox_hklindx, hrange, krange):
    filepath = f'{direc}\{scan}-allmag.nxs'
    h_cen_indx, k_cen_indx, l_cen_indx = centreofbox_hklindx
    I, h, k, l= VRSM.getIHKL(filepath)
    hnew = h[h_cen_indx[0]-hrange:h_cen_indx[0]+hrange]
    knew = k[k_cen_indx[0]-krange:k_cen_indx[0]+krange]

    fig, ax = plt.subplots(1, 2)
    X, Y, I = VRSM.dataforRSM_hk(filepath)
    img = ax[0].pcolor(X,Y,I, cmap=plt.cm.seismic, vmax=5e12)
    ax[0].scatter(h[h_cen_indx], k[k_cen_indx], color='yellow')
    ax[0].plot([hnew[0], hnew[-1]], [knew[0], knew[0]], ls = '--',color='yellow')
    ax[0].plot([hnew[0], hnew[0]], [knew[0], knew[-1]], ls = '--',color='yellow')
    ax[0].plot([hnew[0], hnew[-1]], [knew[-1], knew[-1]], ls = '--',color='yellow')
    ax[0].plot([hnew[-1], hnew[-1]], [knew[0], knew[-1]], ls = '--',color='yellow')

    Inew=I[hindmax[0]-hrange:hindmax[0]+hrange, kindmax[0]-krange:kindmax[0]+krange,:]

    hallflat, kallflat, intensity = [], [], []
    for hind in np.arange(0, len(hnew),1):
        for kind in np.arange(0, len(knew),1):
            intensity += [np.sum(Inew[hind,kind,:])]
            hallflat += [hnew[hind]]
            kallflat += [knew[kind]]

    zdata = intensity
    xdata, ydata = hallflat, kallflat

    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)

    X, Y = np.meshgrid(xi, yi)
    Z = griddata((xdata,ydata), zdata, (X, Y))
    ax[1].pcolor(X,Y,Z, cmap=plt.cm.seismic, vmax=5e12)
    plt.savefig(f'{savedir}\{scan}-{centreofbox_hklindx}-volumefit.png')
    plt.show()
    areas = []
    for kval in knew:
        xdata,ydata = VRSM.getslice(filepath, hmax, kval, lmax,'h')
        output, outputerr = pp.peakfit(xdata, ydata)
        areas += [output['Area']]
    area_givenazi += [np.sum(areas)]

    dict = {'scan':scan,
            'hkl_center': centreofbox_hklindx,
            'channel': pol}
    df = DataFrame(dict)
    df.insert('area', np.sum(area_givenazi))
    df.to_csv(f'{savedir}\{scan}_areafit.dat')

def extractlinescans(filepath, hrange, krange, lrange):
    I, h, k, l= getIHKL(filepath)
    hmax, kmax, lmax = getmax(filepath)
    hindmax, kindmax, lindmax = np.where(I==np.max(I))
    hnew = h[hindmax[0]-hrange:hindmax[0]+hrange]
    knew = k[kindmax[0]-krange:kindmax[0]+krange]
    lnew = l[lindmax[0]-lrange:lindmax[0]+lrange]

    Inew=I[hindmax[0]-hrange:hindmax[0]+hrange, kindmax[0]-krange:kindmax[0]+krange,lindmax[0]-lrange:lindmax[0]+lrange]

    fig, ax = plt.subplots()
    X, Y, Z = dataforRSM_hk(filepath)
    img = ax.pcolor(X,Y,Z, cmap=plt.cm.seismic)
    ax.scatter(hmax, kmax, color='yellow')
    ax.plot([hnew[0], hnew[-1]], [knew[0], knew[0]], ls = '--',color='yellow')
    ax.plot([hnew[0], hnew[0]], [knew[0], knew[-1]], ls = '--',color='yellow')
    ax.plot([hnew[0], hnew[-1]], [knew[-1], knew[-1]], ls = '--',color='yellow')
    ax.plot([hnew[-1], hnew[-1]], [knew[0], knew[-1]], ls = '--',color='yellow')
    plt.show()

    ##hk plane
    hallflat, kallflat, lallflat, intensity = [], [], [], []
    for kind in np.arange(0, len(knew),1):
        for hind in np.arange(0, len(hnew),1):
            intensity += [np.sum(Inew[hind,kind,:])]
            hallflat += [hnew[hind]]
            kallflat += [knew[kind]]

    zdata = intensity
    xdata, ydata = hallflat, kallflat

    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)

    X, Y = np.meshgrid(xi, yi)
    Z = griddata((xdata,ydata), zdata, (X, Y))


    return hallflat, kallflat, intensity, X, Y, Z, hnew, knew, lnew, Inew


def twodplotter(direc, startscan, stopscan, scanstepsize, labels, title,extensionfname,  data1label, data2label, data3axis):
    allX, allY, allI = [], [], []
    fontsize=25
    scans = [int(scan) for scan in np.arange(startscan, stopscan, scanstepsize)]
    ncol = int(len(scans)/4)
    fig, ax = plt.subplots(ncol, 4, figsize=(20,5.7*ncol))
    for scan, axf, labelval in zip(scans, ax.flatten(), labels):
        axf.set_ylabel("k", fontsize=fontsize)
        axf.set_xlabel("h", fontsize=fontsize)
        filepath=direc + f'\{scan}{extensionfname}.nxs'
        X, Y, I =dataforRSM_2d(filepath, data1label, data2label, data3axis)
        img = axf.pcolor(X,Y,I, cmap=plt.cm.seismic)
        axf.set_title(f'{labelval:.2f}', fontsize=fontsize)
#       a.set_aspect('equal')
        axf.spines['bottom'].set_color('white')
        axf.spines['top'].set_color('white')
        axf.spines['right'].set_color('white')
        axf.spines['left'].set_color('white')
        allX += [X]
        allY += [Y]
        allI += [I]
        mypy.colorbar(img)
    fig.suptitle(title, fontsize=fontsize+5)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    plt.show()
    return allX, allY, allI


def differenceoversum(direc, scan1, scan2):
    filepath1=direc + f'\{scan1}{extensionfname}.nxs'
    filepath2=direc + f'\{scan2}{extensionfname}.nxs'
    X1, Y1, I1 = VRSM.dataforRSM_hk(filepath1)
    X2, Y2, I2 = VRSM.dataforRSM_hk(filepath2)
    diff = np.subtract(I1, I2)
    sum = np.add(I1, I2)
    return diff, sum


def findmaximaandminima_seriesofscans(direc, startscan, stopscan, scanstep):
    max,min = 0,1e15
    for scanval in np.arange(startscan, stopscan, scanstep):
        filepath=direc + f'\{scanval}-allmag.nxs'
        X, Y, I = VRSM.dataforRSM_hk(filepath)
        tempmax = np.max(I)
        tempmin = np.min(I)
        if tempmax > max:
            max = tempmax
        else:
            pass
        if tempmin < min:
            min = tempmin
        else:
            pass
    return max, min

def projectingtoonedimension(I, h, k, l, projectingalong):
    onedprojection = []
    if projectingalong == 'h':
        xdata = h[:]
        for i in np.arange(0,len(h), 1):
            onedprojection += [np.sum(I[i,:,:])]
    elif projectingalong == 'k':
        x = k[:]
        for i in np.arange(0,len(k), 1):
            onedprojection += [np.sum(I[:,i,:])]
    elif projectingalong == 'l':
        xdata = l[:]
        for i in np.arange(0, len(l), 1):
            onedprojection += [np.sum(I[:,:,i])]
    return xdata,  onedprojection
