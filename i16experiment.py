import numpy as np
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation
import sympy as sym
import re
import string
import curvefitter as cf
import stats
import fittingfunctions as ff
import pandas as pd
import pp
import lmfit
import lmfit.models as lmm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mypy import mypy
import h5py

##if you make this into a class then you can create the polarisation vectors as part of
##the class variables.
oneoverroottwo = 1./np.sqrt(2)

def getdatafromnxs(filepath, xdatakey, ydatakey):
    h5 = h5py.File(filepath, 'r')
    xdata = h5['entry1']['measurement'][xdatakey][:]
    ydata = h5['entry1']['measurement'][ydatakey][:]
    ic1mon = h5['entry1']['measurement']['ic1monitor'][:]
    count_time = np.average(h5['entry1']['measurement']['count_time'][:])
    tran = h5['entry1']['instrument']['attenuator']['attenuator_transmission'][()]
    return xdata, np.divide(ydata, np.divide(ic1mon,1800)*tran*count_time)

def createtempdepnxs(filedir, scans, xdatakey, ydatakey, peakshape):
    areas, err_areas, temps = [], [], []
    for scanval in scans:
        filepath = f'{filedir}/{scanval}.nxs'
        h5 = h5py.File(filepath, 'r')
        xdata, ydata = getdatafromnxs(filepath, xdatakey, ydatakey)
        fit, error = pp.peakfit(xdata, ydata, type=peakshape)
        areas+= [fit['Area']]
        err_areas += [error['Area']]
        try:
            tempval = np.average(h5['entry1']['instrument']['Ta']['Ta'][()])
        except:
            tempval = h5['entry1']['before_scan']['Ta']['Ta'][()]
        temps+= [tempval]
    return areas, err_areas, temps


def getdatafromdat(filepath):
    with open(filepath) as infile:
        lines = infile.readlines()
    endofmetadata = [lines.index(x) for x in lines if re.search('&END', x)][0]
    metadata = lines[:endofmetadata]
    headers = lines[endofmetadata+1:][0].split('\t')
    dataunpacked =np.genfromtxt(lines[endofmetadata+2:], unpack=True)
    dict_data={}
    for datatype, data in zip(headers, dataunpacked):
        dict_data.update({datatype:data})
    return dict_data, metadata

def getdatafrommetadata(filepath, param):
    dict_data, metadata = getdatafromdat(filepath)
    metadatacut = metadata[:159] + metadata[160:]
    orignindex_uncut = [metadata.index(x) for x in metadata if re.search(param, x)]
    indexes = [metadatacut.index(x) for x in metadatacut if re.search(param, x)]
    paramvals = []
    for index in indexes:
        paramval = float(metadatacut[index][:-2].split('=')[1])
        paramvals += [paramval]
    return paramvals, indexes, orignindex_uncut

def getlineindexofmeta(filepath, param):
    dict_data, metadata = getdatafromdat(filepath)
    orignindex_uncut = [metadata.index(x) for x in metadata if re.search(param, x)]
    paramvals = []
    for index in orignindex_uncut:
        paramval = float(metadata[index][:-2].split('=')[1])
        paramvals += [paramval]
    return paramvals, orignindex_uncut


def getalphaandbeta(filepath):
    dict_data, metadata = getdatafromdat(filepath)
    beta = twotheta/2 - eta


###copy and paste the simallpsi... command from
def getalphabeta_corrector(filepath, psi):
    df = pd.read_csv(filepath, header=8, sep="\s+|;|:")
    df = df.drop(columns=df.columns[0:5])
    azi = df['AZI']
    azi_rad = np.radians(azi)
    alpha = np.radians(df['alpha'])
    beta = np.radians(df['beta'])
    mu=1
    corrector = 1/(mu*(np.divide(1,1 + np.divide(np.sin(alpha),np.sin(beta)))))
    corrector_limited = []

    for psival in psi:
        index = np.where(azi==psival)[0][0]
        # print(psival)
        corrector_limited += [corrector[index]]

    return alpha, beta, corrector_limited


def findmaximaandminima_seriesofscans(direc, startscan, stopscan, scanstep, key):
    max,min = 0,1e15
    for scanval in np.arange(startscan, stopscan, scanstep):
        filepath=direc + f'\{scan}-allmag.nxs'
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

def normalisedataviewer(filepath, ydatakey, normkey, tranline):
    data, metadata = getdatafromdat(filepath)
    print('Using this line in metadata for transmission')
    if tranline == False:
        print(metadata[119])
        tran = float(metadata[119].split('=')[1][:-1])
    else:
        print(metadata[tranline])
        tran =  float(metadata[tranline].split('=')[1][:-1])
    intensity = data[ydatakey]
    ic1monitor = data['ic1monitor']
    avgic1monitor = np.average(ic1monitor)

    try:
        t = np.average(data['t'])
    except:
        t = 1
        print(filepath)
        print('Note, not able to find t in metadata- adjust manually')
    ydata = np.divide(intensity, ic1monitor*tran*t)*800
    return ydata

def normalise(filepath, ydatakey, normkey, tranline):
    data, metadata = getdatafromdat(filepath)
    print('Using this line in metadata for transmission')
    if tranline == False:
        paramvals, orignindex_uncut = getlineindexofmeta(filepath, 'Transmission')
        print(metadata[orignindex_uncut[0]])
        tran = float(metadata[orignindex_uncut[0]].split('=')[1][:-1])
    else:
        print(metadata[tranline])
        tran =  float(metadata[tranline].split('=')[1][:-1])
    intensity = data[ydatakey]
    ic1monitor = data['ic1monitor']
    avgic1monitor = np.average(ic1monitor)

    try:
        t = np.average(data['count_time'])
    except:
        t = 1
        print(filepath)
        print('Note, not able to find t in metadata- adjust manually')
    ydata = np.divide(intensity, (ic1monitor/1800)*tran*t)
    return ydata


def getxydatafromdat(filepath, xdatakey, ydatakey, tranline):
    with open(filepath) as infile:
        lines = infile.readlines()
    endofmetadata = [lines.index(x) for x in lines if re.search('&END', x)][0]
    metadata = lines[:endofmetadata]
    headers = lines[endofmetadata+1:][0].split('\t')
    dataunpacked =np.genfromtxt(lines[endofmetadata+2:], unpack=True)
    dict_data={}
    for datatype, data in zip(headers, dataunpacked):
        dict_data.update({datatype:data})

    xdata = dict_data[xdatakey]
    ydata = normalise(filepath, ydatakey, 'ic1monitor', tranline)
    return xdata, ydata

def getlineindexofmeta(filepath, param):
    dict_data, metadata = getdatafromdat(filepath)
    orignindex_uncut = [metadata.index(x) for x in metadata if re.search(param, x)]
    paramvals = []
    for index in orignindex_uncut:
        print(index)
        paramval = float(metadata[index][:-2].split('=')[1])
        paramvals += [paramval]
    return paramvals, orignindex_uncut


def createtempdep(filedir, scans, xdatakey, ydatakey, peakshape, tranline, metatemp):
    areas, err_areas, temps = [], [], []
    for scanval in scans:
        filepath = f'{filedir}/{scanval}.dat'
        data, metadata = getdatafromdat(filepath)
        x = data[xdatakey]
        y = normalise(filepath, ydatakey, 'ic1monitor', tranline)
        fit, error = pp.peakfit(x, y, type=peakshape)
        areas+= [fit['Area']]
        err_areas += [error['Area']]
        tempval = getdatafrommetadata(filepath, metatemp)
        temps+= [tempval[-1]]
    return areas, err_areas, temps

def runlmfitandreturnarea(xdata, ydata, output=None):
    peak1 = lmm.PseudoVoigtModel(prefix='g1_')
    background = lmm.ConstantModel()
    model = peak1 + background
    params = model.make_params()

    if output == None:
        print('No starting paramters given')
    else:
        amp, center, sig, f, bgd = output['Peak Height'], output['Peak Centre'], output['FWHM'], output['Lorz frac'], output['Background']
        params.add_many(('g1_amplitude', amp), ('g1_center', center), ('g1_sigma', sig), ('g1_fraction', f), ('c', bgd))

    result = model.fit(ydata, params, x=xdata)
    result.plot_fit()
    plt.show()
    integratedint = result.params['g1_amplitude'].value
    integratedint_err = result.params['g1_amplitude'].stderr
    return integratedint, integratedint_err

def interpolatefor2dmap(xdata, ydata, zdata, intmethod):
    xi = np.linspace(np.min(xdata), np.max(xdata), 200)
    yi = np.linspace(np.min(ydata), np.max(ydata), 200)
    X, Y = np.meshgrid(xi, yi)
    I = griddata((xdata,ydata), zdata, (X, Y), method=intmethod)
    return X, Y, I


def fitscanreturninfo(filepath, fitshape, ydatakey, tranline):
    fig, ax = plt.subplots()
    data, metadata = getdatafromdat(filepath)
    xdata = data['eta']
    ydata = normalisedataviewer(filepath,ydatakey, 'ic1monitor', tranline)
    ax.plot(xdata, ydata)
    output, _ = pp.peakfit(xdata, ydata, type=fitshape)
    ax.plot(output['x'], output['y'])
    plt.show()
    intensity = output['Area']
    spara = getdatafrommetadata(filepath, 'spara=')[0]
    sperp = getdatafrommetadata(filepath, 'sperp=')[0]
    sx = getdatafrommetadata(filepath, 'sx=')[0]
    sy = getdatafrommetadata(filepath, 'sy=')[0]
    fwhm = output['FWHM']
    centre = output['Peak Centre']
    return intensity, spara, sperp, sx, sy, fwhm, centre

def sxsymap(scans, direc, fitshape, ydatakey, tranline):
    I1, fwhm1, centre1, allspara1, allsperp1, allsx, allsy = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for scanvalc1 in scans:
        filepath = f'{direc}\{scanvalc1}.dat'
        intensity1, spara, sperp, sx, sy, fwhm, centre = fitscanreturninfo(filepath, fitshape, ydatakey, tranline)
        I1, fwhm1, centre1, allspara1, allsperp1 = np.append(I1, intensity1), np.append(fwhm1, fwhm), np.append(centre1, centre), np.append(allspara1, spara), np.append(allsperp1, sperp)
        allsx, allsy = np.append(allsx, sx), np.append(allsy, sy)
    data1 = [I1, fwhm1, centre1, allspara1, allsperp1, allsx, allsy]
    return data1


def C1C2sxsymap(scansc1, scansc2, direc,ydatakey, fitshape, tranline):
    I1, fwhm1, centre1, allspara1, allsperp1, allsx1, allsy1 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    I2, fwhm2, centre2, allspara2, allsperp2, allsx2, allsy2 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    diff, diffnorm = np.array([]),  np.array([])
    for scanvalc1, scanvalc2 in zip(scansc1, scansc2):
        filepath = f'{direc}\{scanvalc1}.dat'
        intensity1, spara, sperp, sx, sy, fwhm, centre, = fitscanreturninfo(filepath, fitshape, ydatakey, tranline)
        I1, fwhm1, centre1, allspara1, allsperp1 = np.append(I1, intensity1), np.append(fwhm1, fwhm), np.append(centre1, centre), np.append(allspara1, spara), np.append(allsperp1, sperp)
        allsx1, allsy1 = np.append(allsx1, sx), np.append(allsy1, sy)

        filepath = f'{direc}\{scanvalc2}.dat'
        intensity2, spara, sperp, sx, sy, fwhm, centre = fitscanreturninfo(filepath, fitshape, ydatakey, tranline)
        I2, fwhm2, centre2, allspara2, allsperp2 = np.append(I2, intensity2), np.append(fwhm2, fwhm), np.append(centre2, centre), np.append(allspara2, spara), np.append(allsperp2, sperp)
        allsx2, allsy2 = np.append(allsx2, sx), np.append(allsy2, sy)
        diff = np.append(diff,intensity1-intensity2)
        diffnorm = np.append(diffnorm, np.divide(intensity1-intensity2, intensity1+intensity2))

    data1 = [I1, fwhm1, centre1, allspara1, allsperp1, allsx1, allsy1]
    data2 = [I2, fwhm2, centre2, allspara2, allsperp2, allsx2, allsy2]
    return data1, data2, diff, diffnorm

def processazimuthalscan_dataviewer(datadirectory, scan, peakfittype, ydatakey, xdatakey):
    filepath= f'{datadirectory}\{scan}.dat'
    data, metadata = getdatafromdat(filepath)
    for key in data.keys():
        if key[:len(ydatakey)] == ydatakey:
            ydatakeyfull = key
    xdata = data[xdatakey]
    ydata =  normalisebyic1monitor(filepath, ydatakeyfull)
    output, outputerr = pp.peakfit(xdata, ydata, type=peakfittype)
    integratedint = output['Area']
    integratedint_err = outputerr['Area']
    xfit = output['x']
    yfit = output['y']
    psival = getdatafrommetadata(filepath, 'psi=')
    return psival, integratedint, integratedint_err, xfit, yfit, xdata, ydata


##can recreate this with the symbolic functions that you found. replace theta with sym theta
def polarisationvectors(theta):
    oneoverroottwo = 1./np.sqrt(2)
    polout = np.array([oneoverroottwo*np.exp(1j*theta), 1, -oneoverroottwo*np.exp(-1j*theta)])
    poloutc = np.array([1./np.sqrt(2)*np.exp(-1j*theta), 1, -oneoverroottwo*np.exp(1j*theta)])
    polin = np.array([oneoverroottwo*np.exp(-1j*theta), 1, -oneoverroottwo*np.exp(1j*theta)])
    qin = np.array([-oneoverroottwo*1j*np.exp(-1j*theta), 0, -oneoverroottwo*1j*np.exp(1j*theta)])
    qout = np.array([-oneoverroottwo*1j*np.exp(1j*theta), 0, -oneoverroottwo*1j*np.exp(-1j*theta)])#
    return polout, poloutc, polin, qin, qout

def dipdip(theta, Q, K, ivals, jvals):
    oneoverroottwo = 1./np.sqrt(2)
    polout = np.array([oneoverroottwo*np.exp(1j*theta), 1, -oneoverroottwo*np.exp(-1j*theta)])
    poloutc = np.array([1./np.sqrt(2)*np.exp(-1j*theta), 1, -oneoverroottwo*np.exp(1j*theta)])
    polin = np.array([oneoverroottwo*np.exp(-1j*theta), 1, -oneoverroottwo*np.exp(1j*theta)])
    qin = np.array([-oneoverroottwo*1j*np.exp(-1j*theta), 0, -oneoverroottwo*1j*np.exp(1j*theta)])
    qout = np.array([-oneoverroottwo*1j*np.exp(1j*theta), 0, -oneoverroottwo*1j*np.exp(-1j*theta)])#

    tot = 0
    for i in ivals:
        for j in jvals:
            j1, j2 = 1,1
            j3 = Q
            m1 = i
            m2 = j
            m3 = K
            cg = CG(j1, m1, j2, m2, j3, m3)
            tot += np.sum(polin[i+1]*poloutc[j+1]*cg.doit())
    return tot


def fittwopeaks(filepath):
    area = 0
    data, metadata = i16.getdatafromdat(filepath)
    for key in data.keys():
        if key[:8] == 'roi1_sum':
            ydatakey = key
    xdata, ydata = data['eta'], np.divide(data[ydatakey], 1)
    gradient = np.gradient(ydata)

    ###get initial values
    peaks, _ = scipy.signal.find_peaks(gradient)
    maxvals = np.sort(ydata[peaks])[-2:]
    peakindx = []
    for maxval in maxvals:
        indx = np.where(ydata[peaks] == maxval)[0][0]
        peakindx += [peaks[indx]]
    initialcentres = xdata[peakindx]

    ##setting up the model to fit the data
    peak1 = lmm.PseudoVoigtModel(prefix='g1_')
    peak2 = lmm.PseudoVoigtModel(prefix='g2_')
    background = lmm.ConstantModel()
    model = peak1 + peak2 + background
    params = model.make_params()
    params['g1_amplitude'].min=0
    params['g2_amplitude'].min=0
    params['g1_sigma'].max=0.12
    params['g2_sigma'].max=0.12
    params['g1_center'].value = initialcentres[0]
    params['g2_center'].value = initialcentres[1]
    params['g1_center'].min = initialcentres[0]-0.03
    params['g2_center'].min = initialcentres[1]-0.03
    params['g1_center'].max = initialcentres[0]+0.03
    params['g2_center'].max = initialcentres[1]+0.03

    result = model.fit(ydata,params, x=xdata)
    result.plot_fit()
    print(result.params['g1_amplitude'])
    print(result.params['g2_amplitude'])
    print(result.params['g1_sigma'])
    print(result.params['g2_sigma'])
    print(result.params['g1_center'])
    print(result.params['g2_center'])

    area = result.params['g1_amplitude'].value + result.params['g2_amplitude'].value

#     fig, ax= plt.subplots()
#     ax.plot(xdata, ydata)
#     ax.plot(xdata, result.eval())
#     plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(xdata, ydata)
    axes[0].plot(xdata, result.best_fit, '-', label='best fit')
    axes[0].legend()

    comps = result.eval_components(x=xdata)
    axes[1].plot(xdata, ydata)
    axes[1].plot(xdata, comps['g1_'], '--', label='Gaussian component 1')
    axes[1].plot(xdata, comps['g2_'], '--', label='Gaussian component 2')
    axes[1].legend()
    plt.show()
    return result, area


def quadquad(theta, Q, K, ivals, jvals):
    oneoverroottwo = 1./np.sqrt(2)
    polout = np.array([oneoverroottwo*np.exp(1j*theta), 1, -oneoverroottwo*np.exp(-1j*theta)])
    poloutc = np.array([1./np.sqrt(2)*np.exp(-1j*theta), 1, -oneoverroottwo*np.exp(1j*theta)])
    polin = np.array([oneoverroottwo*np.exp(-1j*theta), 1, -oneoverroottwo*np.exp(1j*theta)])
    qin = np.array([-oneoverroottwo*1j*np.exp(-1j*theta), 0, -oneoverroottwo*1j*np.exp(1j*theta)])
    qout = np.array([-oneoverroottwo*1j*np.exp(1j*theta), 0, -oneoverroottwo*1j*np.exp(-1j*theta)])#

    tot = 0
    for i, k in zip(ivals, kvals):
        for j, l in zip(jvals, lvals):
            j1, j2 = 1,1
            j3 = Q
            m1 = i
            m2 = j
            m3 = K
            cg = CG(j1, m1, j2, m2, j3, m3)
            Hi =polin[i+1]*qin[j+1]*cg.doit()
            Hf = poloutc[k+1]*qin[l+1]*cg.doit()
            tot += np.sum(Hi*Hf)
    return tot


def wignerRotMtx(rank, parity, alpha, beta, gamma):
    dim = (rank*2) + 1
    wigRotMtx = np.zeros((dim, dim), dtype=complex)
    for r, m in zip(np.arange(0, dim, 1), np.arange(-rank, rank+1, 1)):
        for c, mp in zip(np.arange(0, dim, 1), np.arange(-rank, rank+1, 1)):
            wigD = Rotation.D(rank, m, mp, alpha, beta, gamma).doit()
            wigRotMtx[r,c] = wigD

    for r in np.arange(0, len(wigRotMtx), 1):
        for c in np.arange(0, len(wigRotMtx), 1):
            wigD = wigRotMtx[r,c]
            if np.abs(np.imag(wigD))< 1e-15:
                newWigDim = 0
            else:
                newWigDim = np.imag(wigD)
            if np.abs(np.real(wigD)) < 1e-15:
                newWigDre = 0
            else:
                newWigDre = np.real(wigD)
            newWigD = newWigDre + newWigDim*1j
            wigRotMtx[r,c] = newWigD

    return wigRotMtx


def applysymmetry(symop, symname, tensor, nooftimestoappsymop):
    tensadded = tensor
    transtens=[tensor]
    for i in np.arange(0,nooftimestoappsymop,1):
        transtens += [np.dot(symop, transtens[-1])]
        tensadded += transtens[-1]
    print(symname + ', ', tensadded)
    return transtens, tensadded

def calculatephasefactor(xyz, hkl):
    phasefac = np.exp(-2*np.pi*1j*np.dot(xyz,hkl))
    if np.abs(np.real(phasefac)) < 1e-15:
        phasefacre = 0
    else:
        phasefacre = np.real(phasefac)
    if np.abs(np.imag(phasefac)) < 1e-15:
        phasefacim = 0
    else:
        phasefacim = np.imag(strucfac)
    return phasefacre + phasefacim*1j


C_m1_1 = oneoverroottwo*np.array([1, -1j, 0])
C_0_1 = np.array([0,0,1])
C_1_1 = oneoverroottwo*np.array([-1, -1j, 0])

C_m2_2 = (1/2.)*np.array([[1, -1j, 0], [-1j, -1, 0], [0,0,0]])
C_m1_2 = (1/2.)*np.array([[0, 0, 1], [0, 0, -1j], [1,-1j,0]])
C_0_2 = (1/np.sqrt(6))*np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]])
C_1_2 = (1/2.)*np.array([[0, 0, -1], [0, 0, -1j], [-1, -1j, 0]])
C_2_2 = (1/2.)*np.array([[1, 1j, 0], [1j, -1, 0], [0, 0, 0]])

C_m3_3 = (1/np.sqrt(8))*np.array([[1, -1j, 0], [-1j, -1, 0], [0, 0, 0],
                                  [-1j, -1, 0], [-1, 1j, 0], [0, 0, 0],
                                  [0, 0, 0], [0, 0, 0], [0, 0, 0]])

C_m2_3 = (1/np.sqrt(12))*np.array([[0, 0, 1], [0, 0, -1j], [1,-1j,0],
                                    [0, 0, -1j], [0, 0, -1], [-1j, -1, 0],
                                    [1, -1j, 0], [-1j, -1, 0], [0, 0, 0]])

C_m1_3 = (1/np.sqrt(120))*np.array([[-3, 1j, 0], [1j, -1, 0], [0, 0, 4],
                                    [1j, -1, 0], [-1, 3j, 0], [0, 0, -4j],
                                    [0, 0, 4], [0, 0, -4j], [4, -4j, 0]])

C_0_3 = (1/np.sqrt(10))*np.array([[0, 0, -1], [0, 0, 0], [-1, 0, 0],
                                  [0, 0, 0], [0, 0, -1], [0, -1, 0],
                                  [-1, 0, 0], [0, -1, 0], [0, 0, 2]])

C_1_3 = (1/np.sqrt(120))*np.array([[3, 1j, 0], [1j, 1, 0], [0, 0, -4],
                                   [1j, 1, 0], [1, 3j, 0], [0, 0, -4j],
                                   [0, 0, -4], [0, 0, -4j], [-4, -4j, 0]])

C_2_3 = (1/np.sqrt(12))*np.array([[0, 0, 1], [0, 0, 1j], [1,1j,0],
                                   [0, 0, 1j], [0, 0, -1], [1j, -1, 0],
                                   [1, 1j, 0], [1j, -1, 0], [0, 0, 0]])

C_m3_3 = (1/np.sqrt(8))*np.array([[-1, -1j, 0], [-1j, 1, 0], [0, 0, 0],
                                  [-1j, 1, 0], [1, 1j, 0], [0, 0, 0],
                                  [0, 0, 0], [0, 0, 0], [0, 0, 0]])

C_m4_4 = (1/4.)*np.array([[1, -1j, 0], [-1j, -1, 0], [0, 0, 0],
                          [-1j, -1, 0], [-1, 1j, 0], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0, 0],
                          [-1j, -1, 0], [-1, 1j, 0], [0, 0, 0],
                          [-1, 1j, 0], [1j, 1, 0], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0, 0],
                           [0, 0, 0], [0, 0, 0], [0, 0, 0]])


C_m3_4 = (1/np.sqrt(8))*np.array([[0,0,1], [0,0,-1j], [1, -1j, 0],
                                  [0,0,-1j], [0, 0, -1], [-1j, -1, 0],
                                  [1, -1j, 0], [-1j, -1, 0], [0, 0, 0],
                                  [0, 0, -1j], [0, 0, -1], [-1j, -1, 0],
                                  [0, 0, -1], [0, 0, 1j], [-1, 1j, 0],
                                  [-1j, -1, 0], [-1, 1j, 0], [0, 0, 0],
                                  [1, -1j, 0], [-1j, -1, 0], [0, 0, 0],
                                  [-1j, -1, 0], [-1, 1j, 0], [0, 0, 0],
                                  [0, 0, 0], [0, 0, 0], [0, 0, 0]])



C_m2_4 = (1/np.sqrt(12))*np.array([[0, 0, 1], [0, 0, -1j], [1,-1j,0], [0, 0, -1j], [0, 0, -1], [-1j, -1, 0], [1, -1j, 0], [-1j, -1, 0], [0, 0, 0]])
C_m1_4 = (1/np.sqrt(120))*np.array([[-3, 1j, 0], [1j, -1, 0], [0, 0, 4], [1j, -1, 0], [-1, 3j, 0], [0, 0, -4j], [0, 0, 4], [0, 0, -4j], [4, -4j, 0]])
C_0_4 = (1/np.sqrt(10))*np.array([[0, 0, -1], [0, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 2]])
C_1_4 = (1/np.sqrt(120))*np.array([[3, 1j, 0], [1j, 1, 0], [0, 0, -4], [1j, 1, 0], [1, 3j, 0], [0, 0, -4j], [0, 0, -4], [0, 0, -4j], [-4, -4j, 0]])
C_2_4 = (1/np.sqrt(12))*np.array([[0, 0, 1], [0, 0, 1j], [1,1j,0], [0, 0, 1j], [0, 0, -1], [1j, -1, 0], [1, 1j, 0], [1j, -1, 0], [0, 0, 0]])
C_m3_4 = (1/np.sqrt(8))*np.array([[-1, -1j, 0], [-1j, 1, 0], [0, 0, 0], [-1j, 1, 0], [1, 1j, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
