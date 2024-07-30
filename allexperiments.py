import numpy as np
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation
import sympy as sym
import re
import string
from MXRS import MagStruct as MS
from MXRS import Xtal as CR
from MXRS import Beam as B
from MXRS import Detector as D
from MXRS import Diffractometer as Diff
from MXRS import Scan as S
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from mypy import mypy
from scipy.optimize import least_squares

def setchannel(channel, b, det):
    if channel == 'sp':
        b.setSigma()
        det.setEta(90)
    elif channel == 'ps':
        b.setPi()
        det.setEta(0)

    elif channel == 'pp':
        b.setPi()
        det.setEta(90)

    elif channel == 'ss':
        b.setSigma()
        det.setEta(0)

    elif channel == 'crp':
        b.setRight()
        det.setEta(90)

    elif channel == 'crs':
        b.setRight()
        det.setEta(0)

    elif channel == 'clp':
        b.setLeft()
        det.setEta(90)

    elif channel== 'cls':
        b.setLeft()
        det.setEta(0)

    return b, det

def azimuthalscan(propvector, h, k, l, cab_magmodes, cac_magmodes, hbc_magmodes, channel):
        #Sigma Pi channel
    plt.rcParams['axes.linewidth']=0.9

    fig, ax = plt.subplots(figsize=(10,5))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    azivals = np.r_[-180:180:36j]

    colours = ['crimson', 'blue', 'black']
    lw=2.5
    scA, det, EAG_CS, b, EAG_MS = createnewexperiment(propvector)
    b, det = setchannel(channel, b, det)
    for bv in cab_magmodes:
        EAG_MS.addBasisVector(bv)
    sp = scA.calculateScattering(h, k, l, azivals, b.energy, coefs=[1,1])
    plt.plot(sp[0], sp[1], color=colours[0], lw=lw, ls='-', label='ab, equal')

    b, det = setchannel(channel, b, det)
    sp = scA.calculateScattering(h, k, l, azivals, b.energy, coefs=[1,1.5])
    plt.plot(sp[0], sp[1], color=colours[0], lw=lw, ls='--', label='ab, unequal')


    scA, det, EAG_CS, b, EAG_MS = createnewexperiment(propvector)
    b, det = setchannel(channel, b, det)
    for bv in cac_magmodes:
        EAG_MS.addBasisVector(bv)
    sp = scA.calculateScattering(h, k, l, azivals, b.energy, coefs=[1,1])
    plt.plot(sp[0], sp[1], color=colours[1], lw=lw, ls='-', label='ac, equal')

    b, det = setchannel(channel, b, det)
    sp = scA.calculateScattering(h, k, l, azivals, b.energy, coefs=[1,1.5])
    plt.plot(sp[0], sp[1], color=colours[1], lw=lw, ls='--', label='ac, unequal')


    scA, det, EAG_CS, b, EAG_MS = createnewexperiment(propvector)
    b, det = setchannel(channel, b, det)
    for bv in hbc_magmodes:
        EAG_MS.addBasisVector(bv)
    sp = scA.calculateScattering(h, k, l, azivals, b.energy, coefs=[1,1])
    plt.plot(sp[0], sp[1], color=colours[2], lw=lw, ls='-', label='bc, equal')


    scA, det, EAG_CS, b, EAG_MS = createnewexperiment(propvector)
    b, det = setchannel(channel, b, det)
    for bv in hbc_magmodes:
        EAG_MS.addBasisVector(bv)
    sp = scA.calculateScattering(h,k,l, azivals, b.energy, coefs=[1.5, 1])
    plt.plot(sp[0], sp[1], color=colours[2], lw=lw, ls='--', label='bc, unequal')

    plt.legend(ncol=3, fontsize=14, loc=1, frameon=False)
    plt.xticks(np.arange(-180, 240, 60))
    plt.xlabel('$\psi$ ($\degree$)')
    plt.ylabel('Intensity (arb. units)')
    plt.ylim(0,np.max(sp[1]))
    plt.title('({:s}, {:s}, {:s}), {:s}'.format(str(h), str(k), str(l), channel))
    plt.show()


def createlistref(hmax, kmax, l, extraref):
    hkl_list = []
    for h in np.arange(0, hmax+1, 1):
        for k in np.arange(0, kmax+1, 1):
            hkl_list += [[h, k, l]]
    if extraref == None:
        a ='do nothing'
    else:
        hkl_list += extraref
    return hkl_list


def calcscattinchannelsallazi(hkl, scA, det, b, azimuthvals, coeffs):
    h, k, l = hkl

    b, det = setchannel('ss', b, det)
    ss = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffs)[1]

    b, det = setchannel('sp', b, det)
    sp = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffs)[1]

    b, det = setchannel('ps', b, det)
    ps = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffs)[1]

    b, det = setchannel('pp', b, det)
    pp = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffs)[1]

    b, det = setchannel('crp', b, det)
    crp = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffs)[1]

    b, det = setchannel('crs', b, det)
    crs = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffs)[1]

    b, det = setchannel('clp', b, det)
    clp = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffs)[1]

    b, det = setchannel('cls', b, det)
    cls = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffs)[1]

    allchannels = [ss, sp, ps, pp, crp, crs, clp, cls]

    dict_channels = {}
    for channel, clabel in zip(allchannels, ['ss', 'sp', 'ps', 'pp', 'crp', 'crs', 'clp', 'cls']):
        dict_channels.update({clabel:channel})

    return dict_channels


def calcscattinchannelsallazi_withinvdomains(hkl, scA, det, b, azimuthvals, coeffdomain1, coeffdomain2, domainfrac1, domainfrac2):
    h, k, l = hkl

    b, det = setchannel('ss', b, det)
    ss = domainfrac1*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain1)[1] + domainfrac2*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain2)[1]

    b, det = setchannel('sp', b, det)
    sp = domainfrac1*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain1)[1] + domainfrac2*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain2)[1]

    b, det = setchannel('ps', b, det)
    ps = domainfrac1*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain1)[1] + domainfrac2*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain2)[1]

    b, det = setchannel('pp', b, det)
    pp = domainfrac1*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain1)[1] + domainfrac2*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain2)[1]

    b, det = setchannel('crp', b, det)
    crp = domainfrac1*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain1)[1] + domainfrac2*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain2)[1]

    b, det = setchannel('crs', b, det)
    crs = domainfrac1*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain1)[1] + domainfrac2*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain2)[1]

    b, det = setchannel('clp', b, det)
    clp = domainfrac1*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain1)[1] + domainfrac2*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain2)[1]

    b, det = setchannel('cls', b, det)
    cls = domainfrac1*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain1)[1] + domainfrac2*scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=coeffdomain2)[1]

    allchannels = [ss, sp, ps, pp, crp, crs, clp, cls]

    dict_channels = {}
    for channel, clabel in zip(allchannels, ['ss', 'sp', 'ps', 'pp', 'crp', 'crs', 'clp', 'cls']):
        dict_channels.update({clabel:channel})

    return dict_channels

def calcscattinchannelsgivenazi(hkl_list, propvector, scA, det, EAG_CS, b, EAG_MS, azimuthvals, psival):
    ssall, spall, psall, ppall, crpall, crsall, clpall, clsall = [], [], [], [], [], [], [], []
    hprop, kprop, lprop = propvector
    psiindex = np.where(azimuthvals == psival)[0][0]

    for hkl in hkl_list:
        h, k, l = hkl

        b, det = setchannel('ss', b, det)
        ss = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=[1,1])[1][psiindex]
        ssall += [ss]

        b, det = setchannel('sp', b, det)
        sp = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=[1,1])[1][psiindex]
        spall += [sp]

        b, det = setchannel('ps', b, det)
        det.setEta(0)
        ps = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=[1,1])[1][psiindex]
        psall += [ps]

        b, det = setchannel('pp', b, det)
        pp = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=[1,1])[1][psiindex]
        ppall += [pp]

        b, det = setchannel('crp', b, det)
        crp = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=[1,1])[1][psiindex]
        crpall += [crp]

        b, det = setchannel('crs', b, det)
        crs = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=[1,1])[1][psiindex]
        crsall += [crs]

        b, det = setchannel('clp', b, det)
        clp = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=[1,1])[1][psiindex]
        clpall += [clp]

        b, det = setchannel('cls', b, det)
        cls = scA.calculateScattering(h, k, l, azimuthvals, b.energy, coefs=[1,1])[1][psiindex]
        clsall += [cls]

    return ssall, spall, psall, ppall, crpall, crsall, clpall, clsall

def crosschannelratioforrefl(magstrucdata, selectref, hkl_list):
    c = 0
    refindex = 0
    for ref in hkl_list:
        if ref==selectref:
            refindex = c
            break
        c += 1
    cyclab_max = magstrucdata[2]
    ssall, spall, psall, ppall, crpall, crsall, clpall, clsall = magstrucdata[1]
    inc_cr = crpall[refindex] - crsall[refindex]/(crpall[refindex] + crsall[refindex])
    inc_cl = clpall[refindex] - clsall[refindex]/(clpall[refindex] + clsall[refindex])
    inc_pi = ppall[refindex] - psall[refindex]/(ppall[refindex] + psall[refindex])
    return inc_cr, inc_cl, inc_pi

def crosschannelratioformagstruc(magstrucdata, hkl_list):
    inc_cr_all = []
    inc_cl_all = []
    inc_pi_all = []
    for reflection in hkl_list:
        inc_cr, inc_cl, inc_pi = crosschannelratioforrefl(magstrucdata, reflection, hkl_list)
        inc_cr_all += [inc_cr]
        inc_cl_all += [inc_cl]
        inc_pi_all += [inc_pi]
    return inc_cr_all, inc_cl_all, inc_pi_all


def plotcrosschannelratio(title, hkl_list, allthreemagstrucdata):

    fig, ax = plt.subplots(nrows=3, figsize=(40,40))
    fontsize= 40
    plt.rcParams['xtick.labelsize']=fontsize
    plt.rcParams['ytick.labelsize']=fontsize

    i = 0
    titles = ['Cycloid, ab plane', 'Cycloid, ac plane', 'Helix, bc plane']
    for magstruc in allthreemagstrucdata:
        inc_cr_all, inc_cl_all, inc_pi_all  = crosschannelratioformagstruc(magstruc, hkl_list)

        df = pd.DataFrame({
        'Satellite Reflections' : hkl_list,
        "Incident CR": inc_cr_all,
        'Incident CL': inc_cl_all,
        'Incident Pi': inc_pi_all,
         })

        df.plot(ax=ax[i],x="Satellite Reflections", y=["Incident CR", "Incident CL", "Incident Pi"], kind="bar", color=['blue', 'lightgreen', 'red'], stacked=True, fontsize=15, width=0.7)
        ax[i].set_title(titles[i], fontsize=fontsize)
        i += 1

    # # plt.title(r'$\sigma \pi$', fontdict={'fontsize':fontsize})
    for a in ax:
        a.legend(frameon=False,ncol=3, loc = 'upper right', fontsize=fontsize-10)
        a.set_ylabel('Intensity', fontdict={'fontsize':fontsize})
        a.set_xlabel('Satellite Reflections', fontdict={'fontsize':fontsize})
        a.set_xticklabels(hkl_list, fontsize=fontsize)
        a.set_yticks(np.arange(-3, 5, 1))
        a.set_yticklabels(np.arange(-3, 5, 1), fontsize=fontsize)
    plt.subplots_adjust(wspace=0, hspace=0.7)
    plt.savefig(title, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


###domain should be zero if looking at a domain and should be 1 if looking at b domain
def ymodelaziscan(xdata, fittedparameters, channellabel, propvector, hkl, modelabels, domain):
    mode1label, mode2label = modelabels
    Rec, Imc = 1,1
    c1, c2 = fittedparameters
    dict_magmode1 = returndictmagmodes(c1, Rec, Imc, propvector)
    mode1 = dict_magmode1[domain][mode1label]

    dict_magmode2 = returndictmagmodes(c2, Rec, Imc, propvector)
    mode2 = dict_magmode2[domain][mode2label]

    scA, det, EAG_CS, b, EAG_MS = createnewexperiment(propvector)

    for bv in [mode1, mode2]:
        EAG_MS.addBasisVector(bv)

    dict_channels = calcscattinchannelsallazi(hkl, propvector, scA, det, EAG_CS, b, EAG_MS, xdata)
    ymodel = dict_channels[channellabel][0]
    return ymodel

def weighted_residuals(fittedparameters, xdata, ydata, ymodelfunction, otherparameters):
    ymodel = ymodelfunction(xdata, fittedparameters, *otherparameters)
    errors = np.sqrt(np.abs(ymodel))
    weights = 1/errors
    return weights*(ydata-ymodel)

# def ymodelmultipleaziscan(allxdata, fittedparameters, allchannellabel, allpropvector, allhkl, allmodelabels, alldomain, allRec, allImc):

def ymodel_multipleaziscans(allxdata, fittedparameters, allchannellabel, allpropvector, allhkl, allmodelabels, alldomains, allRec, allImc):

    allymodels = []
    for xdata, channellabel, propvector, hkl, modelabels, domain, Rec, Imc in zip(allxdata, allchannellabel, allpropvector, allhkl, allmodelabels, alldomains, allRec, allImc):
        ymodel = ymodelaziscan(xdata, fittedparameters, channellabel, propvector, hkl, modelabels, domain, Rec, Imc)
        allymodels+=[ymodel]

    allymodels = np.asarray(allymodels)
    allymodels = allymodels.flatten()
    return allymodels


def chi_sq(fittedparameters,xdata, ydata, ymodelfunction, otherparameters):
    ymodel = ymodelfunction(xdata, fittedparameters, *otherparameters)
    errors = np.sqrt(ymodel)
    residuals = ydata - ymodel
    return np.sum((residuals/errors)**2)


def r_factor(fittedparameters, xdata, ydata, ymodelfunction, otherparameters):
    ymodel = ymodelfunction(xdata, fittedparameters,*otherparameters)
    residuals = np.sum(np.abs(ydata-ymodel))
    sumy = np.sum(ydata)
    return 100*(residuals/sumy)


def fittingazimuthalscan(initialparams, regressionfunction, ymodel, xdata, ydata, method, lbub, jac, otherparameters):
    lb, ub = lbub
    output = least_squares(regressionfunction, initialparams, bounds=(lb, ub), method=method, args=(xdata, ydata, ymodel, otherparameters), jac=jac)
    fittedparameters  = output.x
    channellabel, propvector, hkl, modelabels, domain, Rec, Imc = otherparameters
    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, marker ='o', lw=0, color='black',  label='data')
    azimuthalvals = np.r_[-180:180:36j]
    ax.plot(azimuthalvals, ymodel(azimuthalvals, fittedparameters, *otherparameters), ls ='--',color='crimson',  label='fit')
    ax.legend(frameon=False)
    ax.set_xlabel('Psi ($\degree$)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.set_xticks(np.arange(-180, 240, 60))
    plt.show()
    print('Fitted parameters | mode coeff {:s} = {:5f}, mode coeff {:s} = {:5f}'.format(modelabels[0], fittedparameters[0], modelabels[1], fittedparameters[1]))
    print('Chi squared statistic: {:5f}'.format(chi_sq(fittedparameters, xdata, ydata, ymodel, otherparameters)))
    print('R-factor statistic: {:5f}'.format(r_factor(fittedparameters, xdata, ydata, ymodel, otherparameters)))


def fittingmultipleaziscans_giventemp(initialparams, regressionfunction, allxdata, allydata, method, lbub, jac, otherparameters):
    azimuthalvals = np.r_[-180:180:36j]
    allydataflatt = allydata.flatten()
    lb, ub = lbub
    output = least_squares(weighted_residuals, initialparams, bounds=(lb, ub), method=method, args=(allxdata, allydataflatt, ymodel_multipleaziscans, otherparameters), jac=jac)
    fittedparameters  = output.x
    allchannellabel, allpropvector, allhkl, allmodelabels, alldomains, allRec, allImc = otherparameters
    fig, ax = plt.subplots(1, len(allxdata), figsize=(7.5*len(allxdata), 5))
    for xdata, ydata, channellabel, propvector, hkl, modelabels, domain, Rec, Imc, a in zip(allxdata, allydata, allchannellabel, allpropvector, allhkl, allmodelabels, alldomains, allRec, allImc, ax):
        a.plot(xdata, ydata, marker ='o', lw=0, color='black',  label='data')
        a.plot(azimuthalvals, ymodelaziscan(azimuthalvals, fittedparameters, channellabel, propvector, hkl, modelabels, domain, Rec, Imc), ls ='--',color='crimson',  label='fit')
        a.legend(frameon=False)
        a.set_xlabel('Psi ($\degree$)')
        a.set_ylabel('Intensity (arb. units)')
        a.set_xticks(np.arange(-180, 240, 60))
    print('Fitted parameters | mode coeff {:s} = {:5f}, mode coeff {:s} = {:5f}'.format(modelabels[0], fittedparameters[0], modelabels[1], fittedparameters[1]))
    print('Chi squared statistic: {:5f}'.format(chi_sq(fittedparameters, allxdata, allydataflatt, ymodel_multipleaziscans, otherparameters)))
    print('R-factor statistic: {:5f}'.format(r_factor(fittedparameters, allxdata, allydataflatt, ymodel_multipleaziscans, otherparameters)))

    plt.show()


#domain 0, is mag prop along a and domain 1 is mag prop along b.
def ymodel_ratio(xdata, fittedparameters, chanlabels, propvector, hkl, modelabels, domain, azi):
    aziindex = np.where(np.abs(xdata - azi) == np.min(np.abs(xdata - azi)))[0][0]
    mode1label, mode2label = modelabels
    c2, scale = fittedparameters
    Rec, Imc = 1, 1
    dict_magmode1 = returndictmagmodes(1, Rec, Imc, propvector)
    mode1 = dict_magmode1[domain][mode1label]

    dict_magmode2 = returndictmagmodes(c2, Rec, Imc, propvector)
    mode2 = dict_magmode2[domain][mode2label]

    scA, det, EAG_CS, b, EAG_MS = createnewexperiment(propvector)

    calcscattinchannelsgivenazi(hkl_list, propvector, scA, det, EAG_CS, b, EAG_MS, azimuthvals, psival)

    for bv in [mode1, mode2]:
        EAG_MS.addBasisVector(bv)

    dict_channels = calcscattinchannelsallazi(hkl, propvector, scA, det, EAG_CS, b, EAG_MS, xdata)

    chanlab1, chanlab2 = chanlabels
    channel1 = dict_channels[chanlab1][aziindex]
    channel2 = dict_channels[chanlab2][aziindex]
    ratio = scale * np.divide(channel1 - channel2, channel1 + channel2)
    return ratio


def flattendata(scanno, direc):
    filepath = f"{direc}\{scanno}-mag.nxs"
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
    return intensity, hallflat, kallflat
