# Last Updated: 2020-Apr-30

# VMOMS Code for finding Moment Solutions to Grad-Shafranov Equation
# Original: L.L. Lao et al, Computer Physics Communication 27 (1982) 129-146

# TYPE
# Fourier Resolution: 2
# Symmetry: Up-Down, Toroidal
# Dimensions: 2
# Driving Factors as input: Pressure, Current

# ALL YOU NEED TO DO:
# Call vmoms() function with appropriate inputs
# Call flux_surface_plot() using output of vmoms() to visualize surface plots
# Call other_variables() using output of vmoms() to calculate and plot several other physical variables
# Call tex_report() using output of vmoms() and other_variables() to print a readable report (convert from .tex to .pdf externally)

# TROUBLESHOOTING:
#   If "Singular Matrix" error appears, make sure rhodt[0] isn't too close to 0
#   If there are issues with np.linalg.solve in fyp() of vmoms(), try np.linalg.lstsq (see commented line there)

# NOTE:
#   One can change number of points in theta array th in vmoms() if needed. Keep it odd to be compatible with Simpson's Rule for integration
#   One can change lmb1 and lmb2 in vmoms(), they are small values used in estimating initial values of y. their exact values don't have much impact on final results
#   In function fyp() in vmoms(), one can choose different interpolation method for pressure and current. Theoretically they are expected to have zero derivative at boundaries, which can be specified in CubicSpline() interpolation function using option bc_type, but it was ignored because our grids don't start at 0 and may not end at exact edge of the system, thus imposing 0 derivatives for pressure and current will be incorrect. Still, other options can be used instead of default 'not-a-knot'
#   In other_variables(), psi, cflx, phi, btor, jphi and jeqv are calculated in poloidal plane only, to get their total values on the flux surfaces in 3D multiply them by 2.0*np.pi

# Tested with Python 3.6.4, Numpy 1.13.3, Scipy 1.3.0, Matplotlib 2.1.1

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt
import os
import sys
from gbvpsolver import gbvpsolver

def plotter(x, y, ttl = '', xlbl = '', xunit = '', ylbl = '', yunit = '', fldr = '', frmt = 'png', dpi = 200):
    ''' (array, array, [str, str, str, str, str, str, str, int]) --> bool
        Plots x and y and saves in fldr folder
        Defines title using ttl, xlabel with xlbl and xunit, ylabel with ylbl and yunit
        File name is yunit + frmt
    '''
    plt.clf()
    plt.tight_layout()
    plt.ticklabel_format(style = 'sci', axis = 'y', scilimits = (-3, 3))
    plt.plot(x, y)
    if xunit == '':
        plt.xlabel(xlbl)
    else:
        plt.xlabel(xlbl + ' (' + xunit + ')')
    if yunit == '':
        plt.ylabel(ylbl)
    else:
        plt.ylabel(ylbl + ' (' + yunit + ')')
    plt.title(ttl)
    if ttl == '':
        fyl = ylbl.replace(' ', '_') + '.' + frmt
    else:
        fyl = ttl.replace(' ', '_') + '.' + frmt
    plt.savefig(fldr + '/' +  fyl, dpi = dpi, format = frmt, bbox_inches = 'tight')
    return True

def thavg(fdt, dth):
    ''' (array, float) --> float
        Calculates average of data fdt on uniform array of theta in 0 to np.pi with gap dth, using Simpson's Rule
        Please prefer to have odd number of points in fdt to be compatible with Simpson's Rule
    '''
    return scipy.integrate.simps(fdt, dx = dth) / np.pi

def geotrn1(trng, elip):
    ''' (float, float) --> float, float
        Calculates R2/R1 and E used in VMOMS from geometric ellipticity and triangularity
    '''
    ctc = (scipy.optimize.root(lambda x: 2 * x - 3 * x**3 - trng * (1 - 2 * x**2), 0.5 * trng).x)[0]
    ctc2 = ctc * ctc
    c2tc = 2.0 * ctc2 - 1
    return (trng - ctc) / (1.0 - c2tc), elip * np.sqrt(1.0 - ctc2) / (trng * ctc - c2tc)

def geotrn2(r1, r2, ee):
    ''' (float, float, float) --> float, float
        Calculates geometric ellipticity and triangularity from R1, R2 and E in VMOMS
    '''
    r2r1 = r2 / r1
    r2r12 = r2r1 * r2r1
    ctc = 4.0 * r2r1 / (1.0 + np.sqrt(1.0 + 32.0 * r2r12))
    ctc2 = ctc * ctc
    c2tc = 2.0 * ctc2 - 1
    stc = np.sqrt(1.0 - ctc2)
    s2tc = 2.0 * stc * ctc
    return r2r1 + ctc - r2r1 * c2tc, ee * (stc + r2r1 * s2tc)

def vmoms(rmaj, tring, ellip, rhodt, prespfx, curifx, curipfx, p0in, ivp_method = 'RK45', rt_method = 'lm', rt_tol = 1.0e-8):
    ''' (float, float, float, array, function, function, function, [str, str]) --> array, array

        Finds Moment Solutions to Grad-Shafranov Equation

        INPUT:
        rmaj = Major Radius (meter; float)
        tring = Geometric Triangularity desired at LCFS rhodt[-1] (unitless; float)
        ellip = Ellipticity desired at LCFS rhodt[-1] (unitless; float)
        rhodt = Array of flux surface label R1, on which pressure and current data are mentioned and on which final solution will be obtained (meter; 1d float array)
        prespfx = Pressure Derivative Function, calculatable on rhodt (pascal/meter; single-input function)
        curifx = Current Function, calculatable on rhodt (ampere; single-input function)
        curipfx = Current Derivative Function, calculatable on rhodt (ampere/meter; single-input function)
        p0in = parameters used to initialize p0 (unitless; array)
               (rmaj/rmin - tring) + p0in[0] roughly gives R0 at rhodt[0]
               p0in[1] * rhodt[0]**2 roughly gives R2 at rhodt[0]
               p0in[2] roughly gives E at rhodt[0]
        ivp_method = method to be used by ODE IVP solver, check odesolver() for details (str; optional)
        rt_method = method to be used by root finder, check rootfinder() for details (str; optional)
        rt_tol = tolerance used by root finder in shooting method (float; optional)

        OUTPUT:
        y = final values of R0, R0x, R2, R2x, E, Ex on ps (R in meter, E unitless; 2d float array)
    '''
    mu0 = 4.0e-7 * np.pi
    lmb1 = 1.0e-4
    lmb2 = 1.0e-4
    ndt = len(rhodt)
    neq = len(p0in) * 2
    rho0 = rhodt[-1]
    rho = rhodt/rho0
    rga = rmaj / rho0
    r2f, eef = geotrn1(tring, ellip)
    p0 = np.array([rga - r2f + p0in[0], p0in[1], p0in[2]])

    # function returning initial values of y
    def fy0(x0, p):
        return np.array([p[0] - lmb1 * x0**2,
                         -2.0 * lmb1 * x0,
                         p[1] * x0**2,
                         2.0 * p[1] * x0,
                         p[2] + lmb2 * x0**2,
                         2.0 * lmb2 * x0])

    # function whose root is found by rootfinder()
    def frt(x, y):
        return np.array([y[0, -1] - (rga - r2f),
                         y[2, -1] - r2f,
                         y[4, -1] - eef])

    # temporary variables to be used in fyp()
    th = np.linspace(0, np.pi, 129)
    dth = th[1] - th[0]
    cs1t = np.cos(th)
    cs2t = np.cos(2.0 * th)
    sn1t = np.sin(th)
    sn2t = np.sin(2.0 * th)
    pif0 = 4.0 * np.pi**2 * rho0**3 / mu0
    pif1 = rho0

    # function returning RHS of ODEs
    def fyp(x, y, p):
        rn = np.array([y[0], x, y[2]])
        rnp = np.array([y[1], 1.0, y[3]])
        ee = y[4]
        eep = y[5]
        prsp = prespfx(x * rho0)
        cri = curifx(x * rho0)
        crip = curipfx(x * rho0)

        r = rn[0] - rn[1] * cs1t + rn[2] * cs2t
        rt = rn[1] * sn1t - 2.0 * rn[2] * sn2t
        rtt = rn[1] * cs1t - 4.0 * rn[2] * cs2t
        rp = rnp[0] - rnp[1] * cs1t + rnp[2] * cs2t
        rtp = rnp[1] * sn1t - 2.0 * rnp[2] * sn2t
        zt = ee * (rn[1] * cs1t + 2.0 * rn[2] * cs2t)
        ztt = -ee * (rn[1] * sn1t + 4.0 * rn[2] * sn2t)
        zp = eep * (rn[1] * sn1t + rn[2] * sn2t) + ee * (rnp[1] * sn1t + rnp[2] * sn2t)
        ztp = eep * (rn[1] * cs1t + 2.0 * rn[2] * cs2t) + ee * (rnp[1] * cs1t + 2.0 * rnp[2] * cs2t)
        tau = rt * zp - rp * zt
        sg = r * tau
        g = sg ** 2
        gtt = rt ** 2 + zt ** 2
        gtp = rt * rp + zt * zp
        gs8 = gtt / g
        gs10 = gs8 * r * zt / sg
        gs12 = gs10 * cs2t
        gs21 = gs8 * r * rt * sn1t / sg
        gs22 = gs8 * r * rt * sn2t / sg

        gs5 = (zt * tau ** 2 + 2.0 * (rt * rp + zt * zp) * r * (rt * ztp - zt * rtp) + r * (rp ** 2 + zp ** 2) * (rtt * zt - ztt * rt)) / (g * sg)
        mr0 = r * zt
        mr2 = r * (ee * rt * sn2t - zt * cs2t)
        mee = r * rt * (rn[1] * sn1t + rn[2] * sn2t)

        asg = thavg(sg, dth)
        asgr2 = thavg(tau / r, dth)
        asggs8 = thavg(gtt / sg, dth)
        asggs5 = thavg(sg * gs5, dth)
        asggs10 = thavg(sg * gs10, dth)
        asggs12 = thavg(sg * gs12, dth)
        asggs21 = thavg(sg * gs21, dth)
        asggs22 = thavg(sg * gs22, dth)

        tmpv0 = gs10 - gs8 * asggs10 / asggs8
        dmr010 = thavg(mr0 * tmpv0, dth)
        dmr210 = thavg(mr2 * tmpv0, dth)
        dmee10 = thavg(mee * tmpv0, dth)

        tmpv0 = gs12 - gs8 * asggs12 / asggs8
        dmr012 = thavg(mr0 * tmpv0, dth)
        dmr212 = thavg(mr2 * tmpv0, dth)
        dmee12 = thavg(mee * tmpv0, dth)

        tmpv0 = gs21 - gs8 * asggs21 / asggs8
        dmr021 = thavg(mr0 * tmpv0, dth)
        dmr221 = thavg(mr2 * tmpv0, dth)
        dmee21 = thavg(mee * tmpv0, dth)

        tmpv0 = gs22 - gs8 * asggs22 / asggs8
        dmr022 = thavg(mr0 * tmpv0, dth)
        dmr222 = thavg(mr2 * tmpv0, dth)
        dmee22 = thavg(mee * tmpv0, dth)

        tmpv0 = (1 - asg / (asgr2 * r ** 2))
        tmpv1 = -pif0 * prsp * (asggs8 ** 2) / cri ** 2
        b0p = tmpv1 * thavg(mr0 * tmpv0, dth)
        b2p = tmpv1 * thavg(mr2 * tmpv0, dth)
        bep = tmpv1 * thavg(mee * tmpv0, dth)

        tmpv0 = (gs8 - asggs8 / (asgr2 * r ** 2))
        tmpv1 = -pif1 * crip / cri
        b0i = tmpv1 * thavg(mr0 * tmpv0, dth)
        b2i = tmpv1 * thavg(mr2 * tmpv0, dth)
        bei = tmpv1 * thavg(mee * tmpv0, dth)

        tmpv0 = -gs5 + asggs5 * gs8 / asggs8
        b05 = thavg(mr0 * tmpv0, dth)
        b25 = thavg(mr2 * tmpv0, dth)
        be5 = thavg(mee * tmpv0, dth)

        b0 = b0p + b0i + b05 + 2.0 * eep * (dmr021 * rnp[1] + dmr022 * rnp[2])
        b2 = b2p + b2i + b25 + 2.0 * eep * (dmr221 * rnp[1] + dmr222 * rnp[2])
        be = bep + bei + be5 + 2.0 * eep * (dmee21 * rnp[1] + dmee22 * rnp[2])
        b = np.array([b0, b2, be])

        a00 = dmr010
        a20 = dmr210
        ae0 = dmee10
        a02 = dmr012 - ee * dmr022
        a22 = dmr212 - ee * dmr222
        ae2 = dmee12 - ee * dmee22
        a0e = -(rn[1] * dmr021 + rn[2] * dmr022)
        a2e = -(rn[1] * dmr221 + rn[2] * dmr222)
        aee = -(rn[1] * dmee21 + rn[2] * dmee22)
        a = np.array([[a00, a02, a0e], [a20, a22, a2e], [ae0, ae2, aee]])
        #yxx = np.linalg.lstsq(a, b, rcond = None)[0]
        yxx = np.linalg.solve(a, b)

        return np.array([rnp[0], yxx[0], rnp[2], yxx[1], eep, yxx[2]])

    x, y = gbvpsolver(lambda x: rho, fyp, fy0, frt, p0, ivp_method = ivp_method, rt_method = rt_method, rt_tol = rt_tol)

    # inverting back final results from normalization
    for k in range(0, neq-2, 2):
        y[k, :] = rho0 * y[k, :]
    y[-1, :] = y[-1, :] / rho0
    return y

def flux_surface_plot(x, y, nplots = 8, plotdir = 'plots'):
    ''' (array, array, [int, str])

        Plots flux surfaces in 2D Cartesian co-ordinates (R,Z)

        INPUT:
        x = surface labels R1, output from vmoms (meter; 1d float array)
        y = R0, R0x, R2, R2x, E, Ex on x, output from vmoms (R in meter, E unitless; 2d float array)
        nplots = number of contours/plots (int; optional)
        plotdir = directory in which the flux surface plot is saved (str; optional)
    '''
    if plotdir[-1] == '/' or plotdir[-1] == '\\':
        plotdir = plotdir[:-1]
    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)
    plt.clf()
    plt.tight_layout()
    ny = len(y[0])
    nplots = min(nplots, ny)
    rr = np.array([[y[0][k] - x[k] * np.cos(t) + y[2][k] * np.cos(2*t) for t in np.linspace(0, 2*np.pi, 65)] for k in range(ny)])
    zz = np.array([[y[4][k] * (x[k] * np.sin(t) + y[2][k] * np.sin(2*t)) for t in np.linspace(0, 2*np.pi, 65)] for k in range(ny)])
    cmap = plt.get_cmap('brg')
    colors = [cmap(i) for i in np.linspace(1, 0, nplots)]
    k = 0
    plt.axes().set_aspect('equal')
    plt.xlabel('R (m)')
    plt.ylabel('Z (m)')
    js = [int(round(i)+0.1) for i in np.linspace(0, len(rr)-1, nplots)]
    for j in js:
        plt.plot(rr[j,:], zz[j,:], color = colors[k])
        k += 1
    plt.savefig(plotdir + '/Flux Surfaces.png', dpi = 200, bbox_inches = 'tight')

def other_variables(rmaj, bt0, x, y, presfx, prespfx, curifx, curipfx, iplot = True, plotdir = 'plots'):
    ''' (float, float, array, array, function, function, function, function, [bool, str]) --> several arrays

        Computes several relevant variables using results of vmoms()

        INPUT:
        rmaj = Major Radius (meter; float)
        bt0 = toroidal magnetic field at rmaj (tesla; float)
        x = surface labels R1, output from vmoms (meter; 1d float array)
        y = R0, R0x, R2, R2x, E, Ex on x, output from vmoms (R in meter, E unitless; 2d float array)
        presfx = Pressure Function, calculatable on x (pascal; single-input function)
        prespfx = Pressure Derivative Function, calculatable on x (pascal/meter; single-input function)
        curifx = Current Function, calculatable on x (ampere; single-input function)
        curipfx = Current Derivative Function, calculatable on x (ampere/meter; single-input function)
        iplot = whether to plot and save all output variables as png (bool; optional)
        plotdir = directory in which all the plots are saved (str; optional)

        OUTPUT:
        trng = Geometrical Triangularity of flux surfaces (unitless; 1d float array)
        elip = Geometrical Ellipticity of flux surfaces (unitless; 1d float array)
        srfc = Surface Area of flux surfaces (meter**2; 1d float array)
        volx = Derivative of volume w.r.t. x (meter**2; 1d float array)
        vol = Volume of flux surfaces (m**3; 1d float array)
        psix = Derivative of Poloidal Flux in a poloidal plane (tesla * meter; 1d float array)
        psi = Poloidal Flux in a poloidal plane (tesla * meter**2; 1d float array)
        cflx = Current Flux Function F in a poloidal plane (ampere; 1d float array)
        phix = Derivative of Toroidal Flux in a poloidal plane (tesla * meter; 1d float array)
        phi = Toroidal Flux in a poloidal plane (tesla * meter**2; 1d float array)
        q = Safety Factor (unitless; 1d float array)
        jeqv = Equivalent Current Density (ampere / meter**2; 1d float array)
        betat = beta-toroidal (unitless; float)
        betap = beta-poloidal (unitless; float)
        betan = beta-normalized (unitless; float)
        # iind = internal inductance (unitless; 1d float array)
    '''
    mu0 = 4.0e-7 * np.pi
    tpi = 2.0 * np.pi
    fpi2 = tpi**2
    xz = np.append(0, x)
    presdt = presfx(x)
    prespdt = prespfx(x)
    curidt = curifx(x)
    curipdt = curipfx(x)

    ntheta = 257
    th = np.linspace(0, np.pi, ntheta)
    dth = th[1] - th[0]
    cs1t = np.cos(th)
    cs2t = np.cos(2.0 * th)
    sn1t = np.sin(th)
    sn2t = np.sin(2.0 * th)

    nx = len(x)
    asg = np.empty(nx)
    asgr2 = np.empty(nx)
    agttsg = np.empty(nx)
    srfc = np.empty(nx)
    trng = np.empty(nx)
    elip = np.empty(nx)
    r = np.empty((nx, ntheta))
    sg = np.empty((nx, ntheta))
    for i in range(nx):
        rn = np.array([y[0,i], x[i], y[2,i]])
        rnp = np.array([y[1,i], 1.0, y[3,i]])
        ee = y[4,i]
        eep = y[5,i]
        r[i,:] = rn[0] - rn[1] * cs1t + rn[2] * cs2t
        rt = rn[1] * sn1t - 2.0 * rn[2] * sn2t
        rp = rnp[0] - rnp[1] * cs1t + rnp[2] * cs2t
        zt = ee * (rn[1] * cs1t + 2.0 * rn[2] * cs2t)
        zp = eep * (rn[1] * sn1t + rn[2] * sn2t) + ee * (rnp[1] * sn1t + rnp[2] * sn2t)
        sg[i,:] = r[i,:] * (rt * zp - rp * zt)
        gtt = rt**2 + zt**2
        asg[i] = thavg(sg[i,:], dth)
        asgr2[i] = thavg(sg[i,:] / r[i,:]**2, dth)
        agttsg[i] = thavg(gtt / sg[i,:], dth)
        srfc[i] = fpi2 * thavg(r[i,:] * np.sqrt(gtt), dth)
        trng[i], elip[i] = geotrn2(x[i], y[2, i], y[4, i])

    volx = fpi2 * asg
    vol = scipy.integrate.cumtrapz(np.append(0, volx), xz)    # assuming volx(x=0)=0 and vol(x=0)=0

    psix = -mu0 * curidt / (tpi * agttsg)
    psi = scipy.integrate.cumtrapz(np.append(0, psix), xz)    # assuming psix(x=0)=0 and psi(x=0)=0

    cflx0 = bt0 * rmaj / mu0
    cflx2x = -2.0 * (curidt * curipdt / agttsg / fpi2 + prespdt * asg / mu0) / asgr2
    cflx2 = scipy.integrate.cumtrapz(cflx2x, x)
    cflx2 = cflx0**2 - cflx2[-1] + cflx2
    cflx2 = np.append(cflx2x[1] * 0.5 * (x[0]**2 - x[1]**2) / x[1] + cflx2[0], cflx2)    # first value calculated with quadratic fit assuming cflx2x(x=0)=0 and using values at x[1]
    cflx = np.sqrt(cflx2)

    phix = cflx * mu0 * asgr2
    phi = scipy.integrate.cumtrapz(np.append(0, phix), xz)    # assuming phix(x=0)=0 and phi(x=0)=0

    q = tpi * cflx * asgr2 * agttsg / curidt

    jphi = np.empty((nx, ntheta))
    #btor = np.empty((nx, ntheta))
    jeqv = np.empty(nx)
    for i in range(nx):
        jphi[i,:] = -(r[i,:] * prespdt[i] + mu0 * cflx2x[i] * 0.5 / r[i,:]) * tpi * agttsg[i] / mu0 / curidt[i]
        #btor[i,:] = cflx[i] * mu0 / r[i,:]    # needs verification
        sgr = sg[i,:]  / r[i,:]
        jeqv[i] = thavg(jphi[i,:] * sgr, dth) / thavg(sgr, dth)

    # Following formulas of beta match against original VMOMS but still need to be checked
    curi0 = np.max(curidt)
    casg = scipy.integrate.cumtrapz(np.append(0, asg), xz)
    spasg = scipy.integrate.trapz(np.append(0, presdt * asg), x = xz)
    betap = spasg * asg[-1] * agttsg[-1] * 8.0 * np.pi**2 / curi0**2 / mu0 / casg[-1]
    betat = spasg * 2.0 * mu0 / bt0**2 / casg[-1]
    betan = betat * 100 * x[-1] * bt0 * 1.0e6 / curi0

    # Following calculates internal inductance but needs verification
    #curidt2 = curidt**2
    #iind = asg * agttsg * scipy.integrate.cumtrapz(np.append(0, curidt2 / agttsg), xz) / casg / curidt2

    if iplot:
        if plotdir[-1] == '/' or plotdir[-1] == '\\':
            plotdir = plotdir[:-1]
        if not os.path.isdir(plotdir):
            os.mkdir(plotdir)
        pdpi = 200
        plotter(x/x[-1], trng, xlbl = 'x', xunit = '', ttl = 'Geometric Triangularity', ylbl = 'δ', yunit = '', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], elip, xlbl = 'x', xunit = '', ttl = 'Geometric Ellipticity', ylbl = 'κ', yunit = '', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], srfc, xlbl = 'x', xunit = '', ttl = 'Surface Area', ylbl = 'A', yunit = 'meter**2', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], volx, xlbl = 'x', xunit = '', ttl = 'Volume Derivative', ylbl = "V'", yunit = 'meter**2', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], vol, xlbl = 'x', xunit = '', ttl = 'Volume', ylbl = 'V', yunit = 'meter**3', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], psix, xlbl = 'x', xunit = '', ttl = 'Poloidal Flux Derivative', ylbl = "ψ'", yunit = 'tesla * meter', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], psi, xlbl = 'x', xunit = '', ttl = 'Poloidal Flux', ylbl = 'ψ', yunit = 'tesla * meter**2', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], cflx, xlbl = 'x', xunit = '', ttl = 'Current Flux Function', ylbl = 'F', yunit = 'ampere', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], phix, xlbl = 'x', xunit = '', ttl = 'Toroidal Flux Derivative', ylbl = "ϕ'", yunit = 'tesla * meter', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], phi, xlbl = 'x', xunit = '', ttl = 'Toroidal Flux', ylbl = 'ϕ', yunit = 'tesla * meter**2', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], q, xlbl = 'x', xunit = '', ttl = 'Safety Factor', ylbl = 'Q', yunit = '', fldr = plotdir, frmt = 'png', dpi = pdpi)
        plotter(x/x[-1], jeqv, xlbl = 'x', xunit = '', ttl = 'Current Density Equivalent', ylbl = 'Jeqv', yunit = 'ampere / meter**2', fldr = plotdir, frmt = 'png', dpi = pdpi)
        #plotter(x/x[-1], iind, xlbl = 'x', xunit = '', ttl = 'Internal Inductance', ylbl = 'Li', yunit = '', fldr = 'plots', frmt = 'png', dpi = pdpi)

    return trng, elip, srfc, volx, vol, psix, psi, cflx, phix, phi, q, jeqv, betat, betap, betan

def tex_report():
    return

if __name__ == '__main__':

    xdt = np.linspace(0, 2, 1024)
    curidt = 1.5e7 * (1.0 - (1.0 - xdt**2 * 0.25)**2)
    presdt = ((15.0 - 5) * (1 - xdt**2 * 0.25)**2 + 5) * ((1.5 - 0.015) * (1 - xdt**2 * 0.25)**2 + 0.015) * 1.602176487e-16 * 1.0e20

    presfx = scipy.interpolate.CubicSpline(xdt, presdt, bc_type = ((1, 0.0), (1, 0.0)))
    curifx = scipy.interpolate.CubicSpline(xdt, curidt, bc_type = ((1, 0.0), (1, 0.0)))
    prespfx = lambda x: presfx(x, 1)
    curipfx = lambda x: curifx(x, 1)

    nrho = 128
    x1 = 2.0
    drho = x1 / nrho
    rhodt = np.linspace(drho, x1, nrho)

    rmaj = 6.2
    triang = 0.3
    ellip = 1.5
    bt0 = 5.3
    p0in = np.array([0.1, 0.01, 1.0])

    y = vmoms(rmaj, triang, ellip, rhodt, prespfx, curifx, curipfx, p0in, ivp_method = 'RK45', rt_method = 'lm', rt_tol = 1.0e-10)

    other_variables(rmaj, bt0, rhodt, y, presfx, prespfx, curifx, curipfx, iplot = True, plotdir = 'plots')

    flux_surface_plot(rhodt, y, nplots = 12, plotdir = 'plots')
